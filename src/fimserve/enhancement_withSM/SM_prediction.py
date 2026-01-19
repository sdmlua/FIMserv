import gc
import tempfile
from pathlib import Path


from .SM_preprocess import *
from .surrogate_model import *
from .utlis import *
from .preprocessFIM import *
        

# MODEL LOADING
def load_model(model):
    """Downloads and loads the model checkpoint."""
    fs = s3fs.S3FileSystem(anon=True)
    bucket_path = "sdmlab/SM_dataset/trained_model/SM_trainedmodel.ckpt"

    with fs.open(bucket_path, 'rb') as s3file:
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp_ckpt:
            tmp_ckpt.write(s3file.read())
            tmp_ckpt_path = tmp_ckpt.name

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(tmp_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    return model, device


# HELPER FUNCTIONS
def create_weight_map(M: int, N: int, device):
    """Creates a Gaussian weight map for smooth patch merging."""
    # Create numpy array first
    weight_map = np.zeros((M, N), dtype=np.float32)
    center_x, center_y = M // 2, N // 2

    # Vectorized calculation
    Y, X = np.ogrid[:M, :N]
    dist_sq = (X - center_y)**2 + (Y - center_x)**2
    weight_map = np.exp(-dist_sq / (2 * (min(M, N) / 2)**2))

    # FIX: Only unsqueeze once to get shape (1, M, N)
    return torch.from_numpy(weight_map).float().unsqueeze(0).to(device)

def save_image(image: torch.Tensor, path: Path, reference_tif: str):
    """Saves the prediction tensor as a GeoTIFF."""
    image_np = image.squeeze().cpu().numpy().astype('float32')
    with rasterio.open(reference_tif) as ref:
        meta = ref.meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": image_np.shape[0],
            "width": image_np.shape[1],
            "count": 1,
            "dtype": 'float32'
        })

        with rasterio.open(path, 'w', **meta) as dst:
            dst.write(image_np, 1)

        # Apply water body mask
        mask_with_PWB(path, path)

        # Binarize and compress
        with rasterio.open(path, 'r+') as dst:
            data = dst.read(1)
            binary_data = np.where(data > 0, 1, 0).astype(np.uint8)
            dst.write(binary_data, 1)

        compress_tif_lzw(path)

# REDICTION
def predict_optimized(dataset, model, shape: torch.Tensor, M: int = 256, N: int = 256,
                      stride: int = 128, device=None, batch_size=32):
    """
    Highly optimized prediction loop.
    - Uses VIEWs instead of COPIES for memory efficiency.
    - Performs on-the-fly padding.
    - Streams batches to GPU while keeping the main map on CPU.
    """

    # SETUP INPUTS
    if isinstance(dataset.x_feature_index, slice):
        X = shape[dataset.x_feature_index]
    elif sorted(dataset.x_feature_index) == list(range(shape.shape[0])):
        X = shape[:]
    else:
        print("Warning: Creating tensor copy for non-contiguous indices (High RAM usage)")
        X = shape[dataset.x_feature_index]

    y = shape[dataset.y_feature_index]

    img_channels, img_rows, img_cols = X.shape

    # SETUP OUTPUT ACCUMULATORS (On CPU)
    weighted_prediction_sum = torch.zeros((1, img_rows, img_cols), dtype=torch.float32, device='cpu')
    weight_sum = torch.zeros((1, img_rows, img_cols), dtype=torch.float32, device='cpu')

    # Weight map: Shape (1, M, N)
    weight_map_gpu = create_weight_map(M, N, device)
    weight_map_cpu = weight_map_gpu.cpu()

    # BATCH PROCESSING LOOP
    batch_patches = []
    batch_coords = []

    total_steps = ((img_rows - 1) // stride + 1) * ((img_cols - 1) // stride + 1)
    processed_steps = 0

    print(f"   Starting inference on {img_rows}x{img_cols} image...")

    for r in range(0, img_rows, stride):
        for c in range(0, img_cols, stride):
            r_end = min(r + M, img_rows)
            c_end = min(c + N, img_cols)

            h_valid = r_end - r
            w_valid = c_end - c

            # Extract patch from CPU tensor (View)
            patch = X[:, r:r_end, c:c_end]

            # Handle Boundary Padding (On-the-fly)
            if h_valid < M or w_valid < N:
                pad_h = M - h_valid
                pad_w = N - w_valid
                patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='constant', value=0)

            batch_patches.append(patch)
            batch_coords.append((r, r_end, c, c_end, h_valid, w_valid))

            # INFERENCE STEP
            if len(batch_patches) >= batch_size:
                batch_tensor = torch.stack(batch_patches).to(device)

                with torch.no_grad():
                    preds = model(batch_tensor).cpu()

                # Accumulate
                for k, (r0, r1, c0, c1, h_val, w_val) in enumerate(batch_coords):
                    pred_valid = preds[k, :, :h_val, :w_val]
                    weight_valid = weight_map_cpu[:, :h_val, :w_val]

                    weighted_prediction_sum[:, r0:r1, c0:c1] += pred_valid * weight_valid
                    weight_sum[:, r0:r1, c0:c1] += weight_valid

                processed_steps += len(batch_patches)
                print(f"   Progress: {processed_steps}/{total_steps} ({100*processed_steps/total_steps:.1f}%)", end='\r')

                batch_patches = []
                batch_coords = []
                del batch_tensor, preds

                if processed_steps % (batch_size * 10) == 0:
                     torch.cuda.empty_cache()

    # Process remaining patches
    if batch_patches:
        batch_tensor = torch.stack(batch_patches).to(device)
        with torch.no_grad():
            preds = model(batch_tensor).cpu()

        for k, (r0, r1, c0, c1, h_val, w_val) in enumerate(batch_coords):
            pred_valid = preds[k, :, :h_val, :w_val]
            weight_valid = weight_map_cpu[:, :h_val, :w_val]
            weighted_prediction_sum[:, r0:r1, c0:c1] += pred_valid * weight_valid
            weight_sum[:, r0:r1, c0:c1] += weight_valid

        print(f"   Progress: 100% - Inference Complete.")

    # NORMALIZE AND FINALIZE
    epsilon = 1e-8
    final_prediction = weighted_prediction_sum / (weight_sum + epsilon)
    final_prediction = (final_prediction > 0.01).float()

    lf_idx = dataset.lf_index
    if isinstance(shape, torch.Tensor):
         lf = shape[lf_idx].unsqueeze(0)
    else:
         lf = None

    return final_prediction, y, lf

# MAIN FUNCTION
def enhanceFIM(huc_id, patch_size=(256, 256), batch_size=32):

    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}\nSYSTEM: {device_type.upper()}\n{'='*60}")

    data_dir = Path(f'./HUC{huc_id}_forcings/')
    model = AttentionUNet(channel=8)
    preprocessor = InferenceDataPreprocessor(data_dir=Path(data_dir), patch_size=patch_size, verbose=True)

    print("Loading model...")
    model, device = load_model(model)

    lf_files = preprocessor.get_all_lf_maps(huc_id)

    for idx, lf_path in enumerate(lf_files, 1):
        lf_filename = lf_path.name
        print(f"\nProcessing [{idx}/{len(lf_files)}]: {lf_filename}")

        static_stack = preprocessor.get_static_stack(huc_id)
        lf_tensor = preprocessor.tif_to_tensor(lf_path, feature_name='low_fidelity')

        print("Merging tensors...")
        area_tensor = torch.cat([static_stack, lf_tensor], dim=0)

        del static_stack
        del lf_tensor
        gc.collect()

        print(f"Tensor Shape: {area_tensor.shape} | Memory: {area_tensor.element_size() * area_tensor.nelement() / 1e9:.2f} GB")

        class Dummy:
            x_feature_index = slice(None)
            y_feature_index = [area_tensor.shape[0] - 1]
            lf_index = area_tensor.shape[0] - 1

        try:
            x, y, lf = predict_optimized(
                Dummy,
                model,
                area_tensor,
                M=patch_size[0],
                N=patch_size[1],
                stride=patch_size[0] // 2,
                device=device,
                batch_size=batch_size
            )

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("OOM Error. Retrying with batch_size=4...")
                torch.cuda.empty_cache()
                gc.collect()
                x, y, lf = predict_optimized(
                    Dummy, model, area_tensor, M=patch_size[0], N=patch_size[1],
                    stride=patch_size[0] // 2, device=device, batch_size=4
                )
            else:
                raise e

        del area_tensor
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        pred_dir = Path(f"./Results/HUC{huc_id}/")
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_path = pred_dir / f"SMprediction_{lf_filename}"

        save_image(x, pred_path, str(lf_path))
        print(f"✓ Saved: {pred_path}")

    print(f"\n{'='*60}\nCOMPLETED\n{'='*60}")