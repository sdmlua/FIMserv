from SM_preprocess import *
from surrogate_model import *
from utlis import *
from preprocessFIM import *

def load_model(model):
    # Set up S3 access
    fs = s3fs.S3FileSystem(anon=True)
    bucket_path = "sdmlab/SM_dataset/trained_model/SM_trainedmodel.ckpt"

    # Download to a temporary file
    with fs.open(bucket_path, 'rb') as s3file:
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp_ckpt:
            tmp_ckpt.write(s3file.read())
            tmp_ckpt_path = tmp_ckpt.name

    # Load checkpoint
    checkpoint = torch.load(tmp_ckpt_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(checkpoint['state_dict'])

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    return model, device

#WEIGHTED AVERAGE for patch
def create_weight_map(M: int, N: int):
    weight_map = np.zeros((M, N), dtype=np.float32)
    center_x, center_y = M // 2, N // 2
    for i in range(M):
        for j in range(N):
            dist_sq = (i - center_x)**2 + (j - center_y)**2
            weight = np.exp(-dist_sq / (2 * (min(M, N) / 2)**2))
            weight_map[i, j] = weight
    return torch.from_numpy(weight_map).float().unsqueeze(0).unsqueeze(0) # (1, 1, M, N)

#If there is Stride
def predict_on_area(dataset, model, shape: torch.Tensor, M: int = 256, N: int = 256, stride: int = 128, device=None):
    # Get row and col size
    shape_row = shape.size(1)
    shape_col = shape.size(2)

    # Pad if needed
    pad_h = (stride * ((shape_row - M) // stride + 1) + M - shape_row)
    pad_w = (stride * ((shape_col - N) // stride + 1) + N - shape_col)

    if pad_h > 0 or pad_w > 0:
        padding = (0, pad_w, 0, pad_h) 
        shape = nn.functional.pad(shape, padding, mode='constant', value=0)

    # Update new shape after padding
    new_row = shape.size(1)
    new_col = shape.size(2)

    # Separate X and Y
    X = shape[dataset.x_feature_index]
    y = shape[dataset.y_feature_index]

    # Initialize weighted prediction sum and weight sum arrays
    weighted_prediction_sum = torch.zeros((1, new_row, new_col), device=device)
    weight_sum = torch.zeros((1, new_row, new_col), device=device)

    # Create the weight map
    weight_map = create_weight_map(M, N).to(device)

    # Loop over patches
    for start_i in range(0, new_row - M + 1, stride):
        for start_j in range(0, new_col - N + 1, stride):
            end_i = start_i + M
            end_j = start_j + N
            patch = X[:, start_i:end_i, start_j:end_j].unsqueeze(0).to(device)

            with torch.no_grad():
                patch_prediction_raw = model(patch)

            weighted_prediction = patch_prediction_raw * weight_map
            weighted_prediction_sum[:, start_i:end_i, start_j:end_j] += weighted_prediction.squeeze(0) 
            weight_sum[:, start_i:end_i, start_j:end_j] += weight_map.squeeze(0)

    epsilon = 1e-8
    final_prediction = weighted_prediction_sum / (weight_sum + epsilon)
    final_prediction = (final_prediction > 0.01).float() 

    # Crop back to original shape (before padding)
    final_prediction = final_prediction[:, :shape_row, :shape_col]
    y = y[:, :shape_row, :shape_col]
    lf = shape[[dataset.lf_index]][:, :shape_row, :shape_col]

    return final_prediction.cpu(), y.cpu(), lf.cpu()

#Save the tif file 
def save_image(image: torch.Tensor, path: Path, reference_tif: str):
    """Save the image as a .tif file.
    
    Args:
        image (torch.Tensor): The image to save
        path (Path): The path to save the image
    """
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
        mask_with_PWB(path, path)

        with rasterio.open(path, 'r+') as dst:
            data = dst.read(1)
            binary_data = np.where(data > 0, 1, 0).astype(np.uint8)
            dst.write(binary_data, 1)

        compress_tif_lzw(path)
        
        
#ENHANCE THE LOW-FIDELITY FLOOD MAP
def Predict_FM(huc_id, patch_size=(256, 256)):
    
    data_dir = Path(f'./HUC{huc_id}_forcings/')
    model =  AttentionUNet(channel=8)
    
    preprocessor = InferenceDataPreprocessor(data_dir=Path(data_dir), patch_size=patch_size, verbose=True)
    
    print("Loading model...")
    model, device = load_model(model)
    print("Model loaded.")
    
    
    lf_files = preprocessor.get_all_lf_maps(huc_id)
    for lf_path in lf_files:
        lf_filename = lf_path.name
        print(f"Predicting for: {lf_filename}\n")

        print(f"Loading static features for HUC {huc_id}...")
        static_stack = preprocessor.get_static_stack(huc_id)
        lf_tensor = preprocessor.tif_to_tensor(lf_path, feature_name='low_fidelity')

        # Combine and validate
        area_tensor = torch.cat([static_stack, lf_tensor], dim=0)
        if area_tensor.shape[0] != 8:
            raise ValueError(f"Expected 8 channels, got {area_tensor.shape[0]} — check missing static feature for HUC {huc_id}.")

        # Define dummy interface
        class Dummy:
            x_feature_index = list(range(area_tensor.shape[0])) 
            y_feature_index = [area_tensor.shape[0] - 1]     
            lf_index = area_tensor.shape[0] - 1

        print(f"Static features loaded for {huc_id}.\n")
        
        # Predict
        print(f"Enhancing {lf_path}...")
        x, y, lf = predict_on_area(Dummy, model, area_tensor, M=patch_size[0], N=patch_size[1], stride=patch_size[0] // 2, device=device)

        # Save result
        pred_dir = Path(f"./Results/HUC{huc_id}/")
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_path = pred_dir / f"SMprediction_{lf_filename}"
        save_image(x, pred_path, lf_path)
        print(f"Enhancement completed for {lf_filename}.\n")
        

