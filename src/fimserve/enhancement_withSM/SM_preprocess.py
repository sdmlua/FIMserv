import torch
import rasterio
import numpy as np
from pathlib import Path
import torch.nn as nn
from scipy.stats import boxcox

GLOBAL_STATS = {
    'elevation': {'mean': 651.62, 'std': 935.30},
    'slope': {'mean': 1.65, 'std': 2.50},
    'flow_acc': {'mean': -101.88, 'std': 65.45},
    'twi': {'mean': -4.23, 'std': 2.27, 'min': -11.01, 'max': 25.39},
    'curve_number': {'mean': 73.96, 'std': 13.29, 'min': 30.12, 'max': 100.00},
    'soil_moisture': {'mean': 28.06, 'std': 9.94, 'min': 0.00, 'max': 76.13},
}

# INFERENCE DATA PREPROCESSOR
class InferenceDataPreprocessor:
    
    STATIC_FEATURES = ['curve_number', 'elevation', 'flow_acc', 'lulc', 'slope', 'soil_moisture', 'twi']
    LF_KEYWORDS = ['hand']

    FEATURE_FILENAME_MAP = {
        'curve_number': 'CN',
        'flow_acc': 'flowacc',
        'soil_moisture': 'SM',
        'elevation': 'elevation',
        'slope': 'slope',
        'twi': 'twi',
        'lulc': 'LULC'
    }

    def __init__(self, data_dir: Path, patch_size=(128, 128), global_stats=None, verbose=False):
        self.data_dir = Path(data_dir)
        self.M, self.N = patch_size
        self.verbose = verbose
        self.global_stats = global_stats if global_stats else GLOBAL_STATS

    def tif_to_tensor(self, path: Path, feature_name: str = None) -> torch.Tensor:
        with rasterio.open(path) as src:
            array = src.read(1).astype(np.float32)
            nodata_value = src.nodata
            if nodata_value is not None:
                array[array == nodata_value] = np.nan
            array = np.nan_to_num(array, nan=0.0)
            tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0)

            if feature_name == 'elevation':
                mean, std = self.global_stats['elevation'].values()
                tensor = (tensor - mean) / (std + 1e-7)
            elif feature_name == 'slope':
                tensor = self.apply_boxcox(tensor)
                mean, std = self.global_stats['slope'].values()
                tensor = (tensor - mean) / (std + 1e-7)
            elif feature_name == 'flow_acc':
                tensor = self.apply_boxcox(tensor)
                mean, std = self.global_stats['flow_acc'].values()
                tensor = (tensor - mean) / (std + 1e-7)
            elif feature_name == 'lulc':
                reclass_map = {1: 1, 2: 2, 4: 3, 3: 4, 8: 5, 6: 6, 7: 7, 5: 8, 9: 9}
                array = array.astype(np.int32)
                reclass_array = np.vectorize(lambda x: reclass_map.get(x, 0))(array).astype(np.float32)
                tensor = torch.tensor(reclass_array, dtype=torch.float32).unsqueeze(0)
            elif feature_name == 'low_fidelity':
                tensor = (tensor > 0).float()
            elif feature_name in self.global_stats:
                min_val = self.global_stats[feature_name]['min']
                max_val = self.global_stats[feature_name]['max']
                tensor = (tensor - min_val) / (max_val - min_val + 1e-7)
                
            return tensor

    def apply_boxcox(self, tensor: torch.Tensor, lmbda=0.5) -> torch.Tensor:
        tensor = tensor + 1e-6
        flat_np = tensor.flatten().numpy()
        transformed = boxcox(flat_np, lmbda=lmbda)
        return torch.tensor(transformed).reshape(tensor.shape)

    def patchify(self, data: torch.Tensor):
        C, H, W = data.shape
        stride_h = stride_w = self.M // 2
        pad_h = (stride_h * ((H - self.M) // stride_h + 1) + self.M - H) % stride_h
        pad_w = (stride_w * ((W - self.N) // stride_w + 1) + self.N - W) % stride_w
        padded = nn.functional.pad(data, (0, int(pad_w), 0, int(pad_h)), mode='constant', value=0)
        patches = padded.unfold(1, self.M, stride_h).unfold(2, self.N, stride_w)
        patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, C, self.M, self.N)
        return patches

    def get_static_stack(self, huc_id: str):
        tensors = []
        for feature in self.STATIC_FEATURES:
            search_key = self.FEATURE_FILENAME_MAP[feature]
            match = list(self.data_dir.glob(f"*{search_key}*{huc_id}*.tif"))
            if not match:
                print(f"Missing static feature: {feature} for {huc_id}")
                return None
            tensors.append(self.tif_to_tensor(match[0], feature_name=feature))
        return torch.cat(tensors, dim=0)

    def get_all_lf_maps(self, huc_id: str):
        return sorted([
            f for f in self.data_dir.glob(f"*{huc_id}*.tif")
            if any(k in f.name.lower() for k in self.LF_KEYWORDS)
        ])

    def preprocess_all_lf_maps(self, huc_id: str):
        static_stack = self.get_static_stack(huc_id)
        if static_stack is None:
            return []

        lf_files = self.get_all_lf_maps(huc_id)
        results = []

        for lf_path in lf_files:
            lf_tensor = self.tif_to_tensor(lf_path, feature_name='low_fidelity')
            combined = torch.cat([static_stack, lf_tensor], dim=0)
            patches = self.patchify(combined)
            results.append((lf_path.name, patches, lf_path))
            if self.verbose:
                print(f"Processed {lf_path.name} with {patches.shape[0]} patches.")

        return results
    
