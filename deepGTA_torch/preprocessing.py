import numpy as np
from scipy.interpolate import interp1d

def load_surface_xplorer_csv(filepath: str):
    """
    Loads TA data from a Surface Xplorer CSV file.
    Returns:
        time_axis (np.ndarray), wavelength_axis (np.ndarray), data_matrix (np.ndarray), metadata (dict)
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
        
    data_lines = []
    metadata = {}
    
    for line in lines:
        if not line.strip():
            continue
        if line.strip()[0].isalpha() or line.startswith('file info'):
            parts = line.split(':', 1)
            if len(parts) == 2:
                metadata[parts[0].strip()] = parts[1].strip()
        else:
            data_lines.append(line)
            
    # Parse data matrix
    matrix = []
    for line in data_lines:
        try:
            row = [float(x) for x in line.split(',') if x.strip()]
            if row:
                matrix.append(row)
        except ValueError:
            pass
            
    d = np.array(matrix)
    t = d[0, 1:]
    wl = d[1:, 0]
    data = d[1:, 1:]
    
    # Handle NaNs by 1D interpolation along wavelength axis
    for i in range(data.shape[0]):
        row = data[i, :]
        mask = np.isnan(row)
        if mask.any():
            valid_idx = np.where(~mask)[0]
            if len(valid_idx) > 0:
                row[mask] = np.interp(np.where(mask)[0], valid_idx, row[valid_idx])
            else:
                row[mask] = 0.0
                
    return t, wl, data, metadata

def preprocess_for_cnn(t: np.ndarray, wl: np.ndarray, data: np.ndarray, target_shape=(256, 64)):
    """
    Interpolates data to CNN input shape and normalizes to [0,1].
    Expected input shape: data(wl, t). Output shape: (1, 256, 64) -> (C, H, W)
    where H is time and W is wavelength.
    """
    # 1. interpolate time axis to 256 points log-spaced
    t_min = max(0.1, t[np.where(t > 0)[0][0]]) if (t > 0).any() else 0.1
    t_max = max(1000.0, t[-1])
    new_t = np.logspace(np.log10(t_min), np.log10(t_max), target_shape[0])
    
    interpolator = interp1d(t, data, axis=1, kind='linear', fill_value='extrapolate')
    data_t = interpolator(new_t)
    
    # 2. interpolate wavelength axis to 64 points
    new_wl = np.linspace(wl[0], wl[-1], target_shape[1])
    interpolator_wl = interp1d(wl, data_t, axis=0, kind='linear', fill_value='extrapolate')
    data_twl = interpolator_wl(new_wl)
    
    # Transpose so rows are time and cols are wavelength
    data_twl = data_twl.T
    
    # Normalize to [0,1] or [-1,1]? The original code did data/max(data).
    data_max = np.max(np.abs(data_twl))
    if data_max > 0:
        data_twl = data_twl / data_max
        
    return data_twl.reshape(1, target_shape[0], target_shape[1]).astype(np.float32)
