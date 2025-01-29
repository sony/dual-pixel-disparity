import numpy as np
import torch
import torch.nn.functional as F

MAX_8BIT = 255.
MAX_16BIT = 65535.

def rescale_image(image: torch.Tensor, scale_factor: float, mode: str):
    height, width = image.shape[2:]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    rescaled_image = F.interpolate(image, size=(new_height, new_width), mode=mode)
    return rescaled_image

def calculate_coc_scale(focus_dis, f_stop, coc_alpha, focal_len, pixel_pitch):
    lens_sensor_dis = focal_len * focus_dis / (focus_dis - focal_len)
    lens_dia = focal_len / f_stop
    return lens_sensor_dis * lens_dia / focus_dis * coc_alpha / pixel_pitch

# input : phase_normalized [0~1]
# output: depth_normalized [0~1]
# all length parameter unit: mm
def phase_to_depth_normalized(phase_normalized, focus_dis, f_stop, coc_alpha, focal_len, pixel_pitch, phase_min, phase_max, depth_min, depth_max):
    phase = phase_normalized * (phase_max - phase_min) + phase_min
    coc_scale = calculate_coc_scale(focus_dis, f_stop, coc_alpha, focal_len, pixel_pitch)
    denom = coc_scale - phase

    if isinstance(phase_normalized, np.ndarray):
        depth_mm = np.where(denom <= 0, depth_max * 1000, (focus_dis * coc_scale) / denom)
    elif torch.is_tensor(phase_normalized):
        depth_mm = torch.where(denom <= 0, torch.full_like(denom, depth_max * 1000), (focus_dis * coc_scale) / denom)
    else:
        raise ValueError("phase_normalized should be either a numpy array or a PyTorch tensor")

    depth_normalized = (depth_mm / 1000 - depth_min) / (depth_max - depth_min) # [0~1]

    if isinstance(depth_normalized, np.ndarray):
        depth_normalized = np.clip(depth_normalized, 0, 1)
    elif torch.is_tensor(depth_normalized):
        depth_normalized = torch.clamp(depth_normalized, 0, 1)
    else:
        raise ValueError("depth_normalized should be either a numpy array or a PyTorch tensor")   

    return depth_normalized

# input : depth_normalized [0~1]
# output: phase_normalized [0~1]
def depth_to_phase(depth_normalized, focus_dis, f_stop, coc_alpha, focal_len, pixel_pitch, phase_min, phase_max, depth_max, depth_min, add_noise=False, noise_type='efmount', sigma_max=5.0, sigma_gain=1.0):
    depth = depth_normalized * 1000 * (depth_max - depth_min) + depth_min
    coc_scale = calculate_coc_scale(focus_dis, f_stop, coc_alpha, focal_len, pixel_pitch)
    phase = coc_scale * (depth - focus_dis) / np.clip(depth, 0.001, None)

    if add_noise:
        if noise_type=='gauss':
            noise = np.random.normal(0, sigma_max, depth.shape).astype(np.float32)
            noise[depth <= 100] = 0  # Set noise to zero for pixels with depth <= 100mm
            phase = phase + noise
        else:
            # Add Laplacian noise        
            sigma = calc_sigma(depth / 1000, f_stop, focus_dis / 1000)
            sigma = np.clip(sigma*sigma_gain, None, sigma_max)
            b = sigma / np.sqrt(2)
            noise = np.random.laplace(scale=b, size=depth.shape)
            noise[depth <= 100] = 0  # Set noise to zero for pixels with depth <= 100mm
            phase = phase + noise

    phase = np.clip(phase, phase_min, phase_max)

    phase_normalized = np.clip(phase, phase_min, phase_max)
    phase_normalized = (phase_normalized - phase_min) / (phase_max - phase_min)

    return phase_normalized

# def calc_sigma(z, F, d):
#     z = np.clip(z, 0.001, None)
#     a = 0.481017921777839
#     b = 1.3851472925386
#     return (b*np.exp((z*(np.log((z*(a/(d/((F)**(-1))))))/b))))
# \[ b \cdot \left(\frac{z \cdot a}{d \cdot F}\right)^{z/b} \]

def calc_sigma(z, F, d):
    z = np.clip(z, 0.001, None)
    c1 = 6.93
    c2 = 0.48
    c3 = 1.39
    return (c1 * (c2*z/(F*d))**(z/c3))
