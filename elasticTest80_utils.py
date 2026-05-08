# elasticTest80_utils.py

import torch, math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from disp_utils7 import FPhaseshift

# =============================================================================
# 第一部分：基础工具函数
# =============================================================================
def ricker_differentiable(freq, length, dt, peak_time):
    """
    支持 Autograd 和 GPU 的 Ricker 子波生成函数。
    解决 Deepwave 原生函数无法处理 GPU Tensor 输入的问题。
    """
    # 确保 freq 是张量
    if not torch.is_tensor(freq):
        freq = torch.tensor(freq)
        
    # 【关键修改】获取 freq 所在的设备，并在同设备上创建时间向量 t
    device = freq.device
    
    # 注意：这里我们直接用 freq 的 dtype 和 device
    t = torch.arange(length, device=device, dtype=freq.dtype) * dt
    t = t - peak_time
    
    # Ricker 公式
    pi2 = math.pi ** 2
    f2 = freq ** 2
    t2 = t ** 2
    
    x = pi2 * f2 * t2
    y = (1 - 2 * x) * torch.exp(-x)
    
    return y

def gaussian_blur_2d(img, sigma):
    """快速 2D 高斯模糊"""
    if sigma <= 0.1:
        return img
    k_size = int(4 * sigma + 1) | 1
    x = torch.arange(k_size, device=img.device).float() - k_size // 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    k_x = kernel_1d.view(1, 1, 1, k_size)
    k_y = kernel_1d.view(1, 1, k_size, 1)
    
    img_in = img.unsqueeze(0).unsqueeze(0)
    img_blur = F.conv2d(img_in, k_y, padding=(k_size//2, 0))
    img_blur = F.conv2d(img_blur, k_x, padding=(0, k_size//2))
    
    return img_blur.squeeze()


def gaussian_blur_v_axis(img, sigma):
    """沿速度轴的 1D 高斯模糊"""
    if sigma <= 0.01:
        return img
    k_size = int(6 * sigma + 1) | 1
    x = torch.arange(k_size, dtype=torch.float32, device=img.device) - k_size // 2
    kernel = torch.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, k_size, 1)
    img_in = img.unsqueeze(0).unsqueeze(0)
    pad = k_size // 2
    return F.conv2d(img_in, kernel, padding=(pad, 0)).squeeze()


def gaussian_smooth_2d(img, sigma):
    """2D 高斯平滑"""
    if sigma <= 0.01:
        return img
    k_size = int(4 * sigma + 1) | 1
    x = torch.arange(k_size, device=img.device).float() - k_size // 2
    y = torch.arange(k_size, device=img.device).float() - k_size // 2
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    img_in = img.unsqueeze(0).unsqueeze(0)
    return F.conv2d(img_in, kernel, padding=k_size//2).squeeze()


def apply_gradient_taper(grad, width=10):
    """梯度边缘衰减"""
    nz, nx = grad.shape
    if width <= 0:
        return grad
    width = min(width, nx // 20, nz // 20)
    if width < 1:
        return grad
    mask = torch.ones_like(grad)
    ramp = torch.linspace(0, 1, width, device=grad.device)
    mask[:, :width] *= ramp.view(1, -1)
    mask[:, -width:] *= torch.flip(ramp, [0]).view(1, -1)
    mask[-width:, :] *= torch.flip(ramp, [0]).view(-1, 1)
    return grad * mask


# =============================================================================
# 第二部分：空间加权工具
# =============================================================================

def get_lateral_weight(nz, nx, mode='ramp', start_x_ratio=0.5, left_val=1.0, right_val=3.0, device='cpu'):
    start_x = int(nx * start_x_ratio)
    lat_w = torch.ones(nx, device=device) * left_val
    if mode == 'ramp' and start_x < nx:
        ramp = torch.linspace(left_val, right_val, nx - start_x, device=device)
        lat_w[start_x:] = ramp
    elif mode == 'step':
        lat_w[start_x:] = right_val
    return lat_w.view(1, -1).expand(nz, nx)


def get_power_depth_weight(nz, nx, exponent=4.0, scale=10.0, base=1.0, device='cpu'):
    z_idx = torch.linspace(0, 1, nz, device=device)
    weight = base + scale * (z_idx ** exponent)
    return torch.clamp(weight, 0.01, 50.0).view(-1, 1).expand(nz, nx)


def get_linear_depth_weight(nz, nx, slope=1.0, base=1.0, device='cpu'):
    z_idx = torch.linspace(0, 1, nz, device=device)
    weight = base + slope * z_idx
    return torch.clamp(weight, 0.1, 50.0).view(-1, 1).expand(nz, nx)


# =============================================================================
# 第三部分：频散谱计算
# =============================================================================

def compute_split_spectra(seis_gather, src_idx_val, gx, dx, time_tensor, fmax_calc, dt_val, 
                          min_traces=2, vmin=50, vmax=1600, fmin=1, vres=400, fres=400):
    """计算左右分离的频散谱"""
    mask_left = gx < src_idx_val
    mask_right = gx > src_idx_val
    cnt_left = int(mask_left.sum().item())
    cnt_right = int(mask_right.sum().item())

    spec_left, spec_right, f_axis = None, None, None

    if cnt_left >= min_traces:
        data_l = torch.flip(seis_gather[mask_left, :].T, dims=[1])
        off_l = torch.flip((gx[mask_left] - src_idx_val).float().abs() * dx, dims=[0])
        spec_left, f_axis, _ = FPhaseshift(
            data_l, off_l, time_tensor, 0, fmin, fmax_calc, vmin, vmax, fres, vres,
            pwsFlag=[1, 1]
        )

    if cnt_right >= min_traces:
        data_r = seis_gather[mask_right, :].T
        off_r = (gx[mask_right] - src_idx_val).float().abs() * dx
        spec_right, f_axis, _ = FPhaseshift(
            data_r, off_r, time_tensor, 0, fmin, fmax_calc, vmin, vmax, fres, vres,
            pwsFlag=[1, 1]
        )

    return spec_left, spec_right, f_axis, cnt_left, cnt_right


# =============================================================================
# 第四部分：统一的损失组件
# =============================================================================

PROC_SIZE = 64


class ResBlock(nn.Module):
    """残差块 - 更好的梯度流动"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        out = self.bn2(self.conv2(out))
        return F.leaky_relu(out + residual, 0.1)


class EnhancedSpectrumEncoder(nn.Module):
    """增强型频散谱编码器"""
    def __init__(self, feature_dim=64):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
        )
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            ResBlock(32),
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            ResBlock(64),
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            ResBlock(128),
        )
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.1),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.global_fc = nn.Sequential(
            nn.Linear(feature_dim * 16, feature_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        f0 = self.stem(x)
        f1 = self.enc1(f0)
        f2 = self.enc2(self.pool1(f1))
        f3 = self.enc3(self.pool2(f2))
        f4 = self.enc4(self.pool3(f3))
        
        g = self.global_pool(f4)
        g = g.view(g.size(0), -1)
        g = self.global_fc(g)
        
        return [f1, f2, f3, f4], g


# -----------------------------------------------------------------------------
# Loss 1: L1 Loss
# -----------------------------------------------------------------------------
class L1Loss(nn.Module):
    """L1损失"""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, obs):
        return F.l1_loss(pred, obs)


# -----------------------------------------------------------------------------
# Loss 2: L2 Loss
# -----------------------------------------------------------------------------
class L2Loss(nn.Module):
    """L2损失"""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, obs):
        return F.mse_loss(pred, obs)


# -----------------------------------------------------------------------------
# Loss 3: NCC Loss
# -----------------------------------------------------------------------------
class NCCLoss(nn.Module):
    """归一化互相关损失"""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, obs):
        p_c = pred - pred.mean()
        o_c = obs - obs.mean()
        p_std = torch.sqrt((p_c**2).sum()) + 1e-9
        o_std = torch.sqrt((o_c**2).sum()) + 1e-9
        ncc = (p_c * o_c).sum() / (p_std * o_std)
        return 1.0 - ncc


# -----------------------------------------------------------------------------
# Loss 4: Sliced Wasserstein Distance
# -----------------------------------------------------------------------------
class SWDLoss(nn.Module):
    """自适应Sliced Wasserstein Distance"""
    def __init__(self, n_projections=96):
        super().__init__()
        self.n_projections = n_projections
        
        angles = torch.linspace(0, np.pi, n_projections, dtype=torch.float32)
        directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        self.register_buffer('directions', directions)
        self.register_buffer('dir_weights', torch.ones(n_projections))
        
    def forward(self, pred, obs):
        device = pred.device
        H, W = pred.shape
        eps = 1e-9
        
        pred_prob = (pred + eps) / (pred.sum() + eps * H * W)
        obs_prob = (obs + eps) / (obs.sum() + eps * H * W)
        
        v_coords = torch.linspace(0, 1, H, device=device)
        f_coords = torch.linspace(0, 1, W, device=device)
        vv, ff = torch.meshgrid(v_coords, f_coords, indexing='ij')
        coords = torch.stack([vv.flatten(), ff.flatten()], dim=1)
        
        pred_flat = pred_prob.flatten()
        obs_flat = obs_prob.flatten()
        
        all_proj = coords @ self.directions.to(device).T
        
        distances = []
        for i in range(self.n_projections):
            proj = all_proj[:, i]
            sort_idx = torch.argsort(proj)
            
            pred_sorted = pred_flat[sort_idx]
            obs_sorted = obs_flat[sort_idx]
            
            pred_cdf = torch.cumsum(pred_sorted, dim=0)
            obs_cdf = torch.cumsum(obs_sorted, dim=0)
            
            w1 = torch.abs(pred_cdf - obs_cdf).mean()
            distances.append(w1)
        
        distances = torch.stack(distances)
        weighted_dist = (distances * self.dir_weights.to(device)).sum() / self.dir_weights.sum()
        
        return weighted_dist


# -----------------------------------------------------------------------------
# Loss 5: Energy Loss
# -----------------------------------------------------------------------------
class EnergyLoss(nn.Module):
    """自适应能量匹配损失"""
    def __init__(self):
        super().__init__()
        self.stage = 1
        
    def set_stage(self, stage):
        self.stage = stage
        
    def forward(self, pred, obs):
        H, W = pred.shape
        
        if self.stage <= 1:
            top_k_ratio = 0.15
        elif self.stage == 2:
            top_k_ratio = 0.25
        elif self.stage == 3:
            top_k_ratio = 0.35
        else:
            top_k_ratio = 0.45
        
        k = max(int(H * W * top_k_ratio), 20)
        
        obs_flat = obs.flatten()
        values, top_idx = torch.topk(obs_flat, k)
        
        pred_flat = pred.flatten()
        pred_top = pred_flat[top_idx]
        
        pred_norm = pred_top / (pred_top.max() + 1e-9)
        obs_norm = values / (values.max() + 1e-9)
        
        weights = torch.softmax(obs_norm * 5, dim=0)
        weighted_l1 = (torch.abs(pred_norm - obs_norm) * weights).sum()
        
        return weighted_l1


# -----------------------------------------------------------------------------
# Loss 6: Frequency Band Loss
# -----------------------------------------------------------------------------
class FreqBandLoss(nn.Module):
    """频带分离损失"""
    def __init__(self, n_bands=4):
        super().__init__()
        self.n_bands = n_bands
        
    def forward(self, pred, obs):
        H, W = pred.shape
        band_width = W // self.n_bands
        
        total_loss = 0.0
        band_weights = [1.5, 1.2, 1.0, 0.8]
        
        for i in range(self.n_bands):
            start = i * band_width
            end = (i + 1) * band_width if i < self.n_bands - 1 else W
            
            p_band = pred[:, start:end]
            o_band = obs[:, start:end]
            
            p_norm = p_band / (p_band.max() + 1e-9)
            o_norm = o_band / (o_band.max() + 1e-9)
            
            band_loss = F.l1_loss(p_norm, o_norm)
            
            w = band_weights[i] if i < len(band_weights) else 1.0
            total_loss = total_loss + w * band_loss
        
        return total_loss / self.n_bands


# -----------------------------------------------------------------------------
# Loss 7: SSIM Loss
# -----------------------------------------------------------------------------
class SSIMLoss(nn.Module):
    """结构相似性损失"""
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        
        sigma = 1.5
        x = torch.arange(window_size).float() - window_size // 2
        gauss = torch.exp(-x**2 / (2 * sigma**2))
        window_1d = gauss / gauss.sum()
        window_2d = window_1d.unsqueeze(1) @ window_1d.unsqueeze(0)
        self.register_buffer('window', window_2d.unsqueeze(0).unsqueeze(0))
        
    def forward(self, pred, obs):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        pred_4d = pred.unsqueeze(0).unsqueeze(0)
        obs_4d = obs.unsqueeze(0).unsqueeze(0)
        
        pad = self.window_size // 2
        window = self.window.to(pred.device)
        
        mu_p = F.conv2d(pred_4d, window, padding=pad)
        mu_o = F.conv2d(obs_4d, window, padding=pad)
        
        mu_p_sq = mu_p ** 2
        mu_o_sq = mu_o ** 2
        mu_po = mu_p * mu_o
        
        sigma_p_sq = F.conv2d(pred_4d ** 2, window, padding=pad) - mu_p_sq
        sigma_o_sq = F.conv2d(obs_4d ** 2, window, padding=pad) - mu_o_sq
        sigma_po = F.conv2d(pred_4d * obs_4d, window, padding=pad) - mu_po
        
        ssim = ((2 * mu_po + C1) * (2 * sigma_po + C2)) / \
               ((mu_p_sq + mu_o_sq + C1) * (sigma_p_sq + sigma_o_sq + C2))
        
        return 1.0 - ssim.mean()


# -----------------------------------------------------------------------------
# Loss 8: Gradient Loss
# -----------------------------------------------------------------------------
class GradLoss(nn.Module):
    """梯度匹配损失"""
    def __init__(self):
        super().__init__()
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', sobel_y.unsqueeze(0).unsqueeze(0))
        
    def forward(self, pred, obs):
        pred_4d = pred.unsqueeze(0).unsqueeze(0)
        obs_4d = obs.unsqueeze(0).unsqueeze(0)
        
        sobel_x = self.sobel_x.to(pred.device)
        sobel_y = self.sobel_y.to(pred.device)
        
        pred_gx = F.conv2d(pred_4d, sobel_x, padding=1)
        pred_gy = F.conv2d(pred_4d, sobel_y, padding=1)
        obs_gx = F.conv2d(obs_4d, sobel_x, padding=1)
        obs_gy = F.conv2d(obs_4d, sobel_y, padding=1)
        
        pred_mag = torch.sqrt(pred_gx**2 + pred_gy**2 + 1e-9)
        obs_mag = torch.sqrt(obs_gx**2 + obs_gy**2 + 1e-9)
        
        mag_loss = F.l1_loss(pred_mag, obs_mag)
        
        pred_dir = torch.stack([pred_gx, pred_gy], dim=-1)
        obs_dir = torch.stack([obs_gx, obs_gy], dim=-1)
        
        pred_dir_norm = pred_dir / (pred_dir.norm(dim=-1, keepdim=True) + 1e-9)
        obs_dir_norm = obs_dir / (obs_dir.norm(dim=-1, keepdim=True) + 1e-9)
        
        dir_sim = (pred_dir_norm * obs_dir_norm).sum(dim=-1)
        
        weight = (obs_mag > obs_mag.mean()).float()
        dir_loss = (1 - dir_sim) * weight
        dir_loss = dir_loss.sum() / (weight.sum() + 1e-9)
        
        return mag_loss + 0.5 * dir_loss


# -----------------------------------------------------------------------------
# Loss 9: Feature Loss
# -----------------------------------------------------------------------------
class FeatLoss(nn.Module):
    """多尺度特征匹配损失"""
    def __init__(self, feature_dim=64, device='cuda'):
        super().__init__()
        self.encoder = EnhancedSpectrumEncoder(feature_dim=feature_dim).to(device)
        self.proc_size = PROC_SIZE
        
    def preprocess(self, spec):
        spec_4d = spec.unsqueeze(0).unsqueeze(0)
        spec_down = F.interpolate(spec_4d, size=(self.proc_size, self.proc_size), 
                                   mode='bilinear', align_corners=True)
        spec_max = spec_down.max() + 1e-9
        spec_norm = spec_down / spec_max
        spec_log = torch.log1p(spec_norm * 20)
        spec_log = spec_log / (spec_log.max() + 1e-9)
        return spec_log
        
    def forward(self, pred, obs):
        pred_4d = self.preprocess(pred)
        obs_4d = self.preprocess(obs)
        
        pred_feats, pred_global = self.encoder(pred_4d)
        obs_feats, obs_global = self.encoder(obs_4d)
        
        total_loss = 0.0
        weights = [0.5, 1.0, 1.5, 2.0]
        
        for i, (p, o) in enumerate(zip(pred_feats, obs_feats)):
            w = weights[i] if i < len(weights) else 1.0
            
            p_norm = F.normalize(p.flatten(1), dim=1)
            o_norm = F.normalize(o.flatten(1), dim=1)
            
            sim = (p_norm * o_norm).sum(dim=1).mean()
            l2 = F.mse_loss(p, o)
            
            total_loss = total_loss + w * ((1.0 - sim) + 0.1 * l2)
        
        loss_global = F.mse_loss(pred_global, obs_global) + \
                      (1 - F.cosine_similarity(pred_global, obs_global, dim=0).mean())
        total_loss = total_loss + 0.5 * loss_global
        
        return total_loss / (sum(weights[:len(pred_feats)]) + 0.5)


# =============================================================================
# 第五部分：统一损失管理器 (模块定义)
# =============================================================================
PROC_SIZE = 256

class UnifiedLossModule(nn.Module):
    """
    智能统一损失模块
    功能：量纲对齐 + 动态加权
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.proc_size = PROC_SIZE
        
        # 1. 实例化所有组件
        self.losses = nn.ModuleDict({
            'energy': EnergyLoss().to(device),
            'ncc': NCCLoss().to(device),
            'feat': FeatLoss(device=device).to(device),
            'freq_band': FreqBandLoss().to(device),
            'grad': GradLoss().to(device),
            'l1': L1Loss().to(device),
            'l2': L2Loss().to(device),
            'ssim': SSIMLoss().to(device),
            'swd': SWDLoss().to(device)
        })
        
        # 2. 状态变量
        self.scale_factors = {}     # 量纲对齐因子 (1 / initial_loss)
        self.group_weights = {}     # 当前组权重
        self.calibrated = False     # 是否已校准
        
        # 默认屏蔽 SWD
        self.ignore_keys = ['swd'] 

    def set_group_weights(self, weights_dict):
        """由 Manager 外部注入权重"""
        self.group_weights = weights_dict

    # [修复点] 添加 set_stage 方法，防止 Manager 调用时报错
    def set_stage(self, stage):
        self.current_stage = stage
        # 传递给子模块 (如果它们需要)
        if hasattr(self.losses['energy'], 'set_stage'):
            self.losses['energy'].set_stage(stage)

    def preprocess(self, spec):
        """统一预处理"""
        spec_4d = spec.unsqueeze(0).unsqueeze(0)
        spec_down = F.interpolate(spec_4d, size=(self.proc_size, self.proc_size), 
                                   mode='bilinear', align_corners=True)
        spec_max = spec_down.max() + 1e-9
        spec_norm = spec_down / spec_max
        spec_log = torch.log1p(spec_norm * 20)
        spec_log = spec_log / (spec_log.max() + 1e-9)
        return spec_log.squeeze()
    
    def forward(self, pred_spec, obs_spec):
        """
        前向传播：
        1. 计算所有 Raw Loss
        2. 如果未校准，进行量纲校准
        3. 应用 Scale * Weight
        """
        device = pred_spec.device
        
        # 异常处理
        if pred_spec.numel() == 0 or obs_spec.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 统一尺寸
        if pred_spec.shape != obs_spec.shape:
            obs_spec = F.interpolate(
                obs_spec.unsqueeze(0).unsqueeze(0),
                size=pred_spec.shape,
                mode='bilinear', align_corners=True
            ).squeeze()
        
        pred_2d = self.preprocess(pred_spec)
        obs_2d = self.preprocess(obs_spec)
        
        # 1. 计算所有原始 Loss
        raw_losses = {}
        total_loss = torch.tensor(0.0, device=device)
        
        # 遍历计算 (跳过忽略项)
        for name, criterion in self.losses.items():
            if name in self.ignore_keys:
                raw_losses[name] = torch.tensor(0.0, device=device)
                continue
            
            # 部分 Loss 需要原始谱，部分需要 2D 图像
            if name == 'feat':
                val = criterion(pred_spec, obs_spec)
            else:
                val = criterion(pred_2d, obs_2d)
            raw_losses[name] = val
        
        # 2. 自动校准 (仅在前几次调用执行)
        if not self.calibrated:
            # 简单策略：使用当前 batch 的值作为基准
            # 避免除零
            for k, v in raw_losses.items():
                if k not in self.ignore_keys:
                    ref_val = v.detach().item()
                    self.scale_factors[k] = 1.0 / (ref_val + 1e-6)
                else:
                    self.scale_factors[k] = 0.0
            self.calibrated = True
            print(f"[Auto-Calibration] Scales initialized: { {k: f'{v:.2f}' for k,v in self.scale_factors.items()} }")

        # 3. 加权求和
        # 公式：WeightedLoss = Raw * ScaleFactor * GroupWeight
        loss_components = {} # 归一化后的分量，用于传给 Manager
        
        for name, val in raw_losses.items():
            if name in self.ignore_keys: continue
            
            w_group = self.group_weights.get(name, 0.0)
            scale = self.scale_factors.get(name, 1.0)
            
            # 归一化后的 Loss (这就实现了量纲对齐)
            norm_val = val * scale
            loss_components[name] = norm_val.detach().item()
            
            if w_group > 0:
                total_loss = total_loss + norm_val * w_group
        
        # 保存当前的组件状态供 Manager 异步读取
        self.last_components = loss_components
        
        return total_loss

# =============================================================================
# 第六部分：全局管理器 (物理接力策略)
# =============================================================================

class UnifiedLossManager:
    """
    中央控制器：负责分发权重和监控状态
    """
    _instance = None
    
    def __new__(cls, device='cuda'):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, device='cuda'):
        if hasattr(self, 'initialized'): return
        
        self.device = device
        self.loss_module = UnifiedLossModule(device=device)
        
        # --- 物理分组定义 ---
        # self.groups = {
        #     'basin': ['energy', 'ncc'],         # 拓荒组：宽收敛域
        #     'builder': ['feat', 'freq_band'],   # 建设组：中频结构
        #     'precision': ['grad', 'l1', 'ssim', 'l2'], # 精修组：高频细节
        #     'precision2': ['feat']
        # }
        
        self.groups = {
            'basin': ['energy', 'ncc'],         # 拓荒组：宽收敛域
            'builder': ['feat', 'freq_band'],   # 建设组：中频结构
            'precision': ['grad', 'l1', 'l2'], # 精修组：高频细节
            'precision2': ['feat']
        }

        # 组映射反查
        self.loss_to_group = {}
        for g, lst in self.groups.items():
            for l_name in lst:
                self.loss_to_group[l_name] = g

        self.current_stage = 1
        self.iter_count = 0
        
        # --- 监控与 EMA ---
        self.loss_history = {g: [] for g in self.groups} # 记录各组的归一化 Loss 均值
        self.ema_losses = {g: 1.0 for g in self.groups}  # 指数移动平均
        self.plateau_counter = 0                         # 平台期计数器
        
        # 初始权重分配
        self.update_weights_by_stage(1)
        
        self.initialized = True
        print("[UnifiedLossManager] Initialized with Physics-Based Relay Logic")

    def set_stage(self, stage_id):
        self.current_stage = stage_id
        # 重置平台期检测，防止上一阶段的平台期影响新阶段
        self.plateau_counter = 0 
        self.update_weights_by_stage(stage_id)
        # 通知模块更新
        self.loss_module.set_stage(stage_id) 
        print(f"[UnifiedLossManager] Switched to Stage {stage_id}")

    def update_weights_by_stage(self, stage, blend_ratio=0.0):
        """
        计算当前阶段的权重分配
        blend_ratio: 0.0 - 1.0, 用于平台期触发时混合下一阶段的权重
        """
        # 基础配置：定义每个阶段各组的相对重要性
        # Basin, Builder, Precision
        # 修改：Stage 4 的 basin 降为 0.0，precision 极大提升
        # 这就是“关门打狗”：不再还要什么能量匹配，只盯着细节打

        # stage_configs = {
        #     1: {'basin': 10.0, 'builder': 1.0,  'precision': 0.1, 'precision2': 0},
        #     2: {'basin': 2.0,  'builder': 10.0, 'precision': 1.0, 'precision2': 0},
        #     3: {'basin': 0.5,  'builder': 5.0,  'precision': 10.0, 'precision2': 0},
            
        #     # Stage 4 激进修改：
        #     # Basin = 0 (彻底断奶)
        #     # Builder = 0.5 (几乎没有)
        #     # Precision = 30.0 (全力以赴)
        #     4: {'basin': 0.5,  'builder': 1.0,  'precision': 30.0, 'precision2': 0}, 

        #     # Stage 5 配置：
        #     # Basin 降到 0.1 (几乎放手，只留一根风筝线，防止极低频漂移)
        #     # Precision 保持 20.0 (全力刷分)
        #     5: {'basin': 0,  'builder': 0,  'precision': 1, 'precision2': 40}, 
        # }

        stage_configs = {
            1: {'basin': 20.0, 'builder': 1.0,  'precision': 0.1, 'precision2': 0},
            2: {'basin': 5.0,  'builder': 20.0, 'precision': 1.0, 'precision2': 0},
            3: {'basin': 0.5,  'builder': 5.0,  'precision': 20.0, 'precision2': 0},
            
            4: {'basin': 0.0,  'builder': 1.0,  'precision': 20.0, 'precision2': 0}, 

            5: {'basin': 0,  'builder': 0,  'precision': 1, 'precision2': 20}, 
        }

        # 加上这行防御性代码，防止 stage 5 越界报错
        # 如果 stage > 5，就默认沿用 5 的配置
        if stage > 5:
            base_cfg = stage_configs[5]
        else:
            base_cfg = stage_configs.get(stage, stage_configs[4])
        
        # 如果触发了混合 (Plateau Trigger)，提前引入下一阶段特性
        if blend_ratio > 0 and stage < 5:
            next_cfg = stage_configs.get(stage + 1)
            final_cfg = {}
            for g in self.groups:
                final_cfg[g] = base_cfg[g] * (1 - blend_ratio) + next_cfg[g] * blend_ratio
        else:
            final_cfg = base_cfg
            
        # 展开到具体 Loss
        flat_weights = {}
        for g_name, weight in final_cfg.items():
            for l_name in self.groups[g_name]:
                # 维稳权重恢复
                flat_weights[l_name] = max(weight, 0.01) 
        
        self.loss_module.set_group_weights(flat_weights)
        self.current_group_cfg = final_cfg 

    def compute_loss(self, pred_spec, obs_spec):
        """计算损失并返回"""
        return self.loss_module(pred_spec, obs_spec)

    def step(self):
        """
        在每次迭代结束时调用。
        功能：
        1. 读取最新的 Loss 组件值。
        2. 更新 EMA。
        3. 检测平台期。
        4. 动态调整权重（如果需要）。
        """
        self.iter_count += 1
        if not hasattr(self.loss_module, 'last_components'):
            return

        components = self.loss_module.last_components
        
        # 1. 聚合各组的平均 Loss
        group_vals = {g: 0.0 for g in self.groups}
        group_counts = {g: 0 for g in self.groups}
        
        for l_name, val in components.items():
            g = self.loss_to_group.get(l_name)
            if g:
                group_vals[g] += val
                group_counts[g] += 1
        
        current_metrics = {}
        for g in self.groups:
            if group_counts[g] > 0:
                current_metrics[g] = group_vals[g] / group_counts[g]
            else:
                current_metrics[g] = 0.0
                
        # 2. 更新 EMA (平滑因子 0.95)
        alpha = 0.95
        for g, val in current_metrics.items():
            self.ema_losses[g] = alpha * self.ema_losses[g] + (1 - alpha) * val
            self.loss_history[g].append(val)
            
        # 3. 平台期检测 logic (Plateau Trigger)
        # 只有在非最后阶段才检测
        if self.current_stage < 4 and self.iter_count > 20:
            # 确定当前主导组 (Dominant Group)
            if self.current_stage == 1:
                focus_group = 'basin'
            elif self.current_stage == 2:
                focus_group = 'builder'
            elif self.current_stage <= 4:
                focus_group = 'precision'
            else:
                focus_group = 'precision2'
            
            # 计算最近 20 次的相对变化率
            recent_ema = self.ema_losses[focus_group]
            
            # 动态调整：如果当前主导组 Loss < 0.3 (归一化后)，说明拟合很好，加速混合
            if recent_ema < 0.3:
                self.plateau_counter += 1 # 加速
            else:
                self.plateau_counter += 0.5 # 正常速度
                
            # 计算混合比例 (0.0 -> 1.0)
            threshold = 80 
            if self.plateau_counter > threshold:
                blend = min(1.0, (self.plateau_counter - threshold) / 100.0)
                if blend > 0:
                    self.update_weights_by_stage(self.current_stage, blend_ratio=blend)

    def get_current_weights_info(self):
        """返回当前权重状态字符串"""
        if not hasattr(self, 'current_group_cfg'): return ""
        cfg = self.current_group_cfg
        return f"[W] Bas:{cfg['basin']:.1f} Bld:{cfg['builder']:.1f} Prc:{cfg['precision']:.1f} Prc2:{cfg['precision2']:.1f}"

# 绑定全局单例
_global_manager = None
def get_neural_loss_manager(device='cuda'):
    global _global_manager
    if _global_manager is None:
        _global_manager = UnifiedLossManager(device=device)
    return _global_manager

def reset_neural_loss_manager():
    global _global_manager
    UnifiedLossManager._instance = None
    _global_manager = None

def print_neural_loss_stats():
    pass

# =============================================================================
# 第七部分 & 第八部分：接口
# =============================================================================
def _process_spec(spec, f_axis, f_max, blur_sigma, use_abs, shallow_boost=False, contrast_pow=1.0):
    if spec is None: return None, 0

    if f_axis is not None and f_axis.numel() > 2:
        f_top = float(f_axis[-1].item())
        nf_idx = int(spec.shape[1] * (float(f_max) / max(f_top, 1.0)))
        nf_idx = max(4, min(nf_idx, spec.shape[1]))
        cut = spec[:, :nf_idx]
    else:
        cut = spec
        nf_idx = spec.shape[1]

    # cut = torch.clamp(cut, min=1e-12)

    # 对数变换（必执行，因use_abs=False）
    if not use_abs: cut = torch.log1p(cut)

    # 垂直轴高斯模糊（按需执行，看blur_sigma）
    if blur_sigma > 0.01: cut = gaussian_blur_v_axis(cut, blur_sigma)

    # 对比度增强（未执行，因contrast_pow=1.0）让亮的更亮、暗的更暗
    if contrast_pow > 1.01:
        max_val = cut.max() + 1e-9
        cut = (cut / max_val) ** contrast_pow

    # 浅层增强
    if shallow_boost:
        V = cut.shape[0]
        w = torch.linspace(2.0, 1.0, V, device=cut.device).view(-1, 1)
        cut = cut * w
    # q_val = torch.quantile(cut.detach(), 0.98)
    # cut = cut / (q_val + 1e-9)
    # cut = cut / (1.0 + cut)


    return cut, nf_idx

def gaussian_blur_v_axis(x, sigma):
    ksize = int(2 * math.ceil(2 * sigma) + 1)
    if ksize % 2 == 0: ksize += 1
    w = torch.arange(ksize, device=x.device).float() - ksize // 2
    g = torch.exp(-w**2 / (2 * sigma**2))
    g = g / g.sum()
    g = g.view(ksize, 1)
    x_in = x.unsqueeze(0).unsqueeze(0)
    k = g.view(1, 1, ksize, 1)
    pad = ksize // 2
    x_pad = F.pad(x_in, (0, 0, pad, pad), mode='replicate')
    out = F.conv2d(x_pad, k)
    return out.squeeze()

def unified_loss(pred_spec, obs_spec, stage_id=1, device='cuda'):
    manager = get_neural_loss_manager(device)
    # Stage已经在外部循环设置，这里直接计算
    return manager.compute_loss(pred_spec, obs_spec)

def get_spectra_loss_split_fast(pred_seis, shot_idx, obs_spectra_storage,
                                sx, gx, dx, time_tensor,
                                fmax_curr, dt_val,
                                blur_sigma=0.0,
                                vres=400, fres=400,
                                stage_id=1,
                                **kwargs):
    device = pred_seis.device
    obs_pack = obs_spectra_storage[shot_idx]
    GLOBAL_FMAX = float(obs_pack['f'][-1].item()) if obs_pack['f'] is not None else 80.0

    pred_L, pred_R, f_pred, _, _ = compute_split_spectra(
        pred_seis, sx[shot_idx], gx, dx, time_tensor,
        GLOBAL_FMAX, dt_val, min_traces=2,
        vres=vres, fres=fres
    )
    
    shallow_boost = (fmax_curr >= 30.0)

    def calc_side(obs_s, pred_s):
        if obs_s is None or pred_s is None:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        o_p, _ = _process_spec(obs_s, obs_pack['f'], fmax_curr, blur_sigma, False, 
                               shallow_boost, 1.0)
        p_p, _ = _process_spec(pred_s, f_pred, fmax_curr, blur_sigma, False, 
                               shallow_boost, 1.0)
        
        if o_p is None or p_p is None or p_p.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        if o_p.shape != p_p.shape:
            o_p = F.interpolate(o_p.unsqueeze(0).unsqueeze(0), size=p_p.shape, 
                                mode='bilinear', align_corners=True).squeeze()

        loss_val = unified_loss(p_p, o_p, stage_id, device)
        return loss_val

    l_L = calc_side(obs_pack['L'], pred_L)
    l_R = calc_side(obs_pack['R'], pred_R)

    cnt_L = obs_pack['cnt_L']
    cnt_R = obs_pack['cnt_R']
    total = cnt_L + cnt_R
    if total > 0:
        return (l_L * cnt_L + l_R * cnt_R) / total
    return torch.tensor(0.0, device=device, requires_grad=True)


