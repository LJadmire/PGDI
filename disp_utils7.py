# disp_utils7.py
import torch
import torch.nn.functional as F
import numpy as np


def _pws_stack(c, pws_strength: float, nuFlag: int):
    """
    改进版 PWS stacking (强相干增强，可微).
    c: complex [F,V,X]
    """

    eps = 1e-12
    F, V, X = c.shape

    # 1. 普通叠加
    s = c.sum(dim=2, keepdim=False)           # [F,V]
    s_abs = torch.abs(s)
    c_abs2_sum = torch.sum(torch.abs(c)**2, dim=2)  # [F,V]

    # 2. semblance-like coherence C (完全可导)
    #    C = |sum(c)|^2 / (X * sum(|c|^2))
    C = (s_abs**2) / (X * c_abs2_sum + eps)   # [F,V]
    C = torch.clamp(C, 0.0, 1.0)

    # 3. 强度控制：C^p
    if pws_strength != 1.0:
        C = C ** pws_strength

    # 4. 叠加
    if nuFlag == 1:
        # normalized PWS = C * s
        img_spec = C * s                      # [F,V]
    else:
        # unnormalized energy version
        img_spec = C * s_abs                  # [F,V] (real)
        img_spec = img_spec.to(torch.complex64)

    return img_spec



def FPhaseshift(uxt, x, t, normFlag, fmin, fmax,
                vmin, vmax, fres, vres, pwsFlag=0):
    """
    Phase-shift dispersion imaging (适用于频散谱全波形反演).

    参数与原版保持一致:
        uxt: [ng, nt] 或 [nt, ng]，可以是 numpy 或 torch.Tensor
        x:   [ng]  道距坐标
        t:   [nt]  时间轴
        normFlag: 0/1  是否在最终图像按频率列归一化
        fmin, fmax: 频率范围
        vmin, vmax: 相速度范围
        fres, vres: 频率/速度采样数（最终输出尺寸）
        pwsFlag:
            - 标量 0       : 普通相移叠加 (no PWS)
            - 标量 >0      : 简单 PWS, 强度 = pwsFlag
            - 序列 [a, b]  : a = PWS 强度, b = nuFlag(0/1)
            - b: 0/1 nuFlag(是否对权重归一)

    返回:
        img_final: [vres, fres]  (速度 × 频率)
        f_out:     [fres]
        v_out:     [vres]
    """
    # -------- 输入整理 --------
    if not torch.is_tensor(uxt):
        uxt = torch.as_tensor(uxt, dtype=torch.float32)
    device = uxt.device

    x = torch.as_tensor(x, dtype=torch.float32, device=device).view(-1)   # [nx]
    t = torch.as_tensor(t, dtype=torch.float32, device=device).view(-1)   # [nt]

    # Deepwave gather 通常是 [ng, nt]，转成 [nt, ng]
    if uxt.shape[0] == x.numel():
        uxt = uxt.T  # -> [nt, nx]

    # 去掉 NaN 道（如果有），避免 NaN 传播
    with torch.no_grad():
        mtrace = torch.nanmean(uxt, dim=0)        # [nx]
        trace_mask = ~torch.isnan(mtrace)
    if trace_mask.sum() < x.numel():
        uxt = uxt[:, trace_mask]
        x = x[trace_mask]

    nt, nx = uxt.shape
    dt = float(torch.mean(torch.diff(t)).item())

    # -------- FFT (只算正频，提高效率 & 更接近 Matlab fftrl) --------
    # 2 的幂 * 2 的零填充，兼顾分辨率与效率
    nfft = int(2 ** np.ceil(np.log2(nt))) * 2

    # real-to-complex FFT, 只返回 [0, f_Nyquist]
    spec = torch.fft.rfft(uxt, n=nfft, dim=0)             # [F_all, nx]
    freqs = torch.fft.rfftfreq(nfft, d=dt).to(device)     # [F_all]

    # -------- 频率裁剪 --------
    f_mask = (freqs >= fmin) & (freqs <= fmax)
    spec_cut = spec[f_mask, :]                            # [F, nx]
    f_vec = freqs[f_mask]                                 # [F]

    if f_vec.numel() == 0:
        # 没有落在范围内的频率，返回空图
        img = torch.zeros(int(vres), int(fres), device=device)
        return (img,
                torch.linspace(fmin, fmax, int(fres), device=device),
                torch.linspace(vmin, vmax, int(vres), device=device))

    # -------- 速度轴 --------
    v_vec = torch.linspace(vmin, vmax, int(vres), device=device)  # [V]

    # -------- 广播构造相移 --------
    # spec_cut: [F, X]
    spec_expanded = spec_cut.unsqueeze(1)                 # [F, 1, X]

    f_expanded = f_vec.view(-1, 1, 1)                     # [F, 1, 1]
    v_expanded = v_vec.view(1, -1, 1)                     # [1, V, 1]
    x_expanded = x.view(1, 1, -1)                         # [1, 1, X]

    # 相移相位: 2π f x / v
    angle = (2.0 * np.pi * f_expanded * x_expanded) / (v_expanded + 1e-12)
    phase_shift = torch.complex(torch.cos(angle), torch.sin(angle))   # [F, V, X]

    # 相移后的复数贡献
    c = phase_shift * spec_expanded                      # [F, V, X]

    # -------- PWS or 普通叠加 --------
    # 解析 pwsFlag:
    #   标量: 强度 = pwsFlag, nuFlag = 1
    #   序列: [强度, nuFlag]
    nuFlag = 1
    if isinstance(pwsFlag, (list, tuple)):
        if len(pwsFlag) >= 1:
            pws_strength = float(pwsFlag[0])
        else:
            pws_strength = 0.0
        if len(pwsFlag) >= 2:
            nuFlag = int(pwsFlag[1])
    else:
        pws_strength = float(pwsFlag)

    if pws_strength <= 0.0:
        # 普通相移叠加：沿道数求和
        img_spec = c.sum(dim=2)                          # [F, V] complex
    else:
        # PWS 叠加
        img_spec = _pws_stack(c, pws_strength, nuFlag)   # [F, V] complex

    # 幅度图 [V, F]
    img_amp = torch.abs(img_spec).T                      # [V, F]

    # -------- 归一化（可选）--------
    # 这里在 (V,F) 上按频率列归一化，便于对比不同频率能量
    if normFlag:
        # 列最大值归一化 (可导，但非最大处梯度为 0，和 Matlab 一致思路)
        col_max = torch.amax(img_amp, dim=0, keepdim=True)  # [1, F]
        img_amp = img_amp / (col_max + 1e-12)

    # -------- 插值到目标 (vres, fres) --------
    img_amp_4d = img_amp.unsqueeze(0).unsqueeze(0)   # [1,1,V,F]
    img_final = F.interpolate(
        img_amp_4d,
        size=(int(vres), int(fres)),
        mode='bilinear',
        align_corners=True
    ).squeeze()                                      # [vres, fres]

    # 不额外 nan_to_num，尽量避免掩盖数值问题；
    # 若前面 eps 足够，一般不会产生 NaN/Inf。
    # 如果你非常担心，可以解开下面两行：
    # img_final = torch.nan_to_num(img_final, nan=0.0, posinf=0.0, neginf=0.0)

    # 输出频率/速度轴
    f_out = torch.linspace(fmin, fmax, int(fres), device=device)
    v_out = torch.linspace(vmin, vmax, int(vres), device=device)

    return img_final, f_out, v_out


