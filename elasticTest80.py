# elasticTest80.py
# Optimized Deep Learning Dispersion Spectrum FWI
# 统一损失管理版本 - 物理阶段接力策略 (Physics-Based Relay Strategy)
# 规避频率作弊
# badInit

import os, time, random, math
import numpy as np
import torch
import deepwave
from deepwave import elastic
import matplotlib.pyplot as plt

from elasticTest80_utils import ( 
    compute_split_spectra,
    get_spectra_loss_split_fast,
    apply_gradient_taper,
    get_linear_depth_weight,
    get_power_depth_weight,
    get_lateral_weight,
    gaussian_smooth_2d,
    get_neural_loss_manager,
    reset_neural_loss_manager,
    print_neural_loss_stats,
    ricker_differentiable,
)

start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ===============================================================
# 1) Model & Init
# ===============================================================
if os.path.exists("model8_2.npy"):
    vs_np = np.load("model8_2.npy")
    if vs_np.shape[0] > vs_np.shape[1]: 
        vs_np = vs_np.T
    vs_true = torch.tensor(vs_np, dtype=torch.float32, device=device)
else:
    nz_t, nx_t = 100, 300
    vs_true = torch.ones(nz_t, nx_t, device=device) * 300.0
    vs_true[50:, :] = 600.0
    vs_true[60:80, 100:200] = 800.0

nz, nx = vs_true.shape
dx = 1.0
vsmin, vsmax = vs_true.min(), vs_true.max()

vs_init = torch.zeros_like(vs_true)
for i in range(nx):
    vs_init[:, i] = torch.linspace(100, 110, nz, device=device)
    # vs_init[:, i] = torch.linspace(vsmin*0.9, vsmax*1.1, nz, device=device)

print(f"Model size: {nz} x {nx}")



# ===============================================================
# 2) Acquisition
# ===============================================================
ds, dg = 6, 2
sx = torch.arange(2, nx - 2, ds, device=device).long()
ns = sx.numel()
sz = torch.ones_like(sx, device=device).long() * 1
gx = torch.arange(0, nx, dg, device=device).long()
ng = gx.numel()
gz = torch.ones_like(gx, device=device).long() * 1

source_locations = torch.zeros(ns, 1, 2, dtype=torch.long, device=device)
source_locations[:, 0, 0] = sz
source_locations[:, 0, 1] = sx
receiver_locations = torch.zeros(ns, ng, 2, dtype=torch.long, device=device)
receiver_locations[:, :, 0] = gz.unsqueeze(0).expand(ns, -1)
receiver_locations[:, :, 1] = gx.unsqueeze(0).expand(ns, -1)

dt = 0.0005
nt = int(0.8 / dt)
true_freq = 30.0
ricker_true = deepwave.wavelets.ricker(true_freq, nt, dt, 1.5 / true_freq).to(device)
src_amp_true = ricker_true.repeat(ns, 1, 1)
time_tensor = torch.arange(nt, device=device).float() * dt

print(f"Shots: {ns}, Receivers: {ng}, Time samples: {nt}")

# ===============================================================
# 3) Generate True Data
# ===============================================================
print("Generating True Data...")
obs_data_list = []
with torch.no_grad():
    for i in range(0, ns, 5):
        i_end = min(i + 5, ns)
        out = elastic(
            *deepwave.common.vpvsrho_to_lambmubuoyancy(
                vs_true * 1.732, vs_true, vs_true * 1.732 * 0.23 + 2000
            ),
            dx, dt,
            source_amplitudes_y=src_amp_true[i:i_end],
            source_locations_y=source_locations[i:i_end],
            receiver_locations_y=receiver_locations[i:i_end],
            pml_width=[0, 20, 20, 20], pml_freq=true_freq,
        )
        obs_data_list.append(out[-2].detach()) 
obs_data = torch.cat(obs_data_list, dim=0)
print(f"Obs data shape: {obs_data.shape}")

# ===============================================================
# 4) Precompute Spectra
# ===============================================================
GLOBAL_VRES = 400
GLOBAL_FRES = 300
GLOBAL_FMAX_PRECOMP = 100.0

print(f"Pre-computing Spectra ({GLOBAL_VRES}x{GLOBAL_FRES})...")
obs_spectra_storage = []

for i in range(ns):
    curr_sx = sx[i]
    with torch.no_grad():
        sl, sr, faxis, nl, nr = compute_split_spectra(
            obs_data[i], curr_sx, gx, dx, time_tensor,
            GLOBAL_FMAX_PRECOMP, dt, min_traces=2,
            vres=GLOBAL_VRES, fres=GLOBAL_FRES 
        )
    obs_spectra_storage.append({
        "L": sl, "R": sr, "f": faxis, 
        "cnt_L": nl, "cnt_R": nr
    })
    
torch.cuda.empty_cache()
print("Spectra computed.")

# ===============================================================
# 5) Initialize Loss Manager
# ===============================================================
print("Initializing Unified Loss Manager (Dynamic Relay Mode)...")
reset_neural_loss_manager()
neural_manager = get_neural_loss_manager(device)

# ===============================================================
# 6) 优化的反演参数
# ===============================================================

# freq: None 表示反演频率
schedule = [
    {"stage": 1, "freq": None, "iters": 150, "lr": 2.0, "sigma": 1.2, "fmax": 12.0, "reset": True, "freq_scale": 1.0},
    {"stage": 2, "freq": None, "iters": 120, "lr": 1.5, "sigma": 0.7, "fmax": 40.0, "reset": False, "freq_scale": 0.3},
    {"stage": 3, "freq": None, "iters": 150, "lr": 1.0, "sigma": 0.5, "fmax": 70.0, "reset": False, "freq_scale": 0.1},
    {"stage": 4, "freq": None, "iters": 200, "lr": 0.8, "sigma": 1.0, "fmax": 100.0, "reset": False, "freq_scale": 0.1},
  
]

vs_param = vs_init.clone().requires_grad_(True)

# 频率反演参数设置
FREQ_INIT = 15.0
FREQ_MIN = 10.0
FREQ_MAX = 35.0


freq_param = torch.tensor([FREQ_INIT], device=device, dtype=torch.float32, requires_grad=True)

optimizer = torch.optim.Adam([vs_param], lr=0.1)

# 降低loss scale，梯度在内部已经做过归一化平衡，这里仅作为最后的缩放
LOSS_SCALE = 100.0 
total_iter = 0

# Batch管理
perm = list(range(ns))
random.shuffle(perm)
ptr = 0

def next_batch(bsz):
    global ptr, perm
    if ptr + bsz > ns:
        random.shuffle(perm)
        ptr = 0
    ids = perm[ptr:ptr + bsz]
    ptr += bsz
    return ids

print("\n" + "="*60)
print("Starting Unified Deep Learning FWI (Physics-Based Relay)...")
print("="*60)

loss_history = []
mse_history = []
freq_history = []

for cfg in schedule:
    s_id = cfg["stage"]
    freq_cfg = cfg["freq"]
    iters = cfg["iters"]
    lr_base = cfg["lr"]
    sigma_curr = cfg["sigma"]
    fmax_disp = cfg["fmax"]
    do_reset = cfg["reset"]
    
    # 获取当前 Stage 的频率学习率缩放
    current_freq_scale = cfg.get("freq_scale", 0.1)

    invert_freq = (freq_cfg is None)
    
    if invert_freq:
        freq_curr_display = f"Invert(init={freq_param.item():.1f})"
    else:
        freq_curr_display = f"{freq_cfg}"

    print(f"\n{'='*60}")
    print(f">>> Stage {s_id}: Freq={freq_curr_display}, Sigma={sigma_curr}, Fmax={fmax_disp}")
    print(f"{'='*60}")

    # 设置物理阶段，管理器会自动调整权重策略
    neural_manager.set_stage(s_id)

    if do_reset or s_id == 1:
        # 彻底重置优化器状态 (丢弃动量)
        if invert_freq:
            optimizer = torch.optim.Adam([
                {'params': [vs_param], 'lr': lr_base},
                {'params': [freq_param], 'lr': lr_base * current_freq_scale}
            ])
        else:
            optimizer = torch.optim.Adam([vs_param], lr=lr_base)
    else:
        # 保留动量，但强制重置学习率 (Warm Restart)
        # 这会给模型一个“冲击”，帮助它跳出浅层极小值
        for pg in optimizer.param_groups:
            if pg['params'][0] is vs_param:
                pg["lr"] = lr_base
            elif invert_freq and pg['params'][0] is freq_param:
                # 动态更新频率的 LR
                pg["lr"] = lr_base * current_freq_scale

    # 深度权重
    if s_id <= 1:
        d_w = get_linear_depth_weight(nz, nx, slope=0.3, base=1.0, device=device)
    elif s_id == 2:
        d_w = get_linear_depth_weight(nz, nx, slope=0.8, base=1.0, device=device)
    elif s_id == 3:
        d_w = get_power_depth_weight(nz, nx, exponent=2.0, scale=8.0, base=1.0, device=device)
    else:
        d_w = get_power_depth_weight(nz, nx, exponent=3.0, scale=20.0, base=1.0, device=device)

    # 横向权重
    if s_id >= 3:
        l_w = get_lateral_weight(nz, nx, mode='ramp', start_x_ratio=0.25, 
                                  left_val=1.0, right_val=15.0, device=device)
    else:
        l_w = torch.ones(nz, nx, device=device)

    total_weight = d_w * l_w
    
    b_size = 5
    max_gn = 80.0 if s_id <= 1 else (50.0 if s_id <= 2 else 25.0)
    
    if s_id <= 1:
        fmax_list = [max(3.0, fmax_disp * 0.5), max(5.0, fmax_disp * 0.8), float(fmax_disp)]
    elif s_id == 2:
        fmax_list = [max(5.0, fmax_disp * 0.6), max(8.0, fmax_disp * 0.85), float(fmax_disp)]
    else:
        fmax_list = [max(8.0, fmax_disp * 0.7), float(fmax_disp)]

    for k in range(iters):
        optimizer.zero_grad()
        
        # LR Schedule (Cosine Decay)
        progress = k / iters
        if s_id == 1 and k < 20:
            # Warmup
            factor = (0.3 + 0.7 * k / 20)
        else:
            # Cosine
            factor = (0.3 + 0.7 * (0.5 + 0.5 * math.cos(math.pi * progress)))
        
        curr_lr = lr_base * factor
        
        for pg in optimizer.param_groups:
            if pg['params'][0] is vs_param:
                pg["lr"] = curr_lr
            elif invert_freq and pg['params'][0] is freq_param:
                pg["lr"] = curr_lr * current_freq_scale

        if invert_freq:
            # 必须直接使用 freq_param 参与计算，不要 detach，不要 item()
            # clamp 会返回一个新的 tensor，并保留梯度链
            freq_curr_tensor = torch.clamp(freq_param, min=FREQ_MIN, max=FREQ_MAX).squeeze()
            
            # 用于不需要梯度的参数 (如 PML 频率设置, 打印等)
            freq_val_float = freq_curr_tensor.detach().item()
        else:
            freq_curr_tensor = torch.tensor(freq_cfg, device=device)
            freq_val_float = freq_cfg

        ricker_stage = ricker_differentiable(freq_curr_tensor, nt, dt, 1.5 / freq_curr_tensor).to(device)
        
        src_amp_stage = ricker_stage.repeat(ns, 1, 1)

        vp_iter = vs_param * 1.732
        rho_iter = vp_iter * 0.23 + 2000.0

        batch_ids = next_batch(b_size)

        out_pred = elastic(
            *deepwave.common.vpvsrho_to_lambmubuoyancy(vp_iter, vs_param, rho_iter),
            dx, dt,
            source_amplitudes_y=src_amp_stage[batch_ids],
            source_locations_y=source_locations[batch_ids],
            receiver_locations_y=receiver_locations[batch_ids],
            pml_width=[0, 20, 20, 20], pml_freq=freq_val_float
        )
        pred_data_batch = out_pred[-2]

        loss_batch = torch.tensor(0.0, device=device)
        
        for b_i, real_shot_id in enumerate(batch_ids):
            l_sum = torch.tensor(0.0, device=device)
            for fm in fmax_list:
                # 频散谱计算 (Dispersion Spectrum Loss)
                # 只要这里的 Loss 对 pred_data_batch 敏感
                # 梯度链条就是：Loss -> pred_data -> src_amp -> ricker -> freq_param
                l1 = get_spectra_loss_split_fast(
                    pred_data_batch[b_i], 
                    real_shot_id, 
                    obs_spectra_storage,
                    sx, gx, dx, time_tensor, 
                    fm, dt,
                    blur_sigma=sigma_curr,
                    vres=GLOBAL_VRES, 
                    fres=GLOBAL_FRES,
                    stage_id=s_id,
                )
                l_sum = l_sum + l1
            loss_batch = loss_batch + (l_sum / len(fmax_list))

        # 注意：这里的 LOSS_SCALE 仅用于优化器数值稳定性
        # 内部的 Loss 平衡已经由 UnifiedLossManager 处理
        loss_batch = (loss_batch / len(batch_ids)) * LOSS_SCALE
        
        # 反向传播：现在完全依靠数据匹配程度来更新频率
        loss_batch.backward()

        # >>>>>> 关键点：调用 Manager 的 Step 更新动态权重状态 <<<<<<
        # 在 backward 之后调用，虽然本次梯度的权重是旧的，但数据会被用于下一轮
        neural_manager.step()

        with torch.no_grad():
            if vs_param.grad is not None:
                grad = vs_param.grad.clone()

                grad = grad / (torch.std(grad) + 1e-12)

                grad = grad * total_weight
                
                taper_w = 5 if s_id == 1 else (3 if s_id <= 2 else 0)
                grad = apply_gradient_taper(grad, taper_w)

                # gn = torch.norm(grad)
                # if gn > max_gn:
                #     grad = grad * (max_gn / (gn + 1e-12))

                # smooth_sigma = sigma_curr * 1.5 if s_id <= 2 else sigma_curr
                if s_id >= 3:
                    smooth_sigma = 0.0
                else:
                    smooth_sigma = sigma_curr * 0.5

                if smooth_sigma > 0.05:
                    grad = gaussian_smooth_2d(grad, sigma=smooth_sigma)

                vs_param.grad.copy_(grad)
            
            if invert_freq and freq_param.grad is not None:
                # 如果 scale 几乎为 0 (Stage 5)，则直接不更新梯度，或者将梯度置零
                if current_freq_scale < 1e-6:
                    freq_param.grad.zero_()  # 显式切断梯度，确保绝对锁死
                else:
                    # 正常的梯度裁剪逻辑
                    if s_id == 1:
                        freq_grad_max = 10.0 
                    else:
                        freq_grad_max = 1.0 
                    
                    freq_gn = torch.abs(freq_param.grad)
                    if freq_gn > freq_grad_max:
                        a=1
                        # freq_param.grad.copy_(freq_param.grad * (freq_grad_max / (freq_gn + 1e-12)))

        optimizer.step()
        
        with torch.no_grad():
            vs_param.clamp_(min=vsmin * 0.95, max=vsmax * 1.05)
            if invert_freq:
                freq_param.clamp_(min=FREQ_MIN, max=FREQ_MAX)

        loss_history.append(loss_batch.item())
        if invert_freq:
            freq_history.append(freq_param.item())

        if (k % 20 == 0) or (k == iters - 1):
            with torch.no_grad():
                mse = torch.mean((vs_param - vs_true) ** 2).item()
            mse_history.append(mse)
            mem_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            
            # 获取当前的动态权重信息用于打印
            curr_stats = neural_manager.get_current_weights_info()
            
            if invert_freq:
                print(f"  Iter {total_iter + k:03d} | Loss: {loss_batch.item():.4f} | "
                      f"MSE: {mse:.1f} | Freq: {freq_param.item():.2f} Hz | "
                      f"LR: {curr_lr:.4f} | {curr_stats}")
            else:
                print(f"  Iter {total_iter + k:03d} | Loss: {loss_batch.item():.4f} | "
                      f"MSE: {mse:.1f} | LR: {curr_lr:.4f} | {curr_stats}")

    total_iter += iters
    
    if invert_freq:
        print(f"  >>> Stage {s_id} End: Estimated Freq = {freq_param.item():.2f} Hz")
    
    print_neural_loss_stats()
    torch.cuda.empty_cache()

# ===============================================================
# 7) Results
# ===============================================================


print("\n" + "="*60)
print("Inversion Complete!")
print("="*60)

vs_final = vs_param.detach().cpu().numpy()
vs_true_np = vs_true.detach().cpu().numpy()
vs_init_np = vs_init.detach().cpu().numpy()
rel_err = np.linalg.norm(vs_final - vs_true_np) / (np.linalg.norm(vs_true_np) + 1e-12) * 100.0
final_mse = np.mean((vs_final - vs_true_np) ** 2)

print(f"\n最终相对误差: {rel_err:.2f}%")
print(f"最终MSE: {final_mse:.2f}")




from datetime import datetime

# 自动生成文件夹名称（包含时间戳和关键参数）
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# result_folder = f"./results63_proposed_badInitfs"
result_folder = f"./results80_proposed_badInit_{final_mse:.1f}_{rel_err:.1f}pct"
# result_folder = f"./results_allLoss{timestamp}_err{rel_err:.1f}pct"

os.makedirs(result_folder, exist_ok=True)
print(f"\n结果将保存到: {result_folder}")







# ===============================================================
# 8) Main Visualization
# ===============================================================
plt.figure(figsize=(16, 12))

plt.subplot(3, 2, 1)
plt.imshow(vs_true_np, aspect="auto", cmap="jet", vmin=float(vsmin), vmax=float(vsmax))
plt.colorbar(label='Vs (m/s)')
plt.title("True Vs Model")

plt.subplot(3, 2, 2)
plt.imshow(vs_init_np, aspect="auto", cmap="jet", vmin=float(vsmin), vmax=float(vsmax))
plt.colorbar(label='Vs (m/s)')
plt.title("Initial Vs Model")

plt.subplot(3, 2, 3)
plt.imshow(vs_final, aspect="auto", cmap="jet", vmin=float(vsmin), vmax=float(vsmax))
plt.colorbar(label='Vs (m/s)')
plt.title(f"Inverted Vs Model (Error = {rel_err:.2f}%)")

plt.subplot(3, 2, 4)
error_map = np.abs(vs_final - vs_true_np)
plt.imshow(error_map, aspect="auto", cmap="hot", vmax=np.percentile(error_map, 95))
plt.colorbar(label='|Error| (m/s)')
plt.title("Absolute Error")

plt.subplot(3, 2, 5)
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss History')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.subplot(3, 2, 6)
mid_x = nx // 2
plt.plot(vs_true_np[:, mid_x], range(nz), 'b-', linewidth=2, label='True')
plt.plot(vs_init_np[:, mid_x], range(nz), 'g--', linewidth=1.5, label='Initial')
plt.plot(vs_final[:, mid_x], range(nz), 'r-', linewidth=2, label='Inverted')
plt.gca().invert_yaxis()
plt.xlabel('Vs (m/s)')
plt.ylabel('Depth (grid)')
plt.title(f'Depth Profile at x={mid_x}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{result_folder}/elastic_unified_deep.jpg", dpi=200, bbox_inches='tight')
plt.show()

total_time = time.time() - start_time
print(f"\n总耗时: {total_time/60:.2f} 分钟")

# ===============================================================
# 9) MSE曲线分析
# ===============================================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(mse_history, 'b-o', markersize=3)
plt.xlabel('Checkpoint')
plt.ylabel('MSE')
plt.title('MSE Evolution')
plt.grid(True, alpha=0.3)

# 各阶段MSE下降率
stage_boundaries = []
iter_count = 0
for cfg in schedule:
    stage_boundaries.append(iter_count)
    iter_count += cfg['iters']

plt.subplot(1, 2, 2)
# 计算每个阶段的MSE下降
stage_mse = []
check_per_stage = []
idx = 0
for i, cfg in enumerate(schedule):
    n_checks = cfg['iters'] // 20 + 1
    if idx + n_checks <= len(mse_history):
        stage_start = mse_history[idx]
        stage_end = mse_history[min(idx + n_checks - 1, len(mse_history) - 1)]
        reduction = (stage_start - stage_end) / stage_start * 100 if stage_start > 0 else 0
        stage_mse.append(reduction)
        idx += n_checks
    else:
        stage_mse.append(0)

stages = [f"Stage {i+1}" for i in range(len(schedule))]
colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']
plt.bar(stages, stage_mse, color=colors[:len(stages)])
plt.xlabel('Stage')
plt.ylabel('MSE Reduction (%)')
plt.title('MSE Reduction per Stage')
plt.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(stage_mse):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(f"{result_folder}/mse_analysis.jpg", dpi=200, bbox_inches='tight')
plt.show()

# ===============================================================
# 10) QC Plot - 频散谱对比
# ===============================================================
print("\n" + "="*60)
print("Generating QC Plot...")
print("="*60)

target_shots = [3, 15, 27]
target_shots = [s for s in target_shots if s < ns]
if len(target_shots) == 0:
    target_shots = [0, ns//2, ns-1]

qc_freq = true_freq

vs_final_qc = vs_param.detach().clone().to(device)
vs_final_qc = torch.clamp(vs_final_qc, min=vsmin, max=vsmax)
vp_final_qc = vs_final_qc * 1.732
rho_final_qc = vp_final_qc * 0.23 + 2000.0

fig, axs = plt.subplots(len(target_shots), 5, figsize=(20, 4*len(target_shots)))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

if len(target_shots) == 1:
    axs = axs.reshape(1, -1)

for row_idx, shot_id in enumerate(target_shots):
    print(f"  Processing Shot ID: {shot_id}...")
    
    real_seis = obs_data[shot_id].detach().cpu()
    obs_spec_pack = obs_spectra_storage[shot_id]
    if obs_spec_pack['R'] is not None:
        real_spec = obs_spec_pack['R'].detach().cpu()
        direction = 'Right'
    elif obs_spec_pack['L'] is not None:
        real_spec = obs_spec_pack['L'].detach().cpu()
        direction = 'Left'
    else:
        print(f"    Warning: No spectrum for shot {shot_id}")
        continue
    
    ricker_qc = deepwave.wavelets.ricker(qc_freq, nt, dt, 1.5 / qc_freq).to(device)
    src_amp_qc = ricker_qc.repeat(1, 1, 1)
    src_loc_qc = source_locations[shot_id:shot_id+1]
    rec_loc_qc = receiver_locations[shot_id:shot_id+1]
    
    with torch.no_grad():
        out_qc = elastic(
            *deepwave.common.vpvsrho_to_lambmubuoyancy(vp_final_qc, vs_final_qc, rho_final_qc),
            dx, dt,
            source_amplitudes_y=src_amp_qc,
            source_locations_y=src_loc_qc,
            receiver_locations_y=rec_loc_qc,
            pml_width=[0, 20, 20, 20],
            pml_freq=qc_freq
        )
        pred_seis_tensor = out_qc[-2].squeeze(0)
    
    pred_seis = pred_seis_tensor.cpu()
    
    with torch.no_grad():
        pred_L, pred_R, f_axis_qc, _, _ = compute_split_spectra(
            pred_seis_tensor, sx[shot_id], gx, dx, time_tensor,
            GLOBAL_FMAX_PRECOMP, dt, min_traces=2,
            vres=GLOBAL_VRES, fres=GLOBAL_FRES
        )
    
    if direction == 'Right' and pred_R is not None:
        pred_spec = pred_R.cpu()
    elif pred_L is not None:
        pred_spec = pred_L.cpu()
    else:
        pred_spec = torch.zeros_like(real_spec)


    # --- C01. 计算初始模型数据的谱 ---
    out_qc_init = elastic(
        *deepwave.common.vpvsrho_to_lambmubuoyancy(vs_init * 1.732, vs_init, vs_init * 1.732 * 0.23 + 2000.0),
        dx, dt,
        source_amplitudes_y=src_amp_qc,
        source_locations_y=src_loc_qc,
        receiver_locations_y=rec_loc_qc,
        pml_width=[0, 20, 20, 20],
        pml_freq=qc_freq
    )
    pred_seis_tensor_init = out_qc_init[-2].squeeze(0)

    pred_L_init, pred_R_init, f_axis_qc_init, _, _ = compute_split_spectra(
        pred_seis_tensor_init, sx[shot_id], gx, dx, time_tensor,
        GLOBAL_FMAX_PRECOMP, dt, min_traces=2,
        vres=GLOBAL_VRES, fres=GLOBAL_FRES
    )
    
    if direction == 'Right' and pred_R_init is not None:
        pred_spec_init = pred_R_init.cpu()
    elif pred_L_init is not None:
        pred_spec_init = pred_L_init.cpu()
    else:
        pred_spec_init = torch.zeros_like(real_spec)

    
    # 绘图
    clim = torch.quantile(real_seis.abs(), 0.98).item()
    ax1 = axs[row_idx, 0]
    ax1.imshow(real_seis.T, aspect='auto', cmap='gray', vmin=-clim, vmax=clim)
    ax1.set_title(f"Shot {shot_id}: Obs Seis")
    ax1.set_xlabel("Trace")
    ax1.set_ylabel("Time Sample")
    
    clim_p = torch.quantile(pred_seis.abs(), 0.98).item()
    ax2 = axs[row_idx, 1]
    ax2.imshow(pred_seis.T, aspect='auto', cmap='gray', vmin=-clim_p, vmax=clim_p)
    ax2.set_title(f"Shot {shot_id}: Pred Seis")
    ax2.set_xlabel("Trace")
    ax2.set_yticks([])
    
    real_spec_log = torch.log1p(real_spec)
    vmax_s = real_spec_log.max().item()
    ax3 = axs[row_idx, 2]
    ax3.imshow(real_spec_log, aspect='auto', cmap='jet', origin='lower', vmin=0, vmax=vmax_s*0.9)
    ax3.set_title(f"Shot {shot_id}: Obs Spectra ({direction})")
    ax3.set_xlabel("Freq Bin")
    ax3.set_ylabel("Velocity Bin")
    
    pred_spec_log = torch.log1p(pred_spec)
    vmax_p = pred_spec_log.max().item()
    ax4 = axs[row_idx, 3]
    ax4.imshow(pred_spec_log, aspect='auto', cmap='jet', origin='lower', vmin=0, vmax=vmax_p*0.9)
    ax4.set_title(f"Shot {shot_id}: Pred Spectra")
    ax4.set_xlabel("Freq Bin")
    ax4.set_yticks([])

    # 5. 初始模型频散谱
    pred_spec_log_init = torch.log1p(pred_spec_init)
    vmax_p = pred_spec_log_init.max().item()
    ax5 = axs[row_idx, 4]
    ax5.imshow(pred_spec_log_init, aspect='auto', cmap='jet', origin='lower', vmin=0, vmax=vmax_p*0.9)
    ax5.set_title(f"Shot {shot_id}: Pred Spectra")
    ax5.set_xlabel("Freq Bin")
    ax5.set_yticks([])


save_path = f"{result_folder}/QC_Unified.jpg"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\nQC Plot saved to: {save_path}")
plt.close()

# ===============================================================
# 11) 频散谱差异分析
# ===============================================================
print("\nGenerating Spectrum Difference Analysis...")

fig2, axs2 = plt.subplots(len(target_shots), 3, figsize=(15, 4*len(target_shots)))
if len(target_shots) == 1:
    axs2 = axs2.reshape(1, -1)

for row_idx, shot_id in enumerate(target_shots):
    obs_spec_pack = obs_spectra_storage[shot_id]
    
    if obs_spec_pack['R'] is not None:
        real_spec = obs_spec_pack['R'].detach().cpu()
        direction = 'Right'
    elif obs_spec_pack['L'] is not None:
        real_spec = obs_spec_pack['L'].detach().cpu()
        direction = 'Left'
    else:
        continue
    
    with torch.no_grad():
        ricker_qc = deepwave.wavelets.ricker(qc_freq, nt, dt, 1.5 / qc_freq).to(device)
        out_qc = elastic(
            *deepwave.common.vpvsrho_to_lambmubuoyancy(vp_final_qc, vs_final_qc, rho_final_qc),
            dx, dt,
            source_amplitudes_y=ricker_qc.repeat(1, 1, 1),
            source_locations_y=source_locations[shot_id:shot_id+1],
            receiver_locations_y=receiver_locations[shot_id:shot_id+1],
            pml_width=[0, 20, 20, 20], pml_freq=qc_freq
        )
        pred_seis_tensor = out_qc[-2].squeeze(0)
        pred_L, pred_R, _, _, _ = compute_split_spectra(
            pred_seis_tensor, sx[shot_id], gx, dx, time_tensor,
            GLOBAL_FMAX_PRECOMP, dt, min_traces=2,
            vres=GLOBAL_VRES, fres=GLOBAL_FRES
        )
    
    if direction == 'Right' and pred_R is not None:
        pred_spec = pred_R.cpu()
    elif pred_L is not None:
        pred_spec = pred_L.cpu()
    else:
        continue
    
    real_norm = real_spec / (real_spec.max() + 1e-9)
    pred_norm = pred_spec / (pred_spec.max() + 1e-9)
    
    # 尺寸对齐
    if real_norm.shape != pred_norm.shape:
        pred_norm = torch.nn.functional.interpolate(
            pred_norm.unsqueeze(0).unsqueeze(0),
            size=real_norm.shape,
            mode='bilinear', align_corners=True
        ).squeeze()
    
    diff_spec = real_norm - pred_norm
    
    # 绘制
    ax1 = axs2[row_idx, 0]
    im1 = ax1.imshow(real_norm, aspect='auto', cmap='jet', origin='lower', vmin=0, vmax=1)
    ax1.set_title(f"Shot {shot_id}: Obs Spectrum (Normalized)")
    ax1.set_xlabel("Frequency Bin")
    ax1.set_ylabel("Velocity Bin")
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    ax2 = axs2[row_idx, 1]
    im2 = ax2.imshow(pred_norm, aspect='auto', cmap='jet', origin='lower', vmin=0, vmax=1)
    ax2.set_title(f"Shot {shot_id}: Pred Spectrum (Normalized)")
    ax2.set_xlabel("Frequency Bin")
    ax2.set_yticks([])
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    ax3 = axs2[row_idx, 2]
    vmax_diff = max(0.3, torch.quantile(diff_spec.abs(), 0.95).item())
    im3 = ax3.imshow(diff_spec, aspect='auto', cmap='seismic', origin='lower', 
                      vmin=-vmax_diff, vmax=vmax_diff)
    ax3.set_title(f"Shot {shot_id}: Difference (Obs - Pred)")
    ax3.set_xlabel("Frequency Bin")
    ax3.set_yticks([])
    plt.colorbar(im3, ax=ax3, fraction=0.046)

plt.tight_layout()
plt.savefig(f"{result_folder}/Spectrum_Difference.jpg", dpi=200, bbox_inches='tight')
print("Spectrum Difference saved to: ./Spectrum_Difference.jpg")
plt.close()

# ===============================================================
# 12) 深度剖面多位置对比
# ===============================================================
print("\nGenerating Multi-Profile Comparison...")

fig3, axs3 = plt.subplots(2, 3, figsize=(15, 10))

profile_positions = [nx//6, nx//3, nx//2, 2*nx//3, 5*nx//6]
profile_positions = [p for p in profile_positions if 0 <= p < nx]

# 上排：多个位置的深度剖面
for i, px in enumerate(profile_positions[:3]):
    ax = axs3[0, i]
    ax.plot(vs_true_np[:, px], range(nz), 'b-', linewidth=2.5, label='True')
    ax.plot(vs_init_np[:, px], range(nz), 'g--', linewidth=1.5, label='Initial')
    ax.plot(vs_final[:, px], range(nz), 'r-', linewidth=2, label='Inverted')
    ax.invert_yaxis()
    ax.set_xlabel('Vs (m/s)')
    ax.set_ylabel('Depth (grid)')
    ax.set_title(f'Profile at x={px}')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([float(vsmin)*0.9, float(vsmax)*1.1])

# 下排左：水平剖面对比
ax_h1 = axs3[1, 0]
depth_shallow = nz // 4
ax_h1.plot(vs_true_np[depth_shallow, :], 'b-', linewidth=2, label='True')
ax_h1.plot(vs_init_np[depth_shallow, :], 'g--', linewidth=1.5, label='Initial')
ax_h1.plot(vs_final[depth_shallow, :], 'r-', linewidth=2, label='Inverted')
ax_h1.set_xlabel('X (grid)')
ax_h1.set_ylabel('Vs (m/s)')
ax_h1.set_title(f'Horizontal Profile at z={depth_shallow} (Shallow)')
ax_h1.legend(loc='best', fontsize=8)
ax_h1.grid(True, alpha=0.3)

ax_h2 = axs3[1, 1]
depth_mid = nz // 2
ax_h2.plot(vs_true_np[depth_mid, :], 'b-', linewidth=2, label='True')
ax_h2.plot(vs_init_np[depth_mid, :], 'g--', linewidth=1.5, label='Initial')
ax_h2.plot(vs_final[depth_mid, :], 'r-', linewidth=2, label='Inverted')
ax_h2.set_xlabel('X (grid)')
ax_h2.set_ylabel('Vs (m/s)')
ax_h2.set_title(f'Horizontal Profile at z={depth_mid} (Mid)')
ax_h2.legend(loc='best', fontsize=8)
ax_h2.grid(True, alpha=0.3)

ax_h3 = axs3[1, 2]
depth_deep = 3 * nz // 4
ax_h3.plot(vs_true_np[depth_deep, :], 'b-', linewidth=2, label='True')
ax_h3.plot(vs_init_np[depth_deep, :], 'g--', linewidth=1.5, label='Initial')
ax_h3.plot(vs_final[depth_deep, :], 'r-', linewidth=2, label='Inverted')
ax_h3.set_xlabel('X (grid)')
ax_h3.set_ylabel('Vs (m/s)')
ax_h3.set_title(f'Horizontal Profile at z={depth_deep} (Deep)')
ax_h3.legend(loc='best', fontsize=8)
ax_h3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{result_folder}/Multi_Profile.jpg", dpi=200, bbox_inches='tight')
print("Multi-Profile saved to: ./Multi_Profile.jpg")
plt.close()

# ===============================================================
# 13) 误差统计分析
# ===============================================================
print("\nGenerating Error Statistics...")

fig4, axs4 = plt.subplots(2, 2, figsize=(12, 10))

# 误差直方图
ax1 = axs4[0, 0]
error_flat = (vs_final - vs_true_np).flatten()
ax1.hist(error_flat, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax1.axvline(x=error_flat.mean(), color='orange', linestyle='-', linewidth=2, label=f'Mean: {error_flat.mean():.2f}')
ax1.set_xlabel('Error (m/s)')
ax1.set_ylabel('Count')
ax1.set_title('Error Distribution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 相对误差图
ax2 = axs4[0, 1]
rel_error_map = np.abs(vs_final - vs_true_np) / (np.abs(vs_true_np) + 1e-9) * 100
im2 = ax2.imshow(rel_error_map, aspect='auto', cmap='YlOrRd', vmin=0, vmax=np.percentile(rel_error_map, 95))
ax2.set_title('Relative Error (%)')
ax2.set_xlabel('X (grid)')
ax2.set_ylabel('Z (grid)')
plt.colorbar(im2, ax=ax2, label='%')

# 深度方向平均误差
ax3 = axs4[1, 0]
depth_mean_error = np.mean(np.abs(vs_final - vs_true_np), axis=1)
depth_std_error = np.std(np.abs(vs_final - vs_true_np), axis=1)
depths = np.arange(nz)
ax3.fill_betweenx(depths, depth_mean_error - depth_std_error, 
                   depth_mean_error + depth_std_error, alpha=0.3, color='blue')
ax3.plot(depth_mean_error, depths, 'b-', linewidth=2)
ax3.invert_yaxis()
ax3.set_xlabel('Mean Absolute Error (m/s)')
ax3.set_ylabel('Depth (grid)')
ax3.set_title('Depth-wise Error Statistics')
ax3.grid(True, alpha=0.3)

# 水平方向平均误差
ax4 = axs4[1, 1]
lateral_mean_error = np.mean(np.abs(vs_final - vs_true_np), axis=0)
lateral_std_error = np.std(np.abs(vs_final - vs_true_np), axis=0)
x_pos = np.arange(nx)
ax4.fill_between(x_pos, lateral_mean_error - lateral_std_error,
                  lateral_mean_error + lateral_std_error, alpha=0.3, color='green')
ax4.plot(x_pos, lateral_mean_error, 'g-', linewidth=2)
ax4.set_xlabel('X (grid)')
ax4.set_ylabel('Mean Absolute Error (m/s)')
ax4.set_title('Lateral Error Statistics')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{result_folder}/Error_Statistics.jpg", dpi=200, bbox_inches='tight')
print("Error Statistics saved to: ./Error_Statistics.jpg")
plt.close()

# ===============================================================
# 14) 收敛性分析
# ===============================================================
print("\nGenerating Convergence Analysis...")

fig5, axs5 = plt.subplots(1, 3, figsize=(15, 4))

# Loss曲线 with 阶段标记
ax1 = axs5[0]
ax1.plot(loss_history, 'b-', linewidth=1, alpha=0.7)

# 移动平均
window = 10
if len(loss_history) > window:
    loss_smooth = np.convolve(loss_history, np.ones(window)/window, mode='valid')
    ax1.plot(np.arange(window-1, len(loss_history)), loss_smooth, 'r-', linewidth=2, label='Moving Avg')

# 阶段分界线
iter_acc = 0
colors_stage = ['#E3F2FD', '#E8F5E9', '#FFF3E0', '#FCE4EC']
for i, cfg in enumerate(schedule):
    ax1.axvspan(iter_acc, iter_acc + cfg['iters'], alpha=0.3, color=colors_stage[i % len(colors_stage)])
    ax1.axvline(x=iter_acc, color='gray', linestyle='--', alpha=0.5)
    ax1.text(iter_acc + cfg['iters']//2, ax1.get_ylim()[1]*0.95, f"S{i+1}", 
             ha='center', fontsize=10, fontweight='bold')
    iter_acc += cfg['iters']

ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Convergence with Stages')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 学习率曲线
ax2 = axs5[1]
lr_history = []
iter_acc = 0
for cfg in schedule:
    lr_base = cfg['lr']
    iters = cfg['iters']
    for k in range(iters):
        progress = k / iters
        if cfg['stage'] == 1 and k < 20:
            curr_lr = lr_base * (0.3 + 0.7 * k / 20)
        else:
            curr_lr = lr_base * (0.3 + 0.7 * (0.5 + 0.5 * math.cos(math.pi * progress)))
        lr_history.append(curr_lr)
        
ax2.plot(lr_history, 'g-', linewidth=1.5)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Learning Rate')
ax2.set_title('Learning Rate Schedule')
ax2.grid(True, alpha=0.3)

# 阶段分界线
iter_acc = 0
for i, cfg in enumerate(schedule):
    ax2.axvline(x=iter_acc, color='gray', linestyle='--', alpha=0.5)
    iter_acc += cfg['iters']

# MSE下降曲线
ax3 = axs5[2]
if len(mse_history) > 0:
    ax3.plot(mse_history, 'mo-', linewidth=2, markersize=4)
    ax3.set_xlabel('Checkpoint (every 20 iters)')
    ax3.set_ylabel('MSE')
    ax3.set_title('MSE Evolution')
    ax3.grid(True, alpha=0.3)
    
    # 标注最终MSE
    ax3.annotate(f'Final: {mse_history[-1]:.1f}', 
                  xy=(len(mse_history)-1, mse_history[-1]),
                  xytext=(len(mse_history)-1-5, mse_history[-1]*1.2),
                  arrowprops=dict(arrowstyle='->', color='red'),
                  fontsize=10, color='red')

plt.tight_layout()
plt.savefig(f"{result_folder}/Convergence_Analysis.jpg", dpi=200, bbox_inches='tight')
print("Convergence Analysis saved to: ./Convergence_Analysis.jpg")
plt.close()

# ===============================================================
# 15) 保存结果
# ===============================================================
print("\nSaving Results...")

# 保存模型
np.save(f"{result_folder}/vs_inverted.npy", vs_final)
np.save(f"{result_folder}/vs_true.npy", vs_true_np)
np.save(f"{result_folder}/vs_init.npy", vs_init_np)

# 保存历史
np.save(f"{result_folder}/loss_history.npy", np.array(loss_history))
np.save(f"{result_folder}/mse_history.npy", np.array(mse_history))
np.save(f"{result_folder}/freq_history.npy", np.array(freq_history))

print("Results saved successfully!")

# ===============================================================
# 16) 最终汇总
# ===============================================================
print("\n" + "="*60)
print("                    INVERSION SUMMARY")
print("="*60)
print(f"  Model Size:           {nz} x {nx}")
print(f"  Vs Range:             {float(vsmin):.1f} - {float(vsmax):.1f} m/s")
print(f"  Number of Shots:      {ns}")
print(f"  Number of Receivers:  {ng}")
print(f"  Total Iterations:     {total_iter}")
print(f"  Final Relative Error: {rel_err:.2f}%")
print(f"  Final MSE:            {final_mse:.2f}")
print(f"  Total Time:           {total_time/60:.2f} minutes")
print("="*60)

# 损失模块统计
print_neural_loss_stats()

print("\n[DONE] Unified Deep Learning FWI completed successfully!")
print("="*60)

# ===============================================================
# 17) Optional: 动态GIF生成（如果需要）
# ===============================================================
GENERATE_GIF = False  # 设为True以生成动画

if GENERATE_GIF:
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
        
        print("\nGenerating Evolution GIF...")
        
        # 需要在反演过程中保存中间结果
        # 这里假设我们有保存的快照
        if os.path.exists("./snapshots"):
            snapshot_files = sorted([f for f in os.listdir("./snapshots") if f.endswith('.npy')])
            
            if len(snapshot_files) > 0:
                fig_gif, ax_gif = plt.subplots(figsize=(10, 6))
                
                def update_gif(frame):
                    ax_gif.clear()
                    vs_snap = np.load(f"./snapshots/{snapshot_files[frame]}")
                    im = ax_gif.imshow(vs_snap, aspect='auto', cmap='jet', 
                                        vmin=float(vsmin), vmax=float(vsmax))
                    ax_gif.set_title(f'Iteration: {frame * 20}')
                    return [im]
                
                anim = FuncAnimation(fig_gif, update_gif, frames=len(snapshot_files), 
                                      interval=200, blit=False)
                anim.save('./evolution.gif', writer=PillowWriter(fps=5))
                print("Evolution GIF saved to: ./evolution.gif")
                plt.close()
    except Exception as e:
        print(f"GIF generation failed: {e}")

# ===============================================================
# 18) 清理GPU内存
# ===============================================================
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"\nGPU Memory after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")

print("\n" + "="*60)
print("           ALL TASKS COMPLETED SUCCESSFULLY!")
print("="*60)

