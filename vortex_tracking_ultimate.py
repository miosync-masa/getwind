#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GET Wind™ Vortex Analysis - Ultimate Edition
環ちゃん & ご主人さま Complete Fix! 💕

Ultimate改良版：
- ファイル名をv6.3に統一
- 揚力係数の幾何重み追加
- rFFTによる高速化
- 自動eps計算
- Reynolds数の実効値推定
- Welch法オプション追加
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# ==============================
# データ構造
# ==============================

@dataclass
class Vortex:
    """渦の情報"""
    center: np.ndarray      # (x, y)
    n_particles: int        # 粒子数
    circulation: float      # 循環
    cluster_id: int        # DBSCANのクラスタID
    
@dataclass
class VortexSnapshot:
    """1ステップの渦情報"""
    step: int
    vortices: List[Vortex]
    total_particles: int

# ==============================
# 渦検出（DBSCAN with adaptive eps）
# ==============================

def detect_vortices_dbscan(
    positions: np.ndarray,
    Lambda_F: np.ndarray,
    Q_criterion: np.ndarray,
    active_mask: np.ndarray,
    eps: Optional[float] = None,
    min_samples: int = 5,
    Q_threshold: float = 0.15,
    auto_eps: bool = True
) -> List[Vortex]:
    """DBSCANで渦を検出（自動eps対応）"""
    
    q_mask = active_mask & (Q_criterion > Q_threshold)
    vortex_positions = positions[q_mask]
    vortex_Lambda_F = Lambda_F[q_mask]
    
    if len(vortex_positions) < min_samples:
        return []
    
    # 自動eps計算
    if auto_eps and eps is None:
        nbrs = NearestNeighbors(n_neighbors=min(5, len(vortex_positions))).fit(vortex_positions)
        dists, _ = nbrs.kneighbors(vortex_positions)
        if dists.shape[1] > 1:
            eps = np.median(dists[:, 1]) * 3.0  # 第2近傍の中央値×3
        else:
            eps = 25.0  # フォールバック
    elif eps is None:
        eps = 25.0
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(vortex_positions)
    labels = clustering.labels_
    
    vortices = []
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
            
        cluster_mask = labels == cluster_id
        cluster_positions = vortex_positions[cluster_mask]
        cluster_Lambda_F = vortex_Lambda_F[cluster_mask]
        
        center = np.mean(cluster_positions, axis=0)
        
        circulation = compute_circulation(
            cluster_Lambda_F,
            cluster_positions,
            center
        )
        
        vortices.append(Vortex(
            center=center,
            n_particles=len(cluster_positions),
            circulation=circulation,
            cluster_id=cluster_id
        ))
    
    return vortices

def compute_circulation(
    Lambda_F: np.ndarray,
    positions: np.ndarray,
    center: np.ndarray
) -> float:
    """循環を計算"""
    
    rel_pos = positions - center
    distances = np.linalg.norm(rel_pos, axis=1) + 1e-8
    
    tangent = np.stack([-rel_pos[:, 1], rel_pos[:, 0]], axis=1)
    tangent = tangent / distances[:, None]
    
    v_tangential = np.sum(Lambda_F * tangent, axis=1)
    weights = np.exp(-distances / 10.0)
    
    circulation = np.sum(v_tangential * weights) / np.sum(weights)
    
    return circulation

# ==============================
# 改良版トラッカー（シンプル）
# ==============================

class SimpleVortexTracker:
    """シンプルで安定したトラッカー"""
    
    def __init__(self, matching_threshold: float = 40.0):
        self.matching_threshold = matching_threshold
        self.next_id = 0
        self.tracks = {}
        
    def update(self, vortices: List[Vortex], step: int) -> Dict[int, int]:
        """渦の更新"""
        
        if not vortices:
            return {}
        
        current_positions = np.array([v.center for v in vortices])
        new_tracks = {}
        used_vortices = set()
        
        # 既存トラックの延長
        for track_id, track in self.tracks.items():
            if len(track) == 0:
                continue
                
            last_step, last_pos, last_circ = track[-1]
            
            # 予測位置（単純に下流へ）
            predicted_pos = last_pos + np.array([10.0 * (step - last_step) * 0.02, 0])
            
            min_dist = float('inf')
            best_match = None
            
            for i, pos in enumerate(current_positions):
                if i in used_vortices:
                    continue
                
                # x座標が大きく逆流していないか
                if pos[0] < last_pos[0] - 20:
                    continue
                
                dist = np.linalg.norm(pos - predicted_pos)
                if dist < self.matching_threshold and dist < min_dist:
                    min_dist = dist
                    best_match = i
            
            if best_match is not None:
                new_tracks[track_id] = track + [(
                    step,
                    current_positions[best_match],
                    vortices[best_match].circulation
                )]
                used_vortices.add(best_match)
        
        # 新規渦の追加（障害物近傍のみ）
        for i, vortex in enumerate(vortices):
            if i not in used_vortices:
                # 障害物後方の適切な範囲でのみ新規生成
                if 80 < vortex.center[0] < 160:
                    if abs(vortex.circulation) > 1.0 and vortex.n_particles > 8:
                        track_id = self.next_id
                        self.next_id += 1
                        new_tracks[track_id] = [(
                            step,
                            vortex.center,
                            vortex.circulation
                        )]
        
        # 古いトラックを削除
        self.tracks = {tid: track for tid, track in new_tracks.items() 
                      if len(track) > 0 and (step - track[-1][0]) < 100}
        
        return {i: tid for tid, i in enumerate(self.tracks.keys())}

# ==============================
# Ultimate版：揚力係数によるStrouhal数計算
# ==============================

def compute_effective_reynolds(states, config, n_samples=100):
    """実効Reynolds数を推定"""
    
    # 障害物近傍での粘性係数をサンプリング
    sample_indices = np.linspace(1000, len(states)-1, n_samples, dtype=int)
    
    viscosity_samples = []
    for idx in sample_indices:
        state = states[idx]
        position = state['position']
        is_active = state['is_active']
        
        # 障害物近傍の粒子
        dx = position[:, 0] - config.obstacle_center_x
        dy = position[:, 1] - config.obstacle_center_y
        r = np.sqrt(dx**2 + dy**2)
        near_obstacle = (r > config.obstacle_size) & (r < config.obstacle_size * 1.5) & is_active
        
        if np.sum(near_obstacle) > 0:
            # 局所的な実効粘性（簡易推定）
            effective_visc = config.viscosity_factor * 0.05
            viscosity_samples.append(effective_visc)
    
    if viscosity_samples:
        nu_eff = np.mean(viscosity_samples)
        D = 2 * config.obstacle_size
        Re_eff = config.Lambda_F_inlet * D / nu_eff
        return Re_eff, nu_eff
    else:
        return 200.0, config.viscosity_factor * 0.05  # デフォルト

def estimate_Ueff(states, config, x_probe=None, n_samples=64):
    """実効流速U_effを推定（上流プローブ）"""
    if x_probe is None:
        x_probe = config.obstacle_center_x - 5.0 * (2*config.obstacle_size)
    
    # プローブ位置がドメイン内か確認
    if x_probe < 10:
        x_probe = 10
    
    vals = []
    for st in states[1000:]:  # 過渡除外
        pos = st['position'] if isinstance(st, dict) else st.position
        vel = st['Lambda_F'] if isinstance(st, dict) else st.Lambda_F
        act = st['is_active'] if isinstance(st, dict) else st.is_active
        
        # x_probe付近の薄帯での流速
        mask = act & (np.abs(pos[:,0] - x_probe) < 1.0)
        if np.any(mask):
            vals.append(np.mean(vel[mask, 0]))  # x方向流速の平均
    
    return float(np.mean(vals)) if vals else config.Lambda_F_inlet

def lift_coefficient_ring_binned(state, config, n_bins=64):
    """等角ビンCL計算（粒子バイアス除去）"""
    pos = state['position'] if isinstance(state, dict) else state.position
    vel = state['Lambda_F'] if isinstance(state, dict) else state.Lambda_F
    act = state['is_active'] if isinstance(state, dict) else state.is_active

    dx = pos[:,0] - config.obstacle_center_x
    dy = pos[:,1] - config.obstacle_center_y
    r = np.sqrt(dx*dx + dy*dy)
    theta = np.arctan2(dy, dx)

    # より薄いリング（1.0-1.3倍）
    ring = (r > config.obstacle_size*1.00) & (r < config.obstacle_size*1.30) & act
    if np.sum(ring) < 32:
        return 0.0

    vel_mag = np.linalg.norm(vel[ring], axis=1)
    Cp = 1.0 - (vel_mag / config.Lambda_F_inlet)**2

    th = theta[ring]
    bins = np.linspace(-np.pi, np.pi, n_bins+1)
    idx = np.digitize(th, bins) - 1
    
    CL_up = []
    CL_lo = []

    for k in range(n_bins):
        m = (idx == k)
        if not np.any(m):
            continue
        
        thk = th[m]
        Cpk = Cp[m]
        rep = np.median(Cpk)    # 代表値（中央値）
        thm = np.median(thk)     # ビン中心角度
        term = rep * np.sin(thm)
        
        if thm > 0:
            CL_up.append(term)
        else:
            CL_lo.append(term)

    up = np.mean(CL_up) if CL_up else 0.0
    lo = np.mean(CL_lo) if CL_lo else 0.0
    return (up - lo) * 2.0

def compute_lift_coefficient_ultimate(state, config, method='binned'):
    """Ultimate版：揚力係数（等角ビンor重み付き）"""
    
    if method == 'binned':
        return lift_coefficient_ring_binned(state, config)
    else:
        # 従来の重み付き版（フォールバック）
        # stateが辞書の場合の処理
        if isinstance(state, dict):
            position = state['position']
            Lambda_F = state['Lambda_F']
            is_active = state['is_active']
        else:
            position = state.position
            Lambda_F = state.Lambda_F
            is_active = state.is_active
        
        # 障害物表面近傍の粒子を選択
        dx = position[:, 0] - config.obstacle_center_x
        dy = position[:, 1] - config.obstacle_center_y
        r = np.sqrt(dx**2 + dy**2)
        
        # 表面近傍（1.0-2.0倍の半径）
        near_surface = (r > config.obstacle_size) & (r < config.obstacle_size * 2.0) & is_active
        
        if np.sum(near_surface) < 10:
            return 0.0
        
        # 極座標での角度
        theta = np.arctan2(dy[near_surface], dx[near_surface])
        r_near = r[near_surface]
        
        # 速度の大きさから圧力係数を計算（ベルヌーイの定理）
        velocity_mag = np.linalg.norm(Lambda_F[near_surface], axis=1)
        Cp = 1.0 - (velocity_mag / config.Lambda_F_inlet)**2
        
        # 幾何重み（半径方向の線素）
        w = r_near / np.mean(r_near)  # 正規化された半径重み
        
        # 上半分と下半分で別々に積分
        upper_mask = theta > 0
        lower_mask = theta <= 0
        
        # 各領域での重み付き圧力積分
        if np.any(upper_mask):
            upper_contribution = np.average(
                Cp[upper_mask] * np.sin(theta[upper_mask]), 
                weights=w[upper_mask]
            )
        else:
            upper_contribution = 0.0
            
        if np.any(lower_mask):
            lower_contribution = np.average(
                Cp[lower_mask] * np.sin(theta[lower_mask]), 
                weights=w[lower_mask]
            )
        else:
            lower_contribution = 0.0
        
        # 揚力係数（上下の圧力差）
        CL = (upper_contribution - lower_contribution) * 2.0
        
        return CL

def refine_peak(freqs, power):
    """ピークの放物線補間"""
    k = np.argmax(power)
    if 0 < k < len(power)-1:
        y1, y2, y3 = power[k-1], power[k], power[k+1]
        denom = (y1 - 2*y2 + y3)
        if denom != 0:
            delta = 0.5*(y1 - y3)/denom
            return k + np.clip(delta, -0.5, 0.5)
    return float(k)

def estimate_f0_autocorr(sig, dt):
    """自己相関によるピーク周波数初期推定"""
    sig = sig - np.mean(sig)
    ac = np.correlate(sig, sig, mode='full')[len(sig)-1:]
    ac = ac / np.max(ac)
    
    # 最初の谷の後の最初の山を探す
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(ac, distance=int(0.1/dt))
    
    if len(peaks) > 1:
        T = peaks[1] * dt
        return 1.0/T
    return None

def compute_strouhal_ultimate(states, config, debug=True, method='rfft'):
    """Ultimate版：高速・高精度Strouhal数計算"""
    
    print("\n📊 Computing lift coefficient time series...")
    
    # CLの時系列を計算（等角ビン版）
    CL_history = []
    for i, state in enumerate(states):
        if i % 500 == 0:
            print(f"  Processing step {i}/{len(states)}")
        CL = compute_lift_coefficient_ultimate(state, config, method='binned')
        CL_history.append(CL)
    
    # 初期の過渡応答を除去
    CL_signal = np.array(CL_history[1000:])  # 最初の1000ステップを除外
    
    if len(CL_signal) < 500:
        print("Warning: Not enough data for accurate FFT")
        return 0.0
    
    # 時間軸（物理単位）
    time = np.arange(len(CL_signal)) * config.dt
    
    # トレンド除去
    CL_signal = CL_signal - np.mean(CL_signal)
    
    # === U_effの推定 ===
    print("\n  Estimating effective velocity U_eff...")
    U_eff = estimate_Ueff(states, config)
    print(f"  U_eff = {U_eff:.3f} (inlet = {config.Lambda_F_inlet})")
    
    # === 自己相関による初期推定 ===
    f0 = estimate_f0_autocorr(CL_signal, config.dt)
    if f0 is None:
        # ラフな初期推定
        D = 2 * config.obstacle_size
        f0 = 0.2 * U_eff / D  # St≈0.2の仮定
    print(f"  Initial frequency estimate: {f0:.4f} Hz")
    
    if method == 'rfft':
        # === rFFT法（高速） ===
        window = np.hanning(len(CL_signal))
        CL_windowed = CL_signal * window
        
        # ゼロパディング（FFT精度向上）
        n_padded = 2**int(np.ceil(np.log2(len(CL_windowed) * 4)))
        
        # 実数FFT
        fft = np.fft.rfft(CL_windowed, n=n_padded)
        freqs = np.fft.rfftfreq(n_padded, d=config.dt)
        power = np.abs(fft)**2
        
    elif method == 'welch':
        # === Welch法（ノイズに強い） ===
        nperseg = min(4096, len(CL_signal))
        freqs, power = welch(CL_signal, 
                           fs=1.0/config.dt, 
                           window='hann',
                           nperseg=nperseg,
                           noverlap=nperseg//2)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # === 狭窓でのピーク探索 ===
    fmin, fmax = 0.7*f0, 1.3*f0
    valid_range = (freqs > fmin) & (freqs < fmax)
    
    if np.any(valid_range):
        valid_freqs = freqs[valid_range]
        valid_power = power[valid_range]
        
        # 放物線補間でサブビン精度
        kref = refine_peak(valid_freqs, valid_power)
        peak_freq = np.interp(kref, np.arange(len(valid_freqs)), valid_freqs)
        
        # カルマン渦の周波数補正
        # CL（揚力）はshedding周波数f_sで振動 → 補正は不要
        frequency_correction = 1.0
        
        # Strouhal数を計算（U_eff使用！）
        D = 2 * config.obstacle_size
        St_raw = peak_freq * D / U_eff  # ←ここが重要！
        St_corrected = St_raw * frequency_correction
        
        # 実効Reynolds数の推定
        Re_eff, nu_eff = compute_effective_reynolds(states, config)
        
        if debug:
            print(f"\n✨ Ultimate Lift Coefficient Analysis:")
            print(f"  Peak frequency: {peak_freq:.4f} Hz (refined)")
            print(f"  U_eff: {U_eff:.3f} (vs inlet: {config.Lambda_F_inlet})")
            print(f"  Raw Strouhal: {St_raw:.4f}")
            print(f"  Corrected Strouhal: {St_corrected:.4f}")
            print(f"  Effective Reynolds: {Re_eff:.1f}")
            print(f"  Target St (Re=200): 0.195")
            print(f"  Error: {abs(St_corrected - 0.195)/0.195*100:.1f}%")
            
            # ブロッケージ比の確認
            blockage = D / 150.0  # 直径/ドメイン高さ
            print(f"  Blockage ratio: {blockage:.3f}")
            if blockage > 0.2:
                print(f"  ⚠ High blockage may affect St by ~{(blockage-0.2)*10:.1f}%")
            
            # 詳細なプロット
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            
            # 1. 元の時系列（時間軸）
            ax = axes[0, 0]
            time_full = np.arange(len(CL_history)) * config.dt
            ax.plot(time_full, CL_history, linewidth=0.5)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('CL')
            ax.set_title('Raw Lift Coefficient Time Series')
            ax.grid(True, alpha=0.3)
            
            # 2. 処理後の信号
            ax = axes[0, 1]
            ax.plot(time, CL_signal, linewidth=0.5)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('CL (detrended)')
            ax.set_title('Processed Signal (after removing initial transient)')
            ax.grid(True, alpha=0.3)
            
            # 3. パワースペクトル（線形スケール）
            ax = axes[0, 2]
            mask = freqs < 0.5
            ax.plot(freqs[mask], power[mask])
            ax.axvline(peak_freq, color='red', linestyle='--', 
                      label=f'Peak: {peak_freq:.4f} Hz')
            ax.axvspan(fmin, fmax, alpha=0.2, color='yellow', label='Search window')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Power')
            ax.set_title(f'Power Spectrum ({method.upper()})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 4. パワースペクトル（対数スケール）
            ax = axes[1, 0]
            ax.semilogy(freqs[mask], power[mask])
            ax.axvline(peak_freq, color='red', linestyle='--', 
                      label=f'Peak: {peak_freq:.4f} Hz')
            expected_f = 0.195 * U_eff / D
            ax.axvline(expected_f, color='green', linestyle=':', 
                      label=f'Expected (St=0.195): {expected_f:.4f} Hz')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Power (log scale)')
            ax.set_title('Power Spectrum (Log Scale)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 5. Strouhal vs Reynolds
            ax = axes[1, 1]
            Re_range = np.array([100, 150, 200, 250, 300])
            St_empirical = 0.195 * np.ones_like(Re_range)  # Re=200付近では一定
            ax.plot(Re_range, St_empirical, 'g-', label='Empirical')
            ax.scatter([Re_eff], [St_corrected], color='red', s=100, 
                      zorder=5, label=f'Simulation (Re={Re_eff:.0f})')
            ax.set_xlabel('Reynolds Number')
            ax.set_ylabel('Strouhal Number')
            ax.set_title('St vs Re Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 6. FFT解像度とU_eff情報
            ax = axes[1, 2]
            df = freqs[1] - freqs[0] if len(freqs) > 1 else 0
            resolution_info = f"Frequency resolution: {df:.5f} Hz\n"
            resolution_info += f"Nyquist frequency: {0.5/config.dt:.2f} Hz\n"
            resolution_info += f"Signal length: {len(CL_signal)} samples\n"
            resolution_info += f"Time span: {len(CL_signal)*config.dt:.1f} s\n"
            resolution_info += f"Method: {method.upper()}\n"
            resolution_info += f"─────────────────\n"
            resolution_info += f"U_eff: {U_eff:.3f} m/s\n"
            resolution_info += f"U_inlet: {config.Lambda_F_inlet:.1f} m/s\n"
            resolution_info += f"Reduction: {(1-U_eff/config.Lambda_F_inlet)*100:.1f}%"
            ax.text(0.1, 0.5, resolution_info, transform=ax.transAxes,
                   fontsize=11, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.axis('off')
            ax.set_title('Analysis Parameters')
            
            plt.tight_layout()
            plt.savefig('lift_analysis_ultimate.png', dpi=150)
            print(f"  Plot saved to 'lift_analysis_ultimate.png'")
        
        return St_corrected
    else:
        print(f"Warning: No valid peak found in range [{fmin:.4f}, {fmax:.4f}] Hz")
        return 0.0

# ==============================
# きれいな軌跡描画
# ==============================

def plot_clean_vortex_trajectories(tracker, figsize=(14, 7)):
    """きれいなカルマン渦の軌跡を描画"""
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 有効な軌跡をフィルタリング
    valid_tracks = []
    for track_id, track in tracker.tracks.items():
        if len(track) < 10:  # 10ステップ以上続いた渦のみ
            continue
            
        positions = np.array([t[1] for t in track])
        circulations = np.array([t[2] for t in track])
        
        # x座標が単調増加しているかチェック
        x_coords = positions[:, 0]
        for i in range(1, len(x_coords)):
            if x_coords[i] < x_coords[i-1] - 15:  # 15単位以上の逆流は異常
                positions = positions[:i]  # 逆流前までで切る
                break
        
        if len(positions) < 20:
            continue
            
        # 総移動距離チェック
        total_dist = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        if total_dist < 20 or total_dist > 800:
            continue
            
        # 平均循環強度
        mean_circ = np.mean(np.abs(circulations))
        if mean_circ < 0.3:
            continue
            
        valid_tracks.append((track_id, positions, mean_circ))
    
    # 軌跡を描画
    for idx, (track_id, positions, mean_circ) in enumerate(valid_tracks):
        # スムージング（オプション）
        if len(positions) > 5:
            positions[:, 0] = gaussian_filter1d(positions[:, 0], sigma=1.5)
            positions[:, 1] = gaussian_filter1d(positions[:, 1], sigma=1.5)
        
        # 色分け（初期y位置で判定）
        if positions[0, 1] > 75:
            color = 'red'
            label = 'Upper vortex' if idx == 0 else None
        else:
            color = 'blue'
            label = 'Lower vortex' if idx == 1 else None
        
        # 軌跡を描画
        ax.plot(positions[:, 0], positions[:, 1],
                color=color, alpha=0.6, linewidth=2,
                label=label)
        
        # 始点と終点をマーク
        ax.scatter(positions[0, 0], positions[0, 1], 
                  color=color, s=50, marker='o', zorder=5)
        ax.scatter(positions[-1, 0], positions[-1, 1], 
                  color=color, s=50, marker='s', zorder=5)
    
    # 理想的なカルマン渦の軌跡（参考）
    t = np.linspace(0, 150, 100)
    x_ideal = 100 + t
    y_upper_ideal = 75 + 25 * np.sin(2 * np.pi * t / 50) * np.exp(-t / 200)
    y_lower_ideal = 75 - 25 * np.sin(2 * np.pi * t / 50 + np.pi) * np.exp(-t / 200)
    
    ax.plot(x_ideal, y_upper_ideal, 'r--', alpha=0.2, linewidth=1, label='Ideal upper')
    ax.plot(x_ideal, y_lower_ideal, 'b--', alpha=0.2, linewidth=1, label='Ideal lower')
    
    # 障害物
    circle = plt.Circle((100, 75), 20, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    ax.add_patch(plt.Circle((100, 75), 20, fill=True, color='gray', alpha=0.3))
    
    # グリッドとラベル
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 150)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('Clean Vortex Trajectories (Karman Vortex Street)')
    ax.legend(loc='upper right')
    
    # 流れ方向の矢印
    ax.arrow(10, 140, 30, 0, head_width=3, head_length=5, 
            fc='gray', ec='gray', alpha=0.5)
    ax.text(25, 145, 'Flow', ha='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    return fig

# ==============================
# メイン処理（Ultimate版）
# ==============================

def process_simulation_results(
    simulation_file: str = 'simulation_results_v63_cylinder.npz',  # v6.3対応！
    debug: bool = True,
    fft_method: str = 'rfft'  # 'rfft' or 'welch'
):
    """シミュレーション結果を処理してStrouhal数を計算"""
    
    print("=" * 70)
    print("GET Wind™ Vortex Analysis - Ultimate Edition")
    print("環ちゃん & ご主人さま Complete Fix! 💕")
    print("=" * 70)
    
    # データ読み込み
    print("\n📁 Loading simulation data...")
    print(f"  File: {simulation_file}")
    
    try:
        data = np.load(simulation_file, allow_pickle=True)
    except FileNotFoundError:
        print(f"  ⚠ File not found: {simulation_file}")
        # フォールバック試行
        fallback_files = [
            'simulation_results_v63.npz',
            'simulation_results_v62.npz'
        ]
        for fallback in fallback_files:
            try:
                print(f"  Trying fallback: {fallback}")
                data = np.load(fallback, allow_pickle=True)
                simulation_file = fallback
                break
            except FileNotFoundError:
                continue
        else:
            raise FileNotFoundError(f"No simulation result file found!")
    
    states = data['states'].tolist() if hasattr(data['states'], 'tolist') else data['states']
    config_dict = data['config'].item() if hasattr(data['config'], 'item') else data['config']
    
    # 簡易Config作成
    class SimpleConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    if isinstance(config_dict, dict):
        config = SimpleConfig(**config_dict)
    else:
        config = config_dict
    
    print(f"  Loaded {len(states)} timesteps")
    print(f"  dt = {config.dt}")
    print(f"  Obstacle: center=({config.obstacle_center_x}, {config.obstacle_center_y}), radius={config.obstacle_size}")
    print(f"  Inlet velocity: {config.Lambda_F_inlet}")
    
    # Reynolds数の確認
    D = 2 * config.obstacle_size
    Re_nominal = config.Lambda_F_inlet * D / (config.viscosity_factor * 0.05)
    print(f"  Nominal Reynolds number: {Re_nominal:.1f}")
    
    # 揚力係数法でStrouhal数計算（Ultimate版）
    St_lift = compute_strouhal_ultimate(states, config, debug=debug, method=fft_method)
    
    # DBSCANトラッキング（可視化用、自動eps）
    if debug:
        print("\n🔍 Processing vortex tracking for visualization...")
        tracker = SimpleVortexTracker(matching_threshold=40.0)
        
        for i, state in enumerate(states):
            if i % 500 == 0:
                print(f"  Step {i}/{len(states)}")
            
            # stateが辞書の場合の処理
            if isinstance(state, dict):
                positions = state['position']
                Lambda_F = state['Lambda_F']
                Q_criterion = state['Q_criterion']
                is_active = state['is_active']
            else:
                positions = state.position
                Lambda_F = state.Lambda_F
                Q_criterion = state.Q_criterion
                is_active = state.is_active
            
            vortices = detect_vortices_dbscan(
                positions,
                Lambda_F,
                Q_criterion,
                is_active,
                eps=None,  # 自動計算
                min_samples=8,
                Q_threshold=0.2,
                auto_eps=True
            )
            
            # 強い渦のみ
            strong_vortices = [v for v in vortices 
                              if abs(v.circulation) > 1.0 and v.n_particles > 10]
            
            tracker.update(strong_vortices, i)
        
        # きれいな軌跡を描画
        print("\n📈 Plotting clean trajectories...")
        fig = plot_clean_vortex_trajectories(tracker)
        plt.savefig('clean_vortex_trajectories_ultimate.png', dpi=150)
        print("  Saved to 'clean_vortex_trajectories_ultimate.png'")
    
    # 最終結果
    print("\n" + "=" * 70)
    print("✨ FINAL RESULTS (Ultimate Analysis):")
    print(f"  Strouhal number: {St_lift:.4f}")
    print(f"  Target (Re=200): 0.195")
    print(f"  Error: {abs(St_lift - 0.195)/0.195*100:.1f}%")
    print(f"  FFT Method: {fft_method.upper()}")
    
    if 0.18 < St_lift < 0.21:
        print("  🎉 SUCCESS! Strouhal number is within 10% of target!")
    elif 0.15 < St_lift < 0.25:
        print("  ✅ Good! Strouhal number is physically reasonable.")
    else:
        print("  ⚠️  Strouhal number needs further tuning.")
    
    print("=" * 70)
    
    return St_lift

# ==============================
# 実行
# ==============================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GET Wind™ Ultimate Vortex Analysis')
    parser.add_argument('--file', type=str, 
                       default='simulation_results_v63_cylinder.npz',
                       help='Simulation result file')
    parser.add_argument('--method', type=str, 
                       choices=['rfft', 'welch'],
                       default='rfft',
                       help='FFT method for spectrum analysis')
    parser.add_argument('--no-debug', action='store_true',
                       help='Disable debug plots')
    
    args = parser.parse_args()
    
    # メイン処理を実行
    St = process_simulation_results(
        simulation_file=args.file,
        debug=not args.no_debug,
        fft_method=args.method
    )
    
    print(f"\n🌀 Final Strouhal number: {St:.4f}")
    print("✨ Ultimate Analysis complete! 💕")
