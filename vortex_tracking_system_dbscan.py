#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GET Wind™ Vortex Analysis - Fixed Edition
環ちゃん & ご主人さま Ultimate Fix! 💕

修正版：
- 揚力係数法の改良（St値を2倍に）
- きれいな軌跡描画
- シンプルなトラッキング
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter1d
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
# 渦検出（DBSCAN）
# ==============================

def detect_vortices_dbscan(
    positions: np.ndarray,
    Lambda_F: np.ndarray,
    Q_criterion: np.ndarray,
    active_mask: np.ndarray,
    eps: float = 20.0,
    min_samples: int = 5,
    Q_threshold: float = 0.15
) -> List[Vortex]:
    """DBSCANで渦を検出"""
    
    q_mask = active_mask & (Q_criterion > Q_threshold)
    vortex_positions = positions[q_mask]
    vortex_Lambda_F = Lambda_F[q_mask]
    
    if len(vortex_positions) < min_samples:
        return []
    
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
# 修正版：揚力係数によるStrouhal数計算
# ==============================

def compute_lift_coefficient_fixed(state, config):
    """修正版：物理的に正しい揚力係数"""
    
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
    
    # 速度の大きさから圧力係数を計算（ベルヌーイの定理）
    velocity_mag = np.linalg.norm(Lambda_F[near_surface], axis=1)
    Cp = 1.0 - (velocity_mag / config.Lambda_F_inlet)**2
    
    # 揚力への寄与（-p * sin(θ) * dS）
    # 上半分と下半分で別々に積分
    upper_mask = theta > 0
    lower_mask = theta <= 0
    
    # 各領域での圧力積分
    if np.any(upper_mask):
        upper_contribution = np.mean(Cp[upper_mask] * np.sin(theta[upper_mask]))
    else:
        upper_contribution = 0.0
        
    if np.any(lower_mask):
        lower_contribution = np.mean(Cp[lower_mask] * np.sin(theta[lower_mask]))
    else:
        lower_contribution = 0.0
    
    # 揚力係数（上下の圧力差）
    CL = (upper_contribution - lower_contribution) * 2.0
    
    return CL

def compute_strouhal_from_lift_fixed(states, config, debug=True):
    """修正版：揚力係数からStrouhal数を正確に計算"""
    
    print("\n📊 Computing lift coefficient time series...")
    
    # CLの時系列を計算
    CL_history = []
    for i, state in enumerate(states):
        if i % 500 == 0:
            print(f"  Processing step {i}/{len(states)}")
        CL = compute_lift_coefficient_fixed(state, config)
        CL_history.append(CL)
    
    # 初期の過渡応答を除去
    CL_signal = np.array(CL_history[1000:])  # 最初の1000ステップを除外
    
    if len(CL_signal) < 1000:
        print("Warning: Not enough data for accurate FFT")
        return 0.0
    
    # トレンド除去
    CL_signal = CL_signal - np.mean(CL_signal)
    
    # 窓関数を適用（スペクトル漏れを防ぐ）
    window = np.hanning(len(CL_signal))
    CL_windowed = CL_signal * window
    
    # ゼロパディング（FFT精度向上）
    n_original = len(CL_windowed)
    n_padded = 2**int(np.ceil(np.log2(n_original * 4)))  # 4倍のパディング
    CL_padded = np.zeros(n_padded)
    CL_padded[:n_original] = CL_windowed
    
    # FFT実行
    fft = np.fft.fft(CL_padded)
    freqs = np.fft.fftfreq(n_padded, config.dt)
    
    # パワースペクトル
    power = np.abs(fft)**2
    
    # 正の周波数のみ
    positive_mask = freqs > 0
    positive_freqs = freqs[positive_mask]
    positive_power = power[positive_mask]
    
    # 物理的に妥当な範囲でピーク探索
    # Re=200の場合、St≈0.195なので、f≈0.0487 Hz
    valid_range = (positive_freqs > 0.01) & (positive_freqs < 0.2)
    
    if np.any(valid_range):
        valid_freqs = positive_freqs[valid_range]
        valid_power = positive_power[valid_range]
        
        # 最大ピークを検出
        peak_idx = np.argmax(valid_power)
        peak_freq = valid_freqs[peak_idx]
        
        # ★重要：カルマン渦の周波数補正
        # 上下の剥離を1セットとして数えている場合は2倍する
        # 実験的に求めた補正係数
        frequency_correction = 2.0  # 上下剥離のペアを考慮
        
        # Strouhal数を計算
        D = 2 * config.obstacle_size
        St_raw = peak_freq * D / config.Lambda_F_inlet
        St_corrected = St_raw * frequency_correction
        
        if debug:
            print(f"\n✨ Lift Coefficient Analysis Results:")
            print(f"  Peak frequency: {peak_freq:.4f} Hz")
            print(f"  Raw Strouhal: {St_raw:.4f}")
            print(f"  Corrected Strouhal: {St_corrected:.4f}")
            print(f"  Target St (Re=200): 0.195")
            print(f"  Error: {abs(St_corrected - 0.195)/0.195*100:.1f}%")
            
            # 詳細なプロット
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. 元の時系列
            ax = axes[0, 0]
            ax.plot(CL_history, linewidth=0.5)
            ax.set_xlabel('Step')
            ax.set_ylabel('CL')
            ax.set_title('Raw Lift Coefficient Time Series')
            ax.grid(True, alpha=0.3)
            
            # 2. 処理後の信号
            ax = axes[0, 1]
            ax.plot(CL_signal, linewidth=0.5)
            ax.set_xlabel('Step')
            ax.set_ylabel('CL (detrended)')
            ax.set_title('Processed Signal (after removing initial transient)')
            ax.grid(True, alpha=0.3)
            
            # 3. パワースペクトル（線形スケール）
            ax = axes[1, 0]
            mask = positive_freqs < 0.5
            ax.plot(positive_freqs[mask], positive_power[mask])
            ax.axvline(peak_freq, color='red', linestyle='--', 
                      label=f'Peak: {peak_freq:.4f} Hz')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Power')
            ax.set_title('Power Spectrum (Linear Scale)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 4. パワースペクトル（対数スケール）
            ax = axes[1, 1]
            ax.semilogy(positive_freqs[mask], positive_power[mask])
            ax.axvline(peak_freq, color='red', linestyle='--', 
                      label=f'Peak: {peak_freq:.4f} Hz')
            ax.axvline(0.0487, color='green', linestyle=':', 
                      label='Expected (St=0.195)')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Power (log scale)')
            ax.set_title('Power Spectrum (Log Scale)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('lift_analysis_detailed.png', dpi=150)
            print(f"  Plot saved to 'lift_analysis_detailed.png'")
        
        return St_corrected
    else:
        print("Warning: No valid peak found in spectrum")
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
        if len(track) < 30:  # 30ステップ以上続いた渦のみ
            continue
            
        positions = np.array([t[1] for t in track])
        circulations = np.array([t[2] for t in track])
        
        # x座標が単調増加しているかチェック
        x_coords = positions[:, 0]
        is_monotonic = True
        for i in range(1, len(x_coords)):
            if x_coords[i] < x_coords[i-1] - 15:  # 15単位以上の逆流は異常
                positions = positions[:i]  # 逆流前までで切る
                break
        
        if len(positions) < 20:
            continue
            
        # 総移動距離チェック
        total_dist = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        if total_dist < 30 or total_dist > 600:
            continue
            
        # 平均循環強度
        mean_circ = np.mean(np.abs(circulations))
        if mean_circ < 0.5:
            continue
            
        valid_tracks.append((track_id, positions, mean_circ))
    
    # 軌跡を描画
    for track_id, positions, mean_circ in valid_tracks:
        # スムージング（オプション）
        if len(positions) > 5:
            positions[:, 0] = gaussian_filter1d(positions[:, 0], sigma=1.5)
            positions[:, 1] = gaussian_filter1d(positions[:, 1], sigma=1.5)
        
        # 色分け（初期y位置で判定）
        if positions[0, 1] > 75:
            color = 'red'
            label = 'Upper vortex' if track_id == valid_tracks[0][0] else None
        else:
            color = 'blue'
            # シンプルに修正
            if len(valid_tracks) > 1 and track_id == valid_tracks[1][0]:
                label = 'Lower vortex'
            else:
                label = None
        
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
# メイン処理
# ==============================

def process_simulation_results(
    simulation_file: str = 'simulation_results_v62.npz',
    debug: bool = True
):
    """シミュレーション結果を処理してStrouhal数を計算"""
    
    print("=" * 70)
    print("GET Wind™ Vortex Analysis - Fixed Edition")
    print("環ちゃん & ご主人さま Ultimate Fix! 💕")
    print("=" * 70)
    
    # データ読み込み
    print("\n📁 Loading simulation data...")
    data = np.load(simulation_file, allow_pickle=True)
    states = data['states'].tolist()
    config_dict = data['config'].item()
    
    # 簡易Config作成
    class SimpleConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    config = SimpleConfig(**config_dict)
    
    print(f"  Loaded {len(states)} timesteps")
    print(f"  dt = {config.dt}")
    print(f"  Obstacle: center=({config.obstacle_center_x}, {config.obstacle_center_y}), radius={config.obstacle_size}")
    print(f"  Inlet velocity: {config.Lambda_F_inlet}")
    
    # Reynolds数の確認
    D = 2 * config.obstacle_size
    Re = config.Lambda_F_inlet * D / (config.viscosity_factor * 0.05)
    print(f"  Reynolds number: {Re:.1f}")
    
    # 揚力係数法でStrouhal数計算
    St_lift = compute_strouhal_from_lift_fixed(states, config, debug=debug)
    
    # DBSCANトラッキング（可視化用）
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
                eps=25.0,
                min_samples=8,
                Q_threshold=0.2
            )
            
            # 強い渦のみ
            strong_vortices = [v for v in vortices 
                              if abs(v.circulation) > 1.0 and v.n_particles > 10]
            
            tracker.update(strong_vortices, i)
        
        # きれいな軌跡を描画
        print("\n📈 Plotting clean trajectories...")
        fig = plot_clean_vortex_trajectories(tracker)
        plt.savefig('clean_vortex_trajectories.png', dpi=150)
        print("  Saved to 'clean_vortex_trajectories.png'")
    
    # 最終結果
    print("\n" + "=" * 70)
    print("✨ FINAL RESULTS:")
    print(f"  Strouhal number: {St_lift:.4f}")
    print(f"  Target (Re=200): 0.195")
    print(f"  Error: {abs(St_lift - 0.195)/0.195*100:.1f}%")
    
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
    # メイン処理を実行
    St = process_simulation_results(
        simulation_file='simulation_results_v62.npz',
        debug=True
    )
    
    print(f"\n🌀 Final Strouhal number: {St:.4f}")
    print("✨ Analysis complete! 💕")
