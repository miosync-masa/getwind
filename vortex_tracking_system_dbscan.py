#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GET Wind™ Vortex Tracking System - DBSCAN Edition v2
環ちゃん & ご主人さま Ultimate Edition! 💕

改良点：
- ハンガリアン法による最適マッチング
- 揚力係数（CL）によるStrouhal数計算
- より正確な軌跡追跡
"""

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

# ==============================
# データ構造（超シンプル！）
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
# 渦検出（DBSCAN使用）
# ==============================

def detect_vortices_dbscan(
    positions: np.ndarray,
    Lambda_F: np.ndarray,
    Q_criterion: np.ndarray,
    active_mask: np.ndarray,
    eps: float = 15.0,      # 近傍半径
    min_samples: int = 3,   # 最小粒子数
    Q_threshold: float = 0.1
) -> List[Vortex]:
    """DBSCANで渦を検出"""
    
    # Q > threshold の粒子だけ抽出
    q_mask = active_mask & (Q_criterion > Q_threshold)
    vortex_positions = positions[q_mask]
    vortex_Lambda_F = Lambda_F[q_mask]
    
    if len(vortex_positions) < min_samples:
        return []
    
    # DBSCAN実行
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(vortex_positions)
    labels = clustering.labels_
    
    # 各クラスタ = 渦
    vortices = []
    for cluster_id in set(labels):
        if cluster_id == -1:  # ノイズは無視
            continue
            
        # このクラスタの粒子
        cluster_mask = labels == cluster_id
        cluster_positions = vortex_positions[cluster_mask]
        cluster_Lambda_F = vortex_Lambda_F[cluster_mask]
        
        # 中心計算
        center = np.mean(cluster_positions, axis=0)
        
        # 循環計算
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
    """循環を計算（物理的に正しい版）"""
    
    # 中心からの相対位置
    rel_pos = positions - center
    distances = np.linalg.norm(rel_pos, axis=1) + 1e-8
    
    # 接線ベクトル（反時計回りを正とする基準）
    tangent = np.stack([-rel_pos[:, 1], rel_pos[:, 0]], axis=1)
    tangent = tangent / distances[:, None]
    
    # v·t （速度と接線の内積）
    v_tangential = np.sum(Lambda_F * tangent, axis=1)
    
    # 距離で重み付け
    weights = np.exp(-distances / 10.0)
    
    # 重み付き平均
    # circulation > 0: 反時計回り（CCW）
    # circulation < 0: 時計回り（CW）
    circulation = np.sum(v_tangential * weights) / np.sum(weights)
    
    return circulation

# ==============================
# 渦のフィルタリング（新規追加！）
# ==============================

def filter_strong_vortices(
    vortices: List[Vortex],
    min_circulation: float = 0.5,
    min_particles: int = 5,
    x_max: float = 250.0  # 障害物から離れすぎた渦は除外
) -> List[Vortex]:
    """強い渦のみを抽出"""
    
    strong_vortices = []
    for vortex in vortices:
        if (abs(vortex.circulation) >= min_circulation and
            vortex.n_particles >= min_particles and
            vortex.center[0] <= x_max):
            strong_vortices.append(vortex)
    
    return strong_vortices


# ==============================
# 改良版トラッカー（ハンガリアン法）
# ==============================

class VortexTracker:
    """改良版：渦の軌跡を正確に追跡"""
    
    def __init__(self, matching_threshold: float = 50.0):
        self.matching_threshold = matching_threshold
        self.next_id = 0
        self.tracks = {}
        self.active_ids = {}
        self.velocities = {}  # 渦の移動速度を記録
        
    def update(self, snapshot: VortexSnapshot) -> Dict[int, int]:
        """ハンガリアン法で最適マッチング"""
        
        if not snapshot.vortices:
            self.active_ids = {}
            return {}
        
        current_centers = np.array([v.center for v in snapshot.vortices])
        new_active_ids = {}
        
        if self.active_ids:
            # 前フレームの渦情報
            prev_centers = []
            prev_ids = []
            
            for cluster_id, vortex_id in self.active_ids.items():
                if vortex_id in self.tracks:
                    track = self.tracks[vortex_id]
                    last_pos = track[-1][1]
                    
                    # 速度ベースの予測
                    if vortex_id in self.velocities:
                        velocity = self.velocities[vortex_id]
                    else:
                        velocity = np.array([10.0, 0])  # デフォルト速度
                    
                    # 予測位置
                    predicted = last_pos + velocity
                    prev_centers.append(predicted)
                    prev_ids.append(vortex_id)
            
            if prev_centers:
                prev_centers = np.array(prev_centers)
                
                # コスト行列（距離）
                cost_matrix = cdist(current_centers, prev_centers)
                
                # 閾値を超える場合は高コスト
                cost_matrix[cost_matrix > self.matching_threshold] = 10000
                
                # ハンガリアン法で最適割り当て
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                matched_current = set()
                for i, j in zip(row_ind, col_ind):
                    if cost_matrix[i, j] < self.matching_threshold:
                        vortex_id = prev_ids[j]
                        new_active_ids[i] = vortex_id
                        matched_current.add(i)
                        
                        # 軌跡更新
                        old_pos = self.tracks[vortex_id][-1][1]
                        new_pos = snapshot.vortices[i].center
                        
                        self.tracks[vortex_id].append((
                            snapshot.step,
                            new_pos,
                            snapshot.vortices[i].circulation
                        ))
                        
                        # 速度更新
                        if snapshot.step > self.tracks[vortex_id][-2][0]:
                            dt = snapshot.step - self.tracks[vortex_id][-2][0]
                            self.velocities[vortex_id] = (new_pos - old_pos) / dt
                
                # マッチしなかった渦は新規（厳しい条件）
                for i, vortex in enumerate(snapshot.vortices):
                    if i not in matched_current:
                        # 強い渦のみ新規登録
                        if abs(vortex.circulation) > 1.0 and vortex.n_particles > 8:
                            vortex_id = self.next_id
                            self.next_id += 1
                            new_active_ids[i] = vortex_id
                            
                            self.tracks[vortex_id] = [(
                                snapshot.step,
                                vortex.center,
                                vortex.circulation
                            )]
                            self.velocities[vortex_id] = np.array([10.0, 0])
        else:
            # 初回は全て新規
            for i, vortex in enumerate(snapshot.vortices):
                if abs(vortex.circulation) > 1.0:
                    vortex_id = self.next_id
                    self.next_id += 1
                    new_active_ids[i] = vortex_id
                    
                    self.tracks[vortex_id] = [(
                        snapshot.step,
                        vortex.center,
                        vortex.circulation
                    )]
                    self.velocities[vortex_id] = np.array([10.0, 0])
        
        self.active_ids = new_active_ids
        return new_active_ids

# ==============================
# 揚力係数によるStrouhal数計算
# ==============================

def compute_lift_coefficient(
    state,  # ParticleState
    config  # GETWindConfig
) -> float:
    """揚力係数CLを計算"""
    
    # 障害物近傍の粒子を抽出
    dx = state.position[:, 0] - config.obstacle_center_x
    dy = state.position[:, 1] - config.obstacle_center_y
    r = np.sqrt(dx**2 + dy**2)
    
    # 表面近傍（1.0-1.5倍の半径）
    near_surface = (r > config.obstacle_size) & (r < config.obstacle_size * 1.5) & state.is_active
    
    if np.sum(near_surface) < 10:
        return 0.0
    
    # 角度
    theta = np.arctan2(dy[near_surface], dx[near_surface])
    
    # 各粒子の圧力寄与（簡易モデル）
    # 圧力 ∝ (1 - |v|²/U∞²)
    velocity_mag = np.linalg.norm(state.Lambda_F[near_surface], axis=1)
    Cp = 1.0 - (velocity_mag / config.Lambda_F_inlet)**2
    
    # 揚力への寄与（y方向成分）
    # dL = -p * sin(θ) * ds
    lift_contributions = -Cp * np.sin(theta)
    
    # 平均化
    CL = np.mean(lift_contributions) * 2.0  # 係数調整
    
    return CL

def compute_strouhal_from_lift(
    states,  # List of ParticleState
    config,  # GETWindConfig
    debug: bool = True
) -> float:
    """揚力係数の振動からStrouhal数を計算"""
    
    print("Computing lift coefficient time series...")
    
    # CLの時系列を計算
    CL_history = []
    for i, state in enumerate(states):
        if i % 100 == 0:
            print(f"  Processing step {i}/{len(states)}")
        CL = compute_lift_coefficient(state, config)
        CL_history.append(CL)
    
    # 信号処理
    CL_signal = np.array(CL_history)
    
    # トレンド除去
    CL_signal = CL_signal - np.mean(CL_signal)
    
    # ゼロパディング（FFT精度向上）
    n_original = len(CL_signal)
    n_padded = 2**int(np.ceil(np.log2(n_original * 2)))
    CL_padded = np.zeros(n_padded)
    CL_padded[:n_original] = CL_signal
    
    # FFT実行
    fft = np.fft.fft(CL_padded)
    freqs = np.fft.fftfreq(n_padded, config.dt)
    
    # パワースペクトル
    power = np.abs(fft)**2
    
    # 正の周波数のみ
    positive_mask = (freqs > 0) & (freqs < 1.0)  # 0-1Hz
    positive_freqs = freqs[positive_mask]
    positive_power = power[positive_mask]
    
    # 物理的に妥当な範囲でピーク探索
    valid_range = (positive_freqs > 0.02) & (positive_freqs < 0.2)
    
    if np.any(valid_range):
        valid_freqs = positive_freqs[valid_range]
        valid_power = positive_power[valid_range]
        
        # 最大ピーク
        peak_idx = np.argmax(valid_power)
        peak_freq = valid_freqs[peak_idx]
        
        # Strouhal数
        D = 2 * config.obstacle_size
        St = peak_freq * D / config.Lambda_F_inlet
        
        if debug:
            print(f"\n📊 Lift Coefficient Method:")
            print(f"  Peak frequency: {peak_freq:.4f} Hz")
            print(f"  Strouhal number: {St:.4f}")
            print(f"  Target St (Re=200): 0.195")
            print(f"  Error: {abs(St - 0.195)/0.195*100:.1f}%")
            
            # スペクトルをプロット
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(CL_history)
            plt.xlabel('Step')
            plt.ylabel('CL')
            plt.title('Lift Coefficient Time Series')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.semilogy(positive_freqs[positive_freqs < 0.5], 
                        positive_power[positive_freqs < 0.5])
            plt.axvline(peak_freq, color='red', linestyle='--', 
                       label=f'Peak: {peak_freq:.3f} Hz')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Power')
            plt.title('Power Spectrum')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('lift_coefficient_analysis.png', dpi=150)
            print(f"  Plot saved to 'lift_coefficient_analysis.png'")
        
        return St
    else:
        print("Warning: No valid peak found in spectrum")
        return 0.0

# ==============================
# サブ関数
# ==============================

def analyze_snapshots(snapshots: List[VortexSnapshot]) -> Dict:
    """スナップショットから統計を計算"""
    
    # 渦数の時系列
    n_vortices = [len(s.vortices) for s in snapshots]
    
    # 上下の渦を分離
    upper_counts = []
    lower_counts = []
    
    for snapshot in snapshots:
        upper = sum(1 for v in snapshot.vortices if v.center[1] > 75)
        lower = sum(1 for v in snapshot.vortices if v.center[1] <= 75)
        upper_counts.append(upper)
        lower_counts.append(lower)
    
    return {
        'n_vortices': n_vortices,
        'upper_counts': upper_counts,
        'lower_counts': lower_counts,
        'steps': [s.step for s in snapshots]
    }
    
# ==============================
# 可視化
# ==============================

def visualize_snapshot(snapshot: VortexSnapshot, ax=None):
    """1つのスナップショットを可視化"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    for vortex in snapshot.vortices:
        # 渦の中心
        ax.scatter(vortex.center[0], vortex.center[1], 
                  s=vortex.n_particles*10,  # サイズは粒子数
                  c='red' if vortex.center[1] > 75 else 'blue',
                  alpha=0.6)
        
        # 循環の向きを矢印で表示
        if vortex.circulation > 0:
            marker = '⟲'  # 反時計回り
        else:
            marker = '⟳'  # 時計回り
        ax.text(vortex.center[0], vortex.center[1], marker, 
               fontsize=12, ha='center', va='center')
    
    # 障害物
    circle = plt.Circle((100, 75), 20, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 150)
    ax.set_aspect('equal')
    ax.set_title(f'Step {snapshot.step}: {len(snapshot.vortices)} vortices')
    
    return ax

def plot_vortex_timeline(snapshots: List[VortexSnapshot], tracker: VortexTracker):
    """渦の時系列プロット"""
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 上: 渦数の時間変化
    stats = analyze_snapshots(snapshots)
    ax = axes[0]
    ax.plot(stats['steps'], stats['n_vortices'], 'k-', label='Total', alpha=0.5)
    ax.plot(stats['steps'], stats['upper_counts'], 'r-', label='Upper')
    ax.plot(stats['steps'], stats['lower_counts'], 'b-', label='Lower')
    ax.set_ylabel('Number of Vortices')
    ax.set_xlabel('Step')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 下: 渦の軌跡（強い渦のみ）
    ax = axes[1]
    for vortex_id, track in tracker.tracks.items():
        if len(track) > 5:  # 短い軌跡は除外
            # 最大循環をチェック
            max_circulation = max(abs(t[2]) for t in track)
            if max_circulation > 0.5:  # 強い渦のみ表示
                positions = np.array([t[1] for t in track])
                # 上下で色分け
                color = 'red' if positions[0, 1] > 75 else 'blue'
                ax.plot(positions[:, 0], positions[:, 1], 
                       color=color, alpha=0.6, linewidth=1.5)
    
    # 障害物
    circle = plt.Circle((100, 75), 20, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 150)
    ax.set_aspect('equal')
    ax.set_title('Vortex Trajectories (Strong Vortices Only)')
    
    plt.tight_layout()
    return fig

# ==============================
# メイン処理
# ==============================

def process_simulation_data(
    states,
    config,
    save_file: str = 'vortex_analysis_v2.npz'
):
    """改良版処理（揚力係数法付き）"""
    
    print("=" * 70)
    print("GET Wind™ Vortex Tracking v2 - Ultimate Edition")
    print("環ちゃん & ご主人さま 💕")
    print("=" * 70)
    
    # 揚力係数法でStrouhal数計算
    St_lift = compute_strouhal_from_lift(states, config, debug=True)
    
    # DBSCAN + トラッキング
    print("\nProcessing with DBSCAN...")
    snapshots = []
    tracker = VortexTracker(matching_threshold=50.0)  # 緩めの閾値
    
    for i, state in enumerate(states):
        if i % 500 == 0:
            print(f"  Step {i}/{len(states)}")
        
        vortices = detect_vortices_dbscan(
            state.position,
            state.Lambda_F,
            state.Q_criterion,
            state.is_active,
            eps=20.0,  # 少し大きめ
            min_samples=5,
            Q_threshold=0.15
        )
        
        # 強い渦のみフィルタ
        strong_vortices = filter_strong_vortices(
            vortices,
            min_circulation=0.8,
            min_particles=5
        )
        
        snapshot = VortexSnapshot(
            step=i,
            vortices=strong_vortices,
            total_particles=np.sum(state.is_active)
        )
        snapshots.append(snapshot)
        tracker.update(snapshot)
    
    # 結果まとめ
    print("\n" + "=" * 70)
    print("✨ RESULTS:")
    print(f"  Strouhal (Lift method):  {St_lift:.4f}")
    print(f"  Total vortices tracked:  {tracker.next_id}")
    print(f"  Target St (Re=200):      0.195")
    print("=" * 70)
    
    # グラフ作成
    plot_vortex_timeline(snapshots, tracker)
    plt.savefig('vortex_timeline_v2.png', dpi=150)
    
    return snapshots, tracker, St_lift

# ==============================
# テスト
# ==============================

if __name__ == "__main__":
    print("✨ GET Wind™ Vortex Tracking - DBSCAN Edition (Fixed) ✨")
    print("環ちゃん & ご主人さま Super Simple! 💕")
    print("\nFeatures:")
    print("  • DBSCAN clustering for vortex detection")
    print("  • Fixed Strouhal number calculation")
    print("  • Strong vortex filtering")
    print("  • Automatic period detection")
    print("  • < 600 lines of code!")
