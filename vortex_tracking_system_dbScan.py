#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GET Wind™ Vortex Tracking System - DBSCAN Edition (Fixed)
環ちゃん & ご主人さま Super Simple Edition! 💕

修正版：
- Strouhal数の計算修正
- 強い渦のフィルタリング追加
- より正確な周期検出
"""

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
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
# 軌跡追跡（フレーム間マッチング）
# ==============================

class VortexTracker:
    """渦の軌跡を追跡"""
    
    def __init__(self, matching_threshold: float = 30.0):
        self.matching_threshold = matching_threshold
        self.next_id = 0
        self.tracks = {}  # {vortex_id: [(step, center, circulation), ...]}
        self.active_ids = {}  # 現在のクラスタID → 渦ID
        
    def update(self, snapshot: VortexSnapshot) -> Dict[int, int]:
        """新しいスナップショットで更新"""
        
        if not snapshot.vortices:
            self.active_ids = {}
            return {}
        
        # 現在の渦の中心
        current_centers = np.array([v.center for v in snapshot.vortices])
        
        # 前ステップの渦とマッチング
        new_active_ids = {}
        matched = set()
        
        if self.active_ids:
            # 前ステップの渦の予測位置（単純に右に移動と仮定）
            prev_centers = []
            prev_ids = []
            for cluster_id, vortex_id in self.active_ids.items():
                if vortex_id in self.tracks:
                    last_center = self.tracks[vortex_id][-1][1]
                    # 簡単な予測：少し右に移動
                    predicted = last_center + np.array([2.0, 0])
                    prev_centers.append(predicted)
                    prev_ids.append(vortex_id)
            
            if prev_centers:
                prev_centers = np.array(prev_centers)
                
                # 距離行列
                distances = cdist(current_centers, prev_centers)
                
                # 貪欲マッチング
                for i in range(len(current_centers)):
                    if distances.shape[1] > 0:
                        min_idx = np.argmin(distances[i])
                        min_dist = distances[i, min_idx]
                        
                        if min_dist < self.matching_threshold:
                            vortex_id = prev_ids[min_idx]
                            new_active_ids[i] = vortex_id
                            matched.add(i)
                            
                            # この渦の履歴更新
                            self.tracks[vortex_id].append((
                                snapshot.step,
                                snapshot.vortices[i].center,
                                snapshot.vortices[i].circulation
                            ))
                            
                            # この組み合わせを除外
                            distances[:, min_idx] = np.inf
        
        # マッチしなかった渦は新規
        for i, vortex in enumerate(snapshot.vortices):
            if i not in matched:
                vortex_id = self.next_id
                self.next_id += 1
                new_active_ids[i] = vortex_id
                
                # 新しい軌跡開始
                self.tracks[vortex_id] = [(
                    snapshot.step,
                    vortex.center,
                    vortex.circulation
                )]
        
        self.active_ids = new_active_ids
        return new_active_ids

# ==============================
# 解析関数
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

def compute_strouhal_number(
    tracks: Dict,
    obstacle_size: float,
    inlet_velocity: float,
    dt: float,
    min_circulation: float = 0.5,
    min_track_length: int = 5
) -> float:
    """
    Strouhal数を計算（修正版）
    - 強い渦のみを対象
    - 上下交互の剥離を考慮
    """
    
    # 強い渦の剥離時刻を収集
    shedding_events = []  # (step, y_position, circulation)
    
    for vortex_id, track in tracks.items():
        if len(track) < min_track_length:
            continue
            
        # 最大循環をチェック
        max_circulation = max(abs(t[2]) for t in track)
        if max_circulation < min_circulation:
            continue
        
        # 初期位置で上下判定
        birth_step = track[0][0]
        birth_y = track[0][1][1]
        birth_circulation = track[0][2]
        
        # 強い渦の剥離イベントとして記録
        shedding_events.append((birth_step, birth_y, birth_circulation))
    
    if len(shedding_events) < 4:  # 最低4つは必要
        return 0.0
    
    # 時間順にソート
    shedding_events.sort(key=lambda x: x[0])
    
    # 方法1: 全体の剥離頻度（上下合わせて）
    all_steps = [e[0] for e in shedding_events]
    all_intervals = np.diff(all_steps)
    
    if len(all_intervals) > 0:
        # カルマン渦列は上下交互なので、同じ側の渦の間隔は2倍
        mean_interval = np.mean(all_intervals) * dt  # 全渦の平均間隔
        frequency = 1.0 / (mean_interval * 2.0)  # 片側の周波数
    else:
        return 0.0
    
    # 方法2: 上側のみの周期（検証用）
    upper_events = [e for e in shedding_events if e[1] > 75]
    if len(upper_events) >= 2:
        upper_steps = [e[0] for e in upper_events]
        upper_intervals = np.diff(upper_steps)
        if len(upper_intervals) > 0:
            upper_period = np.mean(upper_intervals) * dt
            upper_frequency = 1.0 / upper_period
            
            # デバッグ情報
            print(f"  Debug: Upper frequency = {upper_frequency:.3f} Hz")
            print(f"  Debug: All vortex frequency = {1.0/mean_interval:.3f} Hz")
    
    # Strouhal数
    D = 2 * obstacle_size
    St = frequency * D / inlet_velocity
    
    return St

def compute_strouhal_number_filtered(
    tracks: Dict,
    obstacle_size: float,
    inlet_velocity: float,
    dt: float,
    min_circulation: float = 1.0,  # より厳しい閾値
    min_track_length: int = 10,
    x_range: Tuple[float, float] = (80, 200)  # 障害物近傍のみ
) -> float:
    """
    より厳密なフィルタリングでStrouhal数を計算
    主要な渦（カルマン渦）のみを対象
    """
    
    # カルマン渦候補を抽出
    karman_vortices = []
    
    for vortex_id, track in tracks.items():
        if len(track) < min_track_length:
            continue
        
        # 軌跡の統計
        max_circulation = max(abs(t[2]) for t in track)
        mean_x = np.mean([t[1][0] for t in track])
        
        # カルマン渦の条件
        if (max_circulation >= min_circulation and
            x_range[0] <= mean_x <= x_range[1]):
            
            birth_step = track[0][0]
            birth_y = track[0][1][1]
            
            # 上下どちらか記録
            side = 'upper' if birth_y > 75 else 'lower'
            karman_vortices.append({
                'step': birth_step,
                'side': side,
                'circulation': max_circulation
            })
    
    if len(karman_vortices) < 4:
        return 0.0
    
    # 上下別に分離
    upper_steps = [v['step'] for v in karman_vortices if v['side'] == 'upper']
    lower_steps = [v['step'] for v in karman_vortices if v['side'] == 'lower']
    
    # より多い方を使用
    if len(upper_steps) >= len(lower_steps) and len(upper_steps) >= 2:
        intervals = np.diff(sorted(upper_steps))
    elif len(lower_steps) >= 2:
        intervals = np.diff(sorted(lower_steps))
    else:
        return 0.0
    
    if len(intervals) > 0:
        mean_interval = np.mean(intervals) * dt
        frequency = 1.0 / mean_interval
        
        D = 2 * obstacle_size
        St = frequency * D / inlet_velocity
        
        return St
    
    return 0.0

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
    states,  # シミュレーション状態のリスト
    config,  # GETWindConfig
    save_file: str = 'vortex_snapshots.npz'
):
    """シミュレーションデータから渦を抽出"""
    
    print("=" * 70)
    print("GET Wind™ Vortex Tracking - DBSCAN Edition")
    print("Processing snapshots...")
    print("=" * 70)
    
    snapshots = []
    tracker = VortexTracker()
    
    for i, state in enumerate(states):
        if i % 100 == 0:
            print(f"Processing step {i}...")
        
        # 渦検出
        vortices = detect_vortices_dbscan(
            state.position,
            state.Lambda_F,
            state.Q_criterion,
            state.is_active,
            eps=15.0,
            min_samples=3,
            Q_threshold=0.1
        )
        
        # スナップショット作成
        snapshot = VortexSnapshot(
            step=i,
            vortices=vortices,
            total_particles=np.sum(state.is_active)
        )
        snapshots.append(snapshot)
        
        # 軌跡更新
        tracker.update(snapshot)
    
    # Strouhal数計算（両方の方法で）
    St_basic = compute_strouhal_number(
        tracker.tracks,
        config.obstacle_size,
        config.Lambda_F_inlet,
        config.dt
    )
    
    St_filtered = compute_strouhal_number_filtered(
        tracker.tracks,
        config.obstacle_size,
        config.Lambda_F_inlet,
        config.dt
    )
    
    print(f"\n✨ Analysis Complete!")
    print(f"Total snapshots: {len(snapshots)}")
    print(f"Total vortices tracked: {tracker.next_id}")
    print(f"Strouhal number (basic): {St_basic:.4f}")
    print(f"Strouhal number (filtered): {St_filtered:.4f}")
    
    # 保存
    np.savez(save_file,
        n_steps=len(snapshots),
        steps=[s.step for s in snapshots],
        n_vortices=[len(s.vortices) for s in snapshots],
        strouhal_number=St_basic,
        strouhal_number_filtered=St_filtered,
        tracks=tracker.tracks
    )
    
    print(f"Results saved to {save_file}")
    
    return snapshots, tracker

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
