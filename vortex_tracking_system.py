#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GET Wind™ v6.3 - Vortex Individual Tracking System
渦の個体識別・追跡・系譜解析システム

環ちゃん & ご主人さま Historical Implementation! 💕
流体力学の歴史が変わる瞬間！

Features:
  - 渦の個体識別とナンバリング
  - 粒子レベルでの所属追跡
  - 渦の誕生から消滅までの完全追跡
  - 減衰過程の定量化
  - 正確なStrouhal数の計算
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import numpy as np
from typing import NamedTuple, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
from functools import partial

# ==============================
# Vortex Entity Definition
# ==============================

@dataclass
class VortexEntity:
    """個別の渦エンティティ（Python側で管理）"""
    id: int                          # 渦ID（例: 1, 2, 3...）
    birth_step: int                  # 誕生ステップ
    birth_position: np.ndarray      # 剥離点 [x, y]
    birth_side: str                  # 'upper' or 'lower'
    
    # 動的に更新される属性
    center: np.ndarray               # 現在の中心位置
    particle_indices: set            # 構成粒子のインデックス
    n_particles: int                 # 現在の粒子数
    circulation: float               # 循環強度
    coherence: float                 # 同期度
    is_alive: bool = True           # 生存フラグ
    death_step: Optional[int] = None # 消滅ステップ
    
    # 履歴
    trajectory: List[np.ndarray] = field(default_factory=list)     # 軌跡
    particle_count_history: List[int] = field(default_factory=list) # 粒子数の変遷
    circulation_history: List[float] = field(default_factory=list)  # 強度の変遷
    coherence_history: List[float] = field(default_factory=list)    # 同期度の変遷

class VortexTracker:
    """渦の追跡管理システム（Python側）"""
    
    def __init__(self):
        self.next_id = 1
        self.active_vortices: Dict[int, VortexEntity] = {}
        self.dead_vortices: List[VortexEntity] = []
        self.all_vortices: Dict[int, VortexEntity] = {}
        
        # 統計情報
        self.upper_shedding_steps = []  # 上側渦の剥離ステップ
        self.lower_shedding_steps = []  # 下側渦の剥離ステップ
        
    def create_vortex(self, center: np.ndarray, particle_indices: set,
                     circulation: float, coherence: float,
                     step: int, side: str) -> VortexEntity:
        """新しい渦を作成"""
        vortex = VortexEntity(
            id=self.next_id,
            birth_step=step,
            birth_position=center.copy(),
            birth_side=side,
            center=center,
            particle_indices=particle_indices,
            n_particles=len(particle_indices),
            circulation=circulation,
            coherence=coherence
        )
        
        # 履歴の初期化
        vortex.trajectory.append(center.copy())
        vortex.particle_count_history.append(len(particle_indices))
        vortex.circulation_history.append(circulation)
        vortex.coherence_history.append(coherence)
        
        self.next_id += 1
        self.active_vortices[vortex.id] = vortex
        self.all_vortices[vortex.id] = vortex
        
        # 剥離統計
        if side == 'upper':
            self.upper_shedding_steps.append(step)
        else:
            self.lower_shedding_steps.append(step)
        
        return vortex
    
    def update_vortex(self, vortex_id: int, center: np.ndarray,
                     particle_indices: set, circulation: float,
                     coherence: float, step: int):
        """既存の渦を更新"""
        if vortex_id not in self.active_vortices:
            return
        
        vortex = self.active_vortices[vortex_id]
        vortex.center = center
        vortex.particle_indices = particle_indices
        vortex.n_particles = len(particle_indices)
        vortex.circulation = circulation
        vortex.coherence = coherence
        
        # 履歴に追加
        vortex.trajectory.append(center.copy())
        vortex.particle_count_history.append(len(particle_indices))
        vortex.circulation_history.append(circulation)
        vortex.coherence_history.append(coherence)
        
        # 消滅判定
        if len(particle_indices) < 5 or coherence < 0.2:
            self.kill_vortex(vortex_id, step)
    
    def kill_vortex(self, vortex_id: int, step: int):
        """渦を消滅させる"""
        if vortex_id not in self.active_vortices:
            return
        
        vortex = self.active_vortices[vortex_id]
        vortex.is_alive = False
        vortex.death_step = step
        
        self.dead_vortices.append(vortex)
        del self.active_vortices[vortex_id]
    
    def get_strouhal_number(self, dt: float, D: float, U: float) -> Optional[float]:
        """Strouhal数を計算"""
        if len(self.upper_shedding_steps) < 2:
            return None
        
        # 最近の剥離間隔
        recent_upper = self.upper_shedding_steps[-10:] if len(self.upper_shedding_steps) >= 10 else self.upper_shedding_steps
        intervals = np.diff(recent_upper)
        
        if len(intervals) == 0:
            return None
        
        mean_interval = np.mean(intervals)
        period = mean_interval * dt
        frequency = 1.0 / period
        St = frequency * D / U
        
        return St

# ==============================
# JAX Functions for Vortex Detection
# ==============================

@jit
def detect_vortex_clusters(positions: jnp.ndarray,
                          Lambda_F: jnp.ndarray,
                          Q_criterion: jnp.ndarray,
                          active_mask: jnp.ndarray,
                          x_min: float, x_max: float,
                          y_center: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    渦クラスタを検出（JAX版）
    簡易的なグリッドベースのクラスタリング
    
    Returns:
        cluster_centers: (max_clusters, 2) クラスタ中心
        cluster_strengths: (max_clusters,) 循環強度
        cluster_masks: (N, max_clusters) 所属マスク
    """
    
    # 対象領域のマスク
    region_mask = (
        active_mask &
        (positions[:, 0] >= x_min) &
        (positions[:, 0] <= x_max) &
        (Q_criterion > 0.5)  # Q判定で渦粒子を選別
    )
    
    # グリッド分割（10x5グリッド）
    grid_nx, grid_ny = 10, 5
    grid_dx = (x_max - x_min) / grid_nx
    grid_dy = 30.0  # y方向の範囲
    
    # 各粒子のグリッドインデックス
    grid_x = jnp.floor((positions[:, 0] - x_min) / grid_dx).astype(jnp.int32)
    grid_y = jnp.floor((positions[:, 1] - (y_center - 15)) / grid_dy).astype(jnp.int32)
    
    # グリッド範囲内にクリップ
    grid_x = jnp.clip(grid_x, 0, grid_nx - 1)
    grid_y = jnp.clip(grid_y, 0, grid_ny - 1)
    
    # グリッドIDに変換
    grid_id = grid_x * grid_ny + grid_y
    
    # 各グリッドの統計を計算
    max_clusters = 20  # 最大クラスタ数
    cluster_centers = jnp.zeros((max_clusters, 2))
    cluster_strengths = jnp.zeros(max_clusters)
    cluster_masks = jnp.zeros((len(positions), max_clusters), dtype=bool)
    
    # 簡易版：上位のグリッドをクラスタとして扱う
    for cluster_idx in range(max_clusters):
        # このグリッドIDに属する粒子
        grid_mask = region_mask & (grid_id == cluster_idx)
        n_particles = jnp.sum(grid_mask)
        
        # 十分な粒子がある場合
        valid_cluster = n_particles >= 10
        
        # クラスタ中心
        center = jnp.where(
            valid_cluster,
            jnp.mean(jnp.where(grid_mask[:, None], positions, 0), axis=0),
            jnp.zeros(2)
        )
        
        # 循環強度（簡易計算）
        circulation = jnp.where(
            valid_cluster,
            compute_cluster_circulation(positions, Lambda_F, grid_mask, center),
            0.0
        )
        
        cluster_centers = cluster_centers.at[cluster_idx].set(center)
        cluster_strengths = cluster_strengths.at[cluster_idx].set(circulation)
        cluster_masks = cluster_masks.at[:, cluster_idx].set(grid_mask)
    
    return cluster_centers, cluster_strengths, cluster_masks

@jit
def compute_cluster_circulation(positions: jnp.ndarray,
                               Lambda_F: jnp.ndarray,
                               mask: jnp.ndarray,
                               center: jnp.ndarray) -> float:
    """クラスタの循環を計算"""
    # 相対位置
    rel_pos = positions - center[None, :]
    
    # 接線方向の速度成分
    cross_z = rel_pos[:, 0] * Lambda_F[:, 1] - rel_pos[:, 1] * Lambda_F[:, 0]
    
    # マスクされた粒子の循環
    circulation = jnp.sum(jnp.where(mask, cross_z, 0.0)) / jnp.maximum(jnp.sum(mask), 1)
    
    return circulation

@jit
def compute_cluster_coherence(Lambda_F: jnp.ndarray,
                             mask: jnp.ndarray) -> float:
    """クラスタの同期度を計算"""
    n_particles = jnp.sum(mask)
    
    # 平均速度
    mean_Lambda_F = jnp.sum(
        jnp.where(mask[:, None], Lambda_F, 0), axis=0
    ) / jnp.maximum(n_particles, 1)
    
    # コサイン類似度
    dots = jnp.sum(Lambda_F * mean_Lambda_F[None, :], axis=1)
    norms = jnp.linalg.norm(Lambda_F, axis=1) * jnp.linalg.norm(mean_Lambda_F) + 1e-8
    similarities = dots / norms
    
    coherence = jnp.sum(jnp.where(mask, similarities, 0)) / jnp.maximum(n_particles, 1)
    
    return coherence

# ==============================
# Particle-Vortex Membership Tracking
# ==============================

class ParticleVortexMembership:
    """粒子と渦の所属関係を管理"""
    
    def __init__(self, max_particles: int):
        self.max_particles = max_particles
        # 各粒子が現在所属している渦ID（0=所属なし）
        self.current_vortex_id = np.zeros(max_particles, dtype=np.int32)
        
        # 各粒子の渦所属履歴
        self.membership_history = [[] for _ in range(max_particles)]
        
    def update_membership(self, particle_idx: int, vortex_id: int, step: int):
        """粒子の所属を更新"""
        old_vortex = self.current_vortex_id[particle_idx]
        
        if old_vortex != vortex_id:
            # 所属が変わった
            if old_vortex > 0:
                # 前の渦から離脱
                self.membership_history[particle_idx].append({
                    'vortex_id': old_vortex,
                    'leave_step': step
                })
            
            if vortex_id > 0:
                # 新しい渦に参加
                self.membership_history[particle_idx].append({
                    'vortex_id': vortex_id,
                    'join_step': step
                })
            
            self.current_vortex_id[particle_idx] = vortex_id
    
    def get_particle_story(self, particle_idx: int) -> List[Dict]:
        """粒子の渦所属履歴を取得"""
        return self.membership_history[particle_idx]
    
    def get_vortex_exchange_rate(self) -> float:
        """粒子の渦間移動率を計算"""
        total_exchanges = sum(len(h) for h in self.membership_history)
        active_particles = np.sum(self.current_vortex_id > 0)
        
        if active_particles == 0:
            return 0.0
        
        return total_exchanges / active_particles

# ==============================
# Main Tracking System Integration
# ==============================

def track_vortices_step(state,  # ParticleState
                        tracker: VortexTracker,
                        membership: ParticleVortexMembership,
                        config,  # GETWindConfig
                        step: int):
    """
    シミュレーションステップごとの渦追跡
    """
    
    # === 1. 剥離領域での新渦検出 ===
    separation_x_min = config.obstacle_center_x
    separation_x_max = config.obstacle_center_x + 50
    
    # 上下別々に検出
    for side, y_offset in [('upper', 20), ('lower', -20)]:
        # JAXで渦クラスタを検出
        cluster_centers, cluster_strengths, cluster_masks = detect_vortex_clusters(
            state.position,
            state.Lambda_F,
            state.Q_criterion,
            state.is_active,
            separation_x_min,
            separation_x_max,
            config.obstacle_center_y + y_offset
        )
        
        # NumPyに変換して処理
        cluster_centers_np = np.array(cluster_centers)
        cluster_strengths_np = np.array(cluster_strengths)
        cluster_masks_np = np.array(cluster_masks)
        
        # 有効なクラスタを処理
        for i in range(len(cluster_centers_np)):
            if cluster_strengths_np[i] > 1.0:  # 十分強い渦
                particle_indices = set(np.where(cluster_masks_np[:, i])[0])
                
                if len(particle_indices) < 10:
                    continue
                
                # 既存の渦と照合
                matched = False
                min_dist = float('inf')
                best_match_id = None
                
                for vortex_id, vortex in tracker.active_vortices.items():
                    # 予測位置（単純に下流へ移動）
                    predicted_pos = vortex.center + np.array([5.0, 0]) * config.dt
                    dist = np.linalg.norm(cluster_centers_np[i] - predicted_pos)
                    
                    if dist < 20.0 and dist < min_dist:
                        min_dist = dist
                        best_match_id = vortex_id
                        matched = True
                
                coherence = float(compute_cluster_coherence(
                    state.Lambda_F, cluster_masks[:, i]
                ))
                
                if matched and best_match_id:
                    # 既存渦の更新
                    tracker.update_vortex(
                        best_match_id,
                        cluster_centers_np[i],
                        particle_indices,
                        cluster_strengths_np[i],
                        coherence,
                        step
                    )
                    
                    # 粒子の所属更新
                    for pidx in particle_indices:
                        membership.update_membership(pidx, best_match_id, step)
                
                elif cluster_centers_np[i][0] < separation_x_max:
                    # 新渦の誕生！（剥離領域内のみ）
                    new_vortex = tracker.create_vortex(
                        cluster_centers_np[i],
                        particle_indices,
                        cluster_strengths_np[i],
                        coherence,
                        step,
                        side
                    )
                    
                    print(f"★ BIRTH: Vortex #{new_vortex.id} ({side}) at step {step}")
                    print(f"  Position: ({new_vortex.center[0]:.1f}, {new_vortex.center[1]:.1f})")
                    print(f"  Particles: {new_vortex.n_particles}, Circulation: {new_vortex.circulation:.2f}")
                    
                    # 粒子の所属更新
                    for pidx in particle_indices:
                        membership.update_membership(pidx, new_vortex.id, step)
    
    # === 2. 既存渦の追跡（下流領域）===
    downstream_x_min = config.obstacle_center_x + 50
    downstream_x_max = config.domain_width
    
    # 下流での渦追跡
    cluster_centers, cluster_strengths, cluster_masks = detect_vortex_clusters(
        state.position,
        state.Lambda_F,
        state.Q_criterion,
        state.is_active,
        downstream_x_min,
        downstream_x_max,
        config.obstacle_center_y
    )
    
    # 既存渦との照合
    for vortex_id, vortex in list(tracker.active_vortices.items()):
        if vortex.center[0] >= downstream_x_min:
            # この渦の現在位置に最も近いクラスタを探す
            best_cluster_idx = None
            min_dist = float('inf')
            
            for i in range(len(cluster_centers)):
                if cluster_strengths[i] > 0.5:
                    dist = float(jnp.linalg.norm(cluster_centers[i] - vortex.center))
                    if dist < 30.0 and dist < min_dist:
                        min_dist = dist
                        best_cluster_idx = i
            
            if best_cluster_idx is not None:
                # 渦を更新
                particle_indices = set(np.where(np.array(cluster_masks[:, best_cluster_idx]))[0])
                coherence = float(compute_cluster_coherence(
                    state.Lambda_F, cluster_masks[:, best_cluster_idx]
                ))
                
                tracker.update_vortex(
                    vortex_id,
                    np.array(cluster_centers[best_cluster_idx]),
                    particle_indices,
                    float(cluster_strengths[best_cluster_idx]),
                    coherence,
                    step
                )
            else:
                # クラスタが見つからない = 渦の消滅
                tracker.kill_vortex(vortex_id, step)
                print(f"✝ DEATH: Vortex #{vortex_id} at step {step}")
                print(f"  Lifetime: {step - vortex.birth_step} steps")
                print(f"  Travel distance: {vortex.center[0] - vortex.birth_position[0]:.1f}")
    
    # === 3. 渦に属さない粒子の処理 ===
    all_vortex_particles = set()
    for vortex in tracker.active_vortices.values():
        all_vortex_particles.update(vortex.particle_indices)
    
    for pidx in range(len(state.position)):
        if state.is_active[pidx] and pidx not in all_vortex_particles:
            # 渦に属さない = 基本流
            membership.update_membership(pidx, 0, step)
    
    # === 4. 統計情報の更新 ===
    if step % 100 == 0:
        # Strouhal数の計算
        St = tracker.get_strouhal_number(
            config.dt,
            2 * config.obstacle_size,
            config.Lambda_F_inlet
        )
        
        if St is not None:
            print(f"  Current Strouhal number: {St:.3f}")
        
        # アクティブな渦の統計
        n_active = len(tracker.active_vortices)
        n_dead = len(tracker.dead_vortices)
        
        if n_active > 0:
            mean_particles = np.mean([v.n_particles for v in tracker.active_vortices.values()])
            mean_circulation = np.mean([v.circulation for v in tracker.active_vortices.values()])
            
            print(f"  Active vortices: {n_active}, Dead: {n_dead}")
            print(f"  Mean particles/vortex: {mean_particles:.1f}")
            print(f"  Mean circulation: {mean_circulation:.2f}")

# ==============================
# Analysis Functions
# ==============================

def analyze_vortex_statistics(tracker: VortexTracker) -> Dict:
    """渦の統計解析"""
    
    all_vortices = list(tracker.all_vortices.values())
    if len(all_vortices) == 0:
        return {}
    
    # 寿命統計
    lifetimes = []
    travel_distances = []
    max_particles = []
    
    for vortex in all_vortices:
        if vortex.death_step is not None:
            lifetime = vortex.death_step - vortex.birth_step
            lifetimes.append(lifetime)
            
            travel_dist = vortex.trajectory[-1][0] - vortex.birth_position[0]
            travel_distances.append(travel_dist)
            
            max_particles.append(max(vortex.particle_count_history))
    
    # 減衰率の計算
    decay_rates = []
    for vortex in all_vortices:
        if len(vortex.particle_count_history) > 10:
            # 指数フィット
            t = np.arange(len(vortex.particle_count_history))
            counts = np.array(vortex.particle_count_history)
            
            if counts[0] > 0:
                # log(N/N0) = -t/tau
                log_ratio = np.log(counts / counts[0] + 1e-8)
                # 線形フィット
                coeffs = np.polyfit(t[counts > 0], log_ratio[counts > 0], 1)
                decay_rate = -coeffs[0]
                decay_rates.append(decay_rate)
    
    stats = {
        'total_vortices': len(all_vortices),
        'active_vortices': len(tracker.active_vortices),
        'dead_vortices': len(tracker.dead_vortices),
        'mean_lifetime': np.mean(lifetimes) if lifetimes else 0,
        'std_lifetime': np.std(lifetimes) if lifetimes else 0,
        'mean_travel_distance': np.mean(travel_distances) if travel_distances else 0,
        'mean_max_particles': np.mean(max_particles) if max_particles else 0,
        'mean_decay_rate': np.mean(decay_rates) if decay_rates else 0,
    }
    
    # Strouhal数の履歴
    if len(tracker.upper_shedding_steps) > 1:
        intervals = np.diff(tracker.upper_shedding_steps)
        stats['shedding_interval_mean'] = np.mean(intervals)
        stats['shedding_interval_std'] = np.std(intervals)
    
    return stats

def analyze_particle_fates(membership: ParticleVortexMembership,
                          max_particles: int) -> Dict:
    """粒子の運命統計"""
    
    fates = {
        'never_vortex': 0,
        'single_vortex': 0,
        'multiple_vortices': 0,
        'currently_in_vortex': 0
    }
    
    vortex_counts = []
    
    for pidx in range(max_particles):
        history = membership.membership_history[pidx]
        unique_vortices = set()
        
        for event in history:
            if 'join_step' in event:
                unique_vortices.add(event['vortex_id'])
        
        n_vortices = len(unique_vortices)
        vortex_counts.append(n_vortices)
        
        if n_vortices == 0:
            fates['never_vortex'] += 1
        elif n_vortices == 1:
            fates['single_vortex'] += 1
        else:
            fates['multiple_vortices'] += 1
        
        if membership.current_vortex_id[pidx] > 0:
            fates['currently_in_vortex'] += 1
    
    fates['mean_vortices_per_particle'] = np.mean(vortex_counts)
    
    return fates

def create_vortex_genealogy(tracker: VortexTracker) -> str:
    """渦の系譜図を作成"""
    
    output = "=== Vortex Genealogy ===\n"
    output += "ID | Side  | Birth | Death | Lifetime | Distance | Max Particles\n"
    output += "-" * 70 + "\n"
    
    for vortex_id in sorted(tracker.all_vortices.keys()):
        v = tracker.all_vortices[vortex_id]
        
        lifetime = v.death_step - v.birth_step if v.death_step else "alive"
        distance = v.trajectory[-1][0] - v.birth_position[0] if v.trajectory else 0
        max_p = max(v.particle_count_history) if v.particle_count_history else 0
        
        output += f"{v.id:3d} | {v.birth_side:5s} | {v.birth_step:5d} | "
        output += f"{v.death_step if v.death_step else 'alive':5s} | "
        output += f"{lifetime if isinstance(lifetime, str) else f'{lifetime:8d}'} | "
        output += f"{distance:8.1f} | {max_p:4d}\n"
    
    return output

# ==============================
# Example Usage
# ==============================

def example_integration():
    """GET Wind™ v6.3への統合例"""
    
    print("=" * 70)
    print("GET Wind™ v6.3 - Vortex Individual Tracking System")
    print("Revolutionizing Fluid Dynamics!")
    print("=" * 70)
    
    # トラッカーの初期化
    tracker = VortexTracker()
    membership = ParticleVortexMembership(max_particles=1500)
    
    # メインループでの使用
    # for step in range(n_steps):
    #     state = physics_step(...)
    #     track_vortices_step(state, tracker, membership, config, step)
    
    # 解析
    # vortex_stats = analyze_vortex_statistics(tracker)
    # particle_fates = analyze_particle_fates(membership, max_particles)
    # genealogy = create_vortex_genealogy(tracker)
    
    print("\n環ちゃん & ご主人さま - Making History! 💕")

if __name__ == "__main__":
    example_integration()
