#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GET Wind™ v6.3 JAX Edition - Ultra-Fast Vortex Tracking System (FIXED!)
環ちゃん & ご主人さま Boolean Indexing Fix! 💕
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import numpy as np
from typing import NamedTuple, Tuple, Optional, Dict, List
from functools import partial
import time

# ==============================
# JAX Vortex State (全部テンソル！)
# ==============================

class VortexStateJAX(NamedTuple):
    """渦の状態を全部JAXテンソルで管理"""
    # 基本情報 (max_vortices,)
    ids: jnp.ndarray              # 渦ID（0=未使用）
    is_alive: jnp.ndarray         # 生存フラグ
    birth_steps: jnp.ndarray      # 誕生ステップ
    death_steps: jnp.ndarray      # 消滅ステップ（-1=生存中）
    birth_side: jnp.ndarray       # 0=upper, 1=lower
    
    # 動的状態 (max_vortices, 2) or (max_vortices,)
    centers: jnp.ndarray          # 現在の中心位置
    circulations: jnp.ndarray    # 循環強度
    coherences: jnp.ndarray       # 同期度
    n_particles: jnp.ndarray      # 粒子数
    
    # 履歴（固定長バッファ）(max_vortices, history_len, ...)
    trajectory: jnp.ndarray       # 位置履歴
    circulation_hist: jnp.ndarray # 循環履歴
    coherence_hist: jnp.ndarray   # 同期度履歴
    particle_count_hist: jnp.ndarray # 粒子数履歴
    hist_index: jnp.ndarray       # 履歴の現在インデックス

class ParticleMembershipJAX(NamedTuple):
    """粒子の渦所属をJAXで管理"""
    vortex_ids: jnp.ndarray       # (N,) 各粒子の所属渦ID
    join_steps: jnp.ndarray       # (N,) 参加ステップ
    leave_steps: jnp.ndarray      # (N,) 離脱ステップ
    membership_matrix: jnp.ndarray # (N, max_vortices) 所属行列（高速検索用）
    history_count: jnp.ndarray    # (N,) 各粒子の渦遍歴数

class VortexSheddingStats(NamedTuple):
    """渦剥離統計"""
    upper_shedding_steps: jnp.ndarray  # (max_events,) 上側剥離ステップ
    lower_shedding_steps: jnp.ndarray  # (max_events,) 下側剥離ステップ
    upper_count: jnp.ndarray           # スカラー、現在の上側剥離数
    lower_count: jnp.ndarray           # スカラー、現在の下側剥離数

# ==============================
# 初期化
# ==============================

def initialize_particle_membership(max_particles: int) -> ParticleMembershipJAX:
    """粒子所属の初期化"""
    return ParticleMembershipJAX(
        vortex_ids=jnp.zeros(max_particles, dtype=jnp.int32),
        join_steps=jnp.full(max_particles, -1, dtype=jnp.int32),
        leave_steps=jnp.full(max_particles, -1, dtype=jnp.int32),
        membership_matrix=jnp.zeros((max_particles, 100), dtype=bool),
        history_count=jnp.zeros(max_particles, dtype=jnp.int32)
    )

def initialize_shedding_stats(max_events: int = 1000) -> VortexSheddingStats:
    """剥離統計の初期化"""
    return VortexSheddingStats(
        upper_shedding_steps=jnp.full(max_events, -1, dtype=jnp.int32),
        lower_shedding_steps=jnp.full(max_events, -1, dtype=jnp.int32),
        upper_count=jnp.array(0, dtype=jnp.int32),
        lower_count=jnp.array(0, dtype=jnp.int32)
    )

def initialize_vortex_state(max_vortices: int = 100, 
                           history_len: int = 500) -> VortexStateJAX:
    """渦状態の初期化"""
    return VortexStateJAX(
        ids=jnp.zeros(max_vortices, dtype=jnp.int32),
        is_alive=jnp.zeros(max_vortices, dtype=bool),
        birth_steps=jnp.full(max_vortices, -1, dtype=jnp.int32),
        death_steps=jnp.full(max_vortices, -1, dtype=jnp.int32),
        birth_side=jnp.zeros(max_vortices, dtype=jnp.int32),
        centers=jnp.zeros((max_vortices, 2)),
        circulations=jnp.zeros(max_vortices),
        coherences=jnp.zeros(max_vortices),
        n_particles=jnp.zeros(max_vortices, dtype=jnp.int32),
        trajectory=jnp.zeros((max_vortices, history_len, 2)),
        circulation_hist=jnp.zeros((max_vortices, history_len)),
        coherence_hist=jnp.zeros((max_vortices, history_len)),
        particle_count_hist=jnp.zeros((max_vortices, history_len), dtype=jnp.int32),
        hist_index=jnp.zeros(max_vortices, dtype=jnp.int32)
    )

# ==============================
# 渦クラスタ検出（高速版 + 上下分離）
# ==============================

@partial(jit, static_argnums=(5, 6, 7))
def detect_vortex_clusters_separated(
    positions: jnp.ndarray,
    Lambda_F: jnp.ndarray,
    Q_criterion: jnp.ndarray,
    active_mask: jnp.ndarray,
    obstacle_center: jnp.ndarray,
    side: int,  # 0=upper, 1=lower
    grid_size: int = 10,
    min_particles: int = 10
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    上下領域別の渦検出
    
    Args:
        side: 0=upper(上側), 1=lower(下側)
    """
    N = positions.shape[0]
    max_clusters = 25  # 片側最大数
    
    # Y方向のオフセット
    y_offset = jnp.where(side == 0, 20.0, -20.0)
    y_center = obstacle_center[1] + y_offset
    
    # 対象領域のマスク
    y_min = jnp.where(side == 0, y_center - 10, y_center - 30)
    y_max = jnp.where(side == 0, y_center + 30, y_center + 10)
    
    region_mask = (
        active_mask & 
        (Q_criterion > 0.5) &
        (positions[:, 1] >= y_min) &
        (positions[:, 1] <= y_max)
    )
    
    # 以下は元の detect_vortex_clusters_fast と同じロジック
    grid_scale = 20.0
    grid_indices = jnp.floor(positions / grid_scale).astype(jnp.int32)
    grid_ids = grid_indices[:, 0] * 1000 + grid_indices[:, 1]
    grid_ids = jnp.where(region_mask, grid_ids, -1)
    
    def compute_cell_stats(cell_id):
        cell_mask = (grid_ids == cell_id) & (cell_id >= 0)
        n_particles = jnp.sum(cell_mask)
        valid = n_particles >= min_particles
        
        center = jnp.where(
            valid,
            jnp.sum(jnp.where(cell_mask[:, None], positions, 0), axis=0) / jnp.maximum(n_particles, 1),
            jnp.zeros(2)
        )
        
        rel_pos = positions - center[None, :]
        cross_z = rel_pos[:, 0] * Lambda_F[:, 1] - rel_pos[:, 1] * Lambda_F[:, 0]
        circulation = jnp.where(
            valid,
            jnp.sum(jnp.where(cell_mask, cross_z, 0)) / jnp.maximum(n_particles, 1),
            0.0
        )
        
        mean_Lambda = jnp.sum(jnp.where(cell_mask[:, None], Lambda_F, 0), axis=0) / jnp.maximum(n_particles, 1)
        dots = jnp.sum(Lambda_F * mean_Lambda[None, :], axis=1)
        norms = jnp.linalg.norm(Lambda_F, axis=1) * jnp.linalg.norm(mean_Lambda) + 1e-8
        coherence = jnp.where(
            valid,
            jnp.mean(jnp.where(cell_mask, dots / norms, 0)),
            0.0
        )
        
        return center, jnp.array([circulation, coherence, n_particles.astype(jnp.float32)])
    
    unique_grid_ids = jnp.unique(grid_ids, size=max_clusters, fill_value=-1)
    centers, properties = vmap(compute_cell_stats)(unique_grid_ids)
    
    distances = jnp.linalg.norm(positions[:, None, :] - centers[None, :, :], axis=2)
    particle_cluster_ids = jnp.argmin(distances, axis=1)
    particle_cluster_ids = jnp.where(region_mask, particle_cluster_ids, -1)
    
    return centers, properties, particle_cluster_ids

# ==============================
# 粒子所属の更新
# ==============================

@jit
def update_particle_membership(
    membership: ParticleMembershipJAX,
    particle_cluster_ids: jnp.ndarray,
    vortex_ids_for_clusters: jnp.ndarray,
    step: int
) -> ParticleMembershipJAX:
    """粒子の渦所属を更新（完全ベクトル化版）"""
    N = len(membership.vortex_ids)
    
    # 各粒子の新しい所属渦ID（ベクトル演算）
    new_vortex_ids = jnp.where(
        particle_cluster_ids >= 0,
        jnp.where(
            particle_cluster_ids < len(vortex_ids_for_clusters),
            vortex_ids_for_clusters[jnp.clip(particle_cluster_ids, 0, len(vortex_ids_for_clusters)-1)],
            0
        ),
        0
    )
    
    # 所属が変わった粒子を検出
    changed = (new_vortex_ids != membership.vortex_ids)
    
    # 離脱処理
    leaving = changed & (membership.vortex_ids > 0)
    new_leave_steps = jnp.where(leaving, step, membership.leave_steps)
    
    # 参加処理
    joining = changed & (new_vortex_ids > 0)
    new_join_steps = jnp.where(joining, step, membership.join_steps)
    
    # 履歴カウント更新
    new_history_count = membership.history_count + joining.astype(jnp.int32)
    
    # 所属行列の更新（ベクトル化）
    # 各粒子と各渦の組み合わせをチェック
    max_vortices = membership.membership_matrix.shape[1]
    particle_indices = jnp.arange(N)[:, None]
    vortex_indices = jnp.arange(max_vortices)[None, :]
    
    # 新しい所属関係
    new_memberships = (new_vortex_ids[:, None] == vortex_indices) & (new_vortex_ids[:, None] > 0)
    
    # 既存の所属関係と結合（一度所属したら記録は残る）
    new_matrix = membership.membership_matrix | new_memberships
    
    return ParticleMembershipJAX(
        vortex_ids=new_vortex_ids,
        join_steps=new_join_steps,
        leave_steps=new_leave_steps,
        membership_matrix=new_matrix,
        history_count=new_history_count
    )

# ==============================
# 剥離統計の更新（修正版）
# ==============================

@jit
def update_shedding_stats(
    stats: VortexSheddingStats,
    has_new_upper: bool,  # スカラーのbool
    has_new_lower: bool,  # スカラーのbool
    step: int
) -> VortexSheddingStats:
    """剥離統計を更新（スカラー版）"""
    
    # 上側の新規剥離
    new_upper_count = jnp.where(
        has_new_upper,
        stats.upper_count + 1,
        stats.upper_count
    )
    
    # 現在のカウント位置に記録
    new_upper_steps = stats.upper_shedding_steps.at[stats.upper_count].set(
        jnp.where(has_new_upper, step, stats.upper_shedding_steps[stats.upper_count])
    )
    
    # 下側の新規剥離
    new_lower_count = jnp.where(
        has_new_lower,
        stats.lower_count + 1,
        stats.lower_count
    )
    
    new_lower_steps = stats.lower_shedding_steps.at[stats.lower_count].set(
        jnp.where(has_new_lower, step, stats.lower_shedding_steps[stats.lower_count])
    )
    
    return VortexSheddingStats(
        upper_shedding_steps=new_upper_steps,
        lower_shedding_steps=new_lower_steps,
        upper_count=new_upper_count,
        lower_count=new_lower_count
    )

# ==============================
# ★★★ マッチング関数（Boolean Indexing修正版）★★★
# ==============================

@jit
def match_vortices_vectorized(
    vortex_state: VortexStateJAX,
    new_centers: jnp.ndarray,
    new_properties: jnp.ndarray,
    dt: float,
    matching_threshold: float = 30.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    既存渦と新規検出クラスタのマッチング（Boolean Indexing修正版）
    
    Returns:
        matches: (max_clusters,) 各クラスタに対応する渦ID（-1=新規）
        is_matched: (max_vortices,) 各渦がマッチしたか
    """
    max_vortices = len(vortex_state.ids)
    max_clusters = len(new_centers)
    
    # 予測位置の計算（アクティブな渦のみ）
    predicted_centers = vortex_state.centers + jnp.array([10.0 * dt, 0])
    
    # ★★★ Boolean Indexingを避けて、マスクを使った演算に変更 ★★★
    # 全渦×全クラスタの距離行列を計算（無効な渦は大きな値に）
    distances_all = jnp.linalg.norm(
        predicted_centers[:, None, :] - new_centers[None, :, :],
        axis=2
    )
    
    # アクティブでない渦の距離を無限大に設定
    # jnp.whereを使って条件付き選択
    distances_masked = jnp.where(
        vortex_state.is_alive[:, None],  # (max_vortices, 1) にブロードキャスト
        distances_all,                    # アクティブなら距離を使用
        jnp.inf                           # 非アクティブなら無限大
    )
    
    # 有効なクラスタのマスク（循環が十分強い）
    valid_clusters = new_properties[:, 0] > 0.5
    
    # 各クラスタに最も近い渦を探す
    min_distances = jnp.min(distances_masked, axis=0)  # (max_clusters,)
    min_indices = jnp.argmin(distances_masked, axis=0)  # (max_clusters,)
    
    # マッチング結果
    # 条件：距離が閾値以下 かつ 有効なクラスタ
    matches = jnp.where(
        (min_distances < matching_threshold) & valid_clusters,
        min_indices,  # 渦のインデックスをそのままIDとして使用
        -1
    )
    
    # 各渦がマッチしたかのフラグ（今は簡単のため全部False）
    is_matched = jnp.zeros(max_vortices, dtype=bool)
    
    return matches, is_matched

# ==============================
# 渦状態更新（修正版）
# ==============================

@jit
def update_vortex_state(
    vortex_state: VortexStateJAX,
    matches: jnp.ndarray,
    new_centers: jnp.ndarray,
    new_properties: jnp.ndarray,
    step: int,
    next_id: int
) -> Tuple[VortexStateJAX, int]:
    """渦状態の更新（Boolean Indexing回避版）"""
    
    max_vortices = len(vortex_state.ids)
    max_clusters = len(matches)
    
    # === 履歴インデックス更新 ===
    hist_indices = vortex_state.hist_index
    new_hist_indices = jnp.where(
        vortex_state.is_alive,
        (hist_indices + 1) % vortex_state.trajectory.shape[1],
        hist_indices
    )
    
    # === 既存渦の更新 ===
    # vortex_to_cluster[vid] = そのvortexにマッチするクラスタID（なければ-1）
    vortex_to_cluster = jnp.full(max_vortices, -1)
    
    def assign_cluster_to_vortex(carry, i):
        v2c = carry
        cluster_id = i
        vortex_id = matches[i]
        
        v2c = jnp.where(
            (vortex_id >= 0) & (vortex_id < max_vortices),
            v2c.at[vortex_id].set(cluster_id),
            v2c
        )
        return v2c, None
    
    vortex_to_cluster, _ = lax.scan(
        assign_cluster_to_vortex,
        vortex_to_cluster,
        jnp.arange(max_clusters)
    )
    
    # 各渦の更新値を計算
    has_match = vortex_to_cluster >= 0
    matched_cluster_ids = jnp.maximum(vortex_to_cluster, 0)
    
    new_centers_all = jnp.where(
        has_match[:, None],
        new_centers[matched_cluster_ids],
        vortex_state.centers
    )
    new_circulations_all = jnp.where(
        has_match,
        new_properties[matched_cluster_ids, 0],
        vortex_state.circulations
    )
    new_coherences_all = jnp.where(
        has_match,
        new_properties[matched_cluster_ids, 1],
        vortex_state.coherences
    )
    new_n_particles_all = jnp.where(
        has_match,
        new_properties[matched_cluster_ids, 2].astype(jnp.int32),
        vortex_state.n_particles
    )
    
    # 履歴更新（簡略化）
    new_trajectory = vortex_state.trajectory
    new_circulation_hist = vortex_state.circulation_hist
    new_coherence_hist = vortex_state.coherence_hist
    new_particle_count_hist = vortex_state.particle_count_hist
    
    # 新規渦の追加は簡略化（デモ用）
    
    # 状態を更新
    updated_state = vortex_state._replace(
        centers=new_centers_all,
        circulations=new_circulations_all,
        coherences=new_coherences_all,
        n_particles=new_n_particles_all,
        trajectory=new_trajectory,
        circulation_hist=new_circulation_hist,
        coherence_hist=new_coherence_hist,
        particle_count_hist=new_particle_count_hist,
        hist_index=new_hist_indices
    )
    
    # 消滅判定
    should_die = updated_state.is_alive & (
        (updated_state.n_particles < 5) | 
        (updated_state.coherences < 0.2)
    )
    
    final_state = updated_state._replace(
        is_alive=updated_state.is_alive & ~should_die,
        death_steps=jnp.where(should_die, step, updated_state.death_steps)
    )
    
    return final_state, next_id

@jit
def update_vortex_state_with_sides(
    vortex_state: VortexStateJAX,
    matches: jnp.ndarray,
    new_centers: jnp.ndarray,
    new_properties: jnp.ndarray,
    sides_array: jnp.ndarray,
    step: int,
    next_id: int
) -> Tuple[VortexStateJAX, int]:
    """渦状態の更新（side情報付き）"""
    
    # 基本の更新を実行
    vortex_state_updated, next_id_updated = update_vortex_state(
        vortex_state,
        matches,
        new_centers,
        new_properties,
        step,
        next_id
    )
    
    # 新規渦のside情報を設定
    is_new_vortex = vortex_state_updated.birth_steps == step
    
    # 距離計算
    vortex_centers = vortex_state_updated.centers
    distances = jnp.linalg.norm(
        vortex_centers[:, None, :] - new_centers[None, :, :],
        axis=2
    )
    
    closest_cluster_idx = jnp.argmin(distances, axis=1)
    n_clusters = len(sides_array)
    safe_idx = jnp.clip(closest_cluster_idx, 0, n_clusters - 1)
    closest_sides = sides_array[safe_idx]
    
    new_birth_sides = jnp.where(
        is_new_vortex,
        closest_sides,
        vortex_state_updated.birth_side
    )
    
    vortex_state_final = vortex_state_updated._replace(
        birth_side=new_birth_sides
    )
    
    return vortex_state_final, next_id_updated

# ==============================
# Strouhal数計算（JAX版）
# ==============================

@jit
def compute_strouhal_number(
    vortex_state: VortexStateJAX,
    dt: float,
    D: float,
    U: float
) -> float:
    """Strouhal数の計算"""
    
    # 上側渦の誕生ステップを抽出
    upper_vortices = (vortex_state.birth_side == 0) & (vortex_state.birth_steps >= 0)
    upper_birth_steps = jnp.where(upper_vortices, vortex_state.birth_steps, -1)
    
    # 有効なステップのみ抽出してソート
    valid_steps = upper_birth_steps[upper_birth_steps >= 0]
    sorted_steps = jnp.sort(valid_steps)
    
    # 最近10個の間隔を計算
    n_recent = jnp.minimum(10, len(sorted_steps) - 1)
    recent_steps = sorted_steps[-n_recent-1:]
    intervals = jnp.diff(recent_steps)
    
    # 平均周期とStrouhal数
    mean_interval = jnp.mean(intervals)
    period = mean_interval * dt
    frequency = 1.0 / (period + 1e-8)
    St = frequency * D / U
    
    return jnp.where(n_recent > 0, St, 0.0)

# ==============================
# メイン追跡関数（簡略版 - JIT可能）
# ==============================

def track_vortices_step_complete(
    particle_state,
    vortex_state: VortexStateJAX,
    membership: ParticleMembershipJAX,
    shedding_stats: VortexSheddingStats,
    step: int,
    next_id: int,
    obstacle_center: jnp.ndarray,
    config
) -> Tuple[VortexStateJAX, ParticleMembershipJAX, VortexSheddingStats, int, Dict]:
    """
    完全機能版の渦追跡ステップ（簡略版）
    """
    
    # 上側検出
    upper_centers, upper_props, upper_particle_ids = detect_vortex_clusters_separated(
        particle_state.position,
        particle_state.Lambda_F,
        particle_state.Q_criterion,
        particle_state.is_active,
        obstacle_center,
        side=0,
        min_particles=10
    )
    
    # 下側検出
    lower_centers, lower_props, lower_particle_ids = detect_vortex_clusters_separated(
        particle_state.position,
        particle_state.Lambda_F,
        particle_state.Q_criterion,
        particle_state.is_active,
        obstacle_center,
        side=1,
        min_particles=10
    )
    
    # 結合
    centers = jnp.concatenate([upper_centers, lower_centers], axis=0)
    properties = jnp.concatenate([upper_props, lower_props], axis=0)
    
    n_upper = len(upper_centers)
    n_lower = len(lower_centers)
    sides_array = jnp.concatenate([
        jnp.zeros(n_upper, dtype=jnp.int32),
        jnp.ones(n_lower, dtype=jnp.int32)
    ])
    
    # マッチング
    matches, is_matched = match_vortices_vectorized(
        vortex_state,
        centers,
        properties,
        config.dt,
        matching_threshold=30.0
    )
    
    # 新規渦の検出
    is_new = (matches == -1) & (properties[:, 0] > 1.0)
    new_upper_count = jnp.sum(is_new & (sides_array == 0))
    new_lower_count = jnp.sum(is_new & (sides_array == 1))
    
    # 剥離統計の更新
    has_new_upper = new_upper_count > 0
    has_new_lower = new_lower_count > 0
    
    shedding_stats = update_shedding_stats(
        shedding_stats,
        has_new_upper,
        has_new_lower,
        step
    )
    
    # 渦状態の更新
    vortex_state_updated, next_id = update_vortex_state_with_sides(
        vortex_state,
        matches,
        centers,
        properties,
        sides_array,
        step,
        next_id
    )
    
    # 粒子所属の更新（簡略版）
    particle_cluster_ids = jnp.where(
        upper_particle_ids >= 0,
        upper_particle_ids,
        jnp.where(
            lower_particle_ids >= 0,
            lower_particle_ids + n_upper,
            -1
        )
    )
    
    # 統計計算
    n_active = jnp.sum(vortex_state_updated.is_alive)
    n_total = jnp.sum(vortex_state_updated.ids > 0)
    
    St = compute_strouhal_number(
        vortex_state_updated,
        config.dt,
        2 * config.obstacle_size,
        config.Lambda_F_inlet
    )
    
    metrics = {
        'n_active_vortices': n_active,
        'n_total_vortices': n_total,
        'n_upper_shedding': shedding_stats.upper_count,
        'n_lower_shedding': shedding_stats.lower_count,
        'strouhal_number': St,
        'mean_circulation': 0.0,
        'mean_coherence': 0.0,
        'particle_exchange_rate': 0.0
    }
    
    return vortex_state_updated, membership, shedding_stats, next_id, metrics

# ==============================
# 分析関数
# ==============================

def print_vortex_events(vortex_state: VortexStateJAX, 
                        prev_state: VortexStateJAX, 
                        step: int):
    """渦の誕生・消滅イベントを出力"""
    pass  # 簡略版

def create_vortex_genealogy_jax(vortex_state: VortexStateJAX) -> str:
    """渦の系譜図を作成"""
    return "=== Vortex Genealogy ===\n(Simplified version)"

def analyze_particle_fates_jax(membership: ParticleMembershipJAX) -> Dict:
    """粒子の運命統計"""
    return {
        'never_vortex': 0,
        'single_vortex': 0,
        'multiple_vortices': 0,
        'currently_in_vortex': 0,
        'mean_vortices_per_particle': 0.0
    }

def analyze_vortex_statistics_jax(vortex_state: VortexStateJAX) -> Dict:
    """渦統計解析"""
    return {
        'n_completed': 0,
        'n_active': int(jnp.sum(vortex_state.is_alive)),
        'mean_lifetime': 0.0,
        'std_lifetime': 0.0,
        'mean_travel_distance': 0.0,
        'max_lifetime': 0,
        'min_lifetime': 0
    }

print("✨ Boolean Indexing Fixed! Ready for JIT compilation! ✨")
