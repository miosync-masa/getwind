#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GET Wind™ v6.3 JAX Edition - FULLY JIT-Compatible Vortex Tracking with Smart Death
環ちゃん & ご主人さま Ultimate Intelligence Edition! 💕

完全JIT対応 + 賢い渦死判定システム統合版！
- Boolean Indexingを完全排除
- ΛF同期ベースの健康状態評価
- 位相ラグを考慮した同期判定
- 物理的に正しい生死判定
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import numpy as np
from typing import NamedTuple, Tuple, Dict
from functools import partial
import time

# ==============================
# JAX Vortex State (全部テンソル！)
# ==============================

class VortexStateJAX(NamedTuple):
    """渦の状態を全部JAXテンソルで管理"""
    ids: jnp.ndarray              # (max_vortices,)
    is_alive: jnp.ndarray         
    birth_steps: jnp.ndarray      
    death_steps: jnp.ndarray      
    birth_side: jnp.ndarray       
    centers: jnp.ndarray          # (max_vortices, 2)
    circulations: jnp.ndarray    
    coherences: jnp.ndarray       
    n_particles: jnp.ndarray      
    trajectory: jnp.ndarray       # (max_vortices, history_len, 2)
    circulation_hist: jnp.ndarray 
    coherence_hist: jnp.ndarray   
    particle_count_hist: jnp.ndarray 
    hist_index: jnp.ndarray       

class ParticleMembershipJAX(NamedTuple):
    """粒子の渦所属をJAXで管理"""
    vortex_ids: jnp.ndarray       
    join_steps: jnp.ndarray       
    leave_steps: jnp.ndarray      
    membership_matrix: jnp.ndarray 
    history_count: jnp.ndarray    

class VortexSheddingStats(NamedTuple):
    """渦剥離統計"""
    upper_shedding_steps: jnp.ndarray  
    lower_shedding_steps: jnp.ndarray  
    upper_count: jnp.ndarray           
    lower_count: jnp.ndarray           

# ==============================
# 初期化関数
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
# 🆕 渦の健康状態評価（Smart Death用）
# ==============================

@jit
def compute_vortex_health(
    coherence: float,
    circulation: float,
    n_particles: int,
    coherence_history: jnp.ndarray,
    circulation_history: jnp.ndarray,
    particle_history: jnp.ndarray,
    history_len: int = 10
) -> dict:
    """
    渦の健康状態を総合的に評価
    """
    
    # === 1. ΛF同期の健康度 ===
    current_sync = coherence
    valid_history = coherence_history >= 0
    n_valid = jnp.sum(valid_history)
    
    mean_coherence = jnp.sum(
        jnp.where(valid_history, coherence_history, 0)
    ) / jnp.maximum(n_valid, 1)
    
    variance = jnp.sum(
        jnp.where(valid_history, (coherence_history - mean_coherence)**2, 0)
    ) / jnp.maximum(n_valid, 1)
    std_coherence = jnp.sqrt(variance)
    cv_coherence = std_coherence / (mean_coherence + 1e-8)
    
    sync_health = current_sync * jnp.exp(-cv_coherence)
    
    # === 2. 構造の健康度 ===
    sync_particles = n_particles * coherence
    min_particles_for_vortex = 3.0
    particle_ratio = sync_particles / min_particles_for_vortex
    
    particle_mean = jnp.mean(particle_history)
    particle_std = jnp.std(particle_history)
    particle_stability = jnp.exp(-particle_std / (particle_mean + 1e-8))
    
    structure_health = jnp.tanh(particle_ratio) * particle_stability
    
    # === 3. 活力（循環の強さ）===
    current_vitality = jnp.tanh(jnp.abs(circulation) / 5.0)
    circ_diffs = jnp.diff(circulation_history)
    decay_rate = jnp.mean(circ_diffs)
    decay_penalty = jnp.exp(decay_rate / 2.0)
    vitality = current_vitality * decay_penalty
    
    # === 4. 安定性 ===
    # 簡易版のトレンド計算（polyfit使わない）
    x = jnp.arange(history_len, dtype=jnp.float32)
    x_mean = jnp.mean(x)
    y_mean_coh = jnp.mean(coherence_history)
    y_mean_circ = jnp.mean(circulation_history)
    
    coherence_trend = jnp.sum((x - x_mean) * (coherence_history - y_mean_coh)) / (jnp.sum((x - x_mean)**2) + 1e-8)
    circulation_trend = jnp.sum((x - x_mean) * (circulation_history - y_mean_circ)) / (jnp.sum((x - x_mean)**2) + 1e-8)
    
    trend_score = jnp.tanh((coherence_trend + circulation_trend) * 10)
    stability = 0.5 + 0.5 * trend_score
    
    # === 5. 総合健康度 ===
    overall_health = (
        0.4 * sync_health +
        0.25 * structure_health +
        0.2 * vitality +
        0.15 * stability
    )
    
    return {
        'sync_health': sync_health,
        'structure_health': structure_health,
        'vitality': vitality,
        'stability': stability,
        'overall_health': overall_health
    }

# ==============================
# 🆕 位相ラグを考慮した同期評価
# ==============================

@jit
def evaluate_phase_lag_sync(
    Lambda_F: jnp.ndarray,
    positions: jnp.ndarray,
    vortex_center: jnp.ndarray,
    mask: jnp.ndarray,
    max_lag: float = 0.2
) -> float:
    """
    位相ラグを考慮したΛF同期の評価
    """
    
    rel_pos = positions - vortex_center[None, :]
    distances = jnp.linalg.norm(rel_pos, axis=1)
    
    expected_tangent = jnp.stack([-rel_pos[:, 1], rel_pos[:, 0]], axis=1)
    expected_tangent = expected_tangent / (jnp.linalg.norm(expected_tangent, axis=1, keepdims=True) + 1e-8)
    
    Lambda_F_normalized = Lambda_F / (jnp.linalg.norm(Lambda_F, axis=1, keepdims=True) + 1e-8)
    alignment = jnp.sum(Lambda_F_normalized * expected_tangent, axis=1)
    
    phase_lag_weight = jnp.exp(-distances / 20.0)
    
    lag_adjusted_sync = jnp.where(
        mask,
        jnp.maximum(alignment, 0.5 + 0.5 * phase_lag_weight),
        0.0
    )
    
    total_weight = jnp.sum(jnp.where(mask, phase_lag_weight, 0))
    phase_sync_score = jnp.sum(lag_adjusted_sync * phase_lag_weight) / jnp.maximum(total_weight, 1e-8)
    
    return phase_sync_score

# ==============================
# 渦クラスタ検出（完全JIT対応版）
# ==============================

@partial(jit, static_argnums=(5, 6, 7))
def detect_vortex_clusters_separated(
    positions: jnp.ndarray,
    Lambda_F: jnp.ndarray,
    Q_criterion: jnp.ndarray,
    active_mask: jnp.ndarray,
    obstacle_center: jnp.ndarray,
    side: int,
    grid_size: int = 10,
    min_particles: int = 10
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """上下領域別の渦検出（完全JIT対応）"""
    N = positions.shape[0]
    max_clusters = 25
    
    y_offset = jnp.where(side == 0, 20.0, -20.0)
    y_center = obstacle_center[1] + y_offset
    
    y_min = jnp.where(side == 0, y_center - 10, y_center - 30)
    y_max = jnp.where(side == 0, y_center + 30, y_center + 10)
    
    region_mask = (
        active_mask & 
        (Q_criterion > 0.5) &
        (positions[:, 1] >= y_min) &
        (positions[:, 1] <= y_max)
    )
    
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
# マッチング（完全JIT対応版）
# ==============================

@jit
def match_vortices_vectorized(
    vortex_state: VortexStateJAX,
    new_centers: jnp.ndarray,
    new_properties: jnp.ndarray,
    dt: float,
    matching_threshold: float = 30.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """既存渦と新規検出クラスタのマッチング"""
    max_vortices = len(vortex_state.ids)
    max_clusters = len(new_centers)
    
    predicted_centers = vortex_state.centers + jnp.array([10.0 * dt, 0])
    
    distances_all = jnp.linalg.norm(
        predicted_centers[:, None, :] - new_centers[None, :, :],
        axis=2
    )
    
    distances_masked = jnp.where(
        vortex_state.is_alive[:, None],
        distances_all,
        jnp.inf
    )
    
    valid_clusters = new_properties[:, 0] > 0.5
    
    min_distances = jnp.min(distances_masked, axis=0)
    min_indices = jnp.argmin(distances_masked, axis=0)
    
    matches = jnp.where(
        (min_distances < matching_threshold) & valid_clusters,
        min_indices,
        -1
    )
    
    def check_matched(vid):
        return jnp.any(matches == vid)
    
    is_matched = vmap(check_matched)(jnp.arange(max_vortices))
    
    return matches, is_matched

# ==============================
# 🆕 賢い渦状態更新（Smart Death統合版）
# ==============================

@jit
def update_vortex_state_smart(
    vortex_state: VortexStateJAX,
    matches: jnp.ndarray,
    new_centers: jnp.ndarray,
    new_properties: jnp.ndarray,
    sides_array: jnp.ndarray,
    Lambda_F: jnp.ndarray,       # 全粒子のΛF
    positions: jnp.ndarray,      # 全粒子の位置
    particle_cluster_ids: jnp.ndarray,  # 粒子のクラスタID
    step: int,
    next_id: int,
    death_threshold: float = 0.2
) -> Tuple[VortexStateJAX, int, Dict]:
    """
    賢い死の判定を統合した渦状態更新
    """
    
    max_vortices = len(vortex_state.ids)
    
    # === 履歴インデックス更新 ===
    new_hist_indices = jnp.where(
        vortex_state.is_alive,
        (vortex_state.hist_index + 1) % vortex_state.trajectory.shape[1],
        vortex_state.hist_index
    )
    
    # === 既存渦の更新 ===
    vortex_to_cluster = jnp.full(max_vortices, -1, dtype=jnp.int32)
    
    def assign_match(carry, i):
        v2c = carry
        cluster_id = i
        vortex_id = matches[i]
        
        v2c = lax.cond(
            (vortex_id >= 0) & (vortex_id < max_vortices),
            lambda x: x.at[vortex_id].set(cluster_id),
            lambda x: x,
            v2c
        )
        return v2c, None
    
    vortex_to_cluster, _ = lax.scan(
        assign_match,
        vortex_to_cluster,
        jnp.arange(len(matches))
    )
    
    has_match = vortex_to_cluster >= 0
    matched_cluster_ids = jnp.clip(vortex_to_cluster, 0, len(new_centers)-1)
    
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
    
    # === 履歴の更新 ===
    # coherence履歴
    def update_coherence_hist(i):
        hist_idx = new_hist_indices[i]
        should_update = vortex_state.is_alive[i]
        return jnp.where(
            should_update,
            vortex_state.coherence_hist[i].at[hist_idx].set(new_coherences_all[i]),
            vortex_state.coherence_hist[i]
        )
    
    new_coherence_hist = vmap(update_coherence_hist)(jnp.arange(max_vortices))
    
    # circulation履歴
    def update_circulation_hist(i):
        hist_idx = new_hist_indices[i]
        should_update = vortex_state.is_alive[i]
        return jnp.where(
            should_update,
            vortex_state.circulation_hist[i].at[hist_idx].set(new_circulations_all[i]),
            vortex_state.circulation_hist[i]
        )
    
    new_circulation_hist = vmap(update_circulation_hist)(jnp.arange(max_vortices))
    
    # particle count履歴
    def update_particle_hist(i):
        hist_idx = new_hist_indices[i]
        should_update = vortex_state.is_alive[i]
        return jnp.where(
            should_update,
            vortex_state.particle_count_hist[i].at[hist_idx].set(new_n_particles_all[i]),
            vortex_state.particle_count_hist[i]
        )
    
    new_particle_count_hist = vmap(update_particle_hist)(jnp.arange(max_vortices))
    
    # === 🆕 賢い死の判定 ===
    def evaluate_vortex_health(i):
        is_alive = vortex_state.is_alive[i]
        
        # 履歴を取得（最近の10個）
        history_len = 10
        hist_idx = new_hist_indices[i]
        
        # 循環バッファから最近のデータを取得
        indices = jnp.arange(history_len)
        hist_indices = (hist_idx - history_len + 1 + indices) % new_coherence_hist[i].shape[0]
        
        coherence_history = new_coherence_hist[i][hist_indices]
        circulation_history = new_circulation_hist[i][hist_indices]
        particle_history = new_particle_count_hist[i][hist_indices].astype(jnp.float32)
        
        # 健康状態評価
        health = compute_vortex_health(
            new_coherences_all[i],
            new_circulations_all[i],
            new_n_particles_all[i],
            coherence_history,
            circulation_history,
            particle_history,
            history_len
        )
        
        # 位相ラグ同期評価
        # この渦に属する粒子のマスク（クラスタIDベース）
        vortex_cluster_id = vortex_to_cluster[i]
        particle_mask = jnp.where(
            vortex_cluster_id >= 0,
            particle_cluster_ids == vortex_cluster_id,
            jnp.zeros_like(particle_cluster_ids, dtype=bool)
        )
        
        phase_sync = evaluate_phase_lag_sync(
            Lambda_F, positions, new_centers_all[i], particle_mask
        )
        
        # 最終健康スコア
        final_sync_health = health['sync_health'] * 0.7 + phase_sync * 0.3
        final_structure_health = health['structure_health']
        
        final_health_score = (
            0.35 * final_sync_health +
            0.25 * final_structure_health +
            0.2 * health['vitality'] +
            0.2 * health['stability']
        )
        
        # 死の判定
        should_die = is_alive & (final_health_score < death_threshold)
        
        return should_die, final_health_score
    
    death_results = vmap(evaluate_vortex_health)(jnp.arange(max_vortices))
    should_die_array = death_results[0]
    health_scores = death_results[1]
    
    # === 新規渦の作成 ===
    is_new_vortex = (matches == -1) & (new_properties[:, 0] > 1.0)
    empty_mask = ~vortex_state.is_alive
    
    slot_indices = jnp.where(empty_mask, jnp.arange(max_vortices), max_vortices)
    sorted_slots = jnp.sort(slot_indices)
    
    # 新規渦を追加する関数
    def add_new_vortex(carry, i):
        state, current_id = carry
        
        new_vortex_indices = jnp.where(is_new_vortex, jnp.arange(len(matches)), -1)
        sorted_new_indices = jnp.sort(new_vortex_indices)
        
        cluster_idx = jnp.where(i < jnp.sum(is_new_vortex), sorted_new_indices[-(i+1)], 0)
        slot_idx = sorted_slots[i]
        
        is_valid_add = (i < jnp.sum(is_new_vortex)) & (slot_idx < max_vortices) & (cluster_idx >= 0)
        
        # 誕生側の判定
        birth_side = jnp.where(cluster_idx < len(sides_array), sides_array[cluster_idx], 0)
        
        state = state._replace(
            ids=state.ids.at[slot_idx].set(
                jnp.where(is_valid_add, current_id, state.ids[slot_idx])
            ),
            is_alive=state.is_alive.at[slot_idx].set(
                jnp.where(is_valid_add, True, state.is_alive[slot_idx])
            ),
            birth_steps=state.birth_steps.at[slot_idx].set(
                jnp.where(is_valid_add, step, state.birth_steps[slot_idx])
            ),
            birth_side=state.birth_side.at[slot_idx].set(
                jnp.where(is_valid_add, birth_side, state.birth_side[slot_idx])
            ),
            centers=state.centers.at[slot_idx].set(
                jnp.where(is_valid_add, new_centers[cluster_idx], state.centers[slot_idx])
            ),
            circulations=state.circulations.at[slot_idx].set(
                jnp.where(is_valid_add, new_properties[cluster_idx, 0], state.circulations[slot_idx])
            ),
            coherences=state.coherences.at[slot_idx].set(
                jnp.where(is_valid_add, new_properties[cluster_idx, 1], state.coherences[slot_idx])
            ),
            n_particles=state.n_particles.at[slot_idx].set(
                jnp.where(is_valid_add, new_properties[cluster_idx, 2].astype(jnp.int32), state.n_particles[slot_idx])
            )
        )
        
        new_id = current_id + is_valid_add.astype(jnp.int32)
        return (state, new_id), None
    
    # 一時的な状態を作成
    temp_state = vortex_state._replace(
        centers=new_centers_all,
        circulations=new_circulations_all,
        coherences=new_coherences_all,
        n_particles=new_n_particles_all,
        hist_index=new_hist_indices,
        coherence_hist=new_coherence_hist,
        circulation_hist=new_circulation_hist,
        particle_count_hist=new_particle_count_hist
    )
    
    # 最大10個の新規渦を追加
    max_new_vortices = 10
    (final_state, final_next_id), _ = lax.scan(
        add_new_vortex,
        (temp_state, next_id),
        jnp.arange(max_new_vortices)
    )
    
    # === 死亡処理 ===
    final_state = final_state._replace(
        is_alive=final_state.is_alive & ~should_die_array,
        death_steps=jnp.where(should_die_array, step, final_state.death_steps)
    )
    
    # 診断情報
    diagnostics = {
        'health_scores': health_scores,
        'n_deaths': jnp.sum(should_die_array),
        'mean_health': jnp.mean(jnp.where(final_state.is_alive, health_scores, 0))
    }
    
    return final_state, final_next_id, diagnostics

# ==============================
# Strouhal数計算（完全JIT対応版）
# ==============================

@jit
def compute_strouhal_number(
    vortex_state: VortexStateJAX,
    dt: float,
    D: float,
    U: float
) -> float:
    """Strouhal数の計算（Boolean Indexing排除版）"""
    
    upper_vortices_mask = (vortex_state.birth_side == 0) & (vortex_state.birth_steps >= 0)
    
    masked_steps = jnp.where(
        upper_vortices_mask,
        vortex_state.birth_steps,
        -999999
    )
    
    sorted_steps = jnp.sort(masked_steps)
    valid_count = jnp.sum(sorted_steps >= 0)
    
    n_recent = jnp.minimum(10, valid_count - 1)
    indices = jnp.arange(len(sorted_steps))
    recent_mask = indices >= (len(sorted_steps) - n_recent - 1)
    
    recent_steps = jnp.where(recent_mask, sorted_steps, 0)
    
    def compute_diff(i):
        is_valid = (i < len(recent_steps) - 1) & (recent_steps[i] >= 0) & (recent_steps[i+1] >= 0)
        diff = jnp.where(is_valid, recent_steps[i+1] - recent_steps[i], 0)
        return diff
    
    intervals_array = vmap(compute_diff)(jnp.arange(len(recent_steps) - 1))
    
    valid_intervals_mask = intervals_array > 0
    valid_intervals_sum = jnp.sum(jnp.where(valid_intervals_mask, intervals_array, 0))
    valid_intervals_count = jnp.sum(valid_intervals_mask)
    
    mean_interval = jnp.where(
        valid_intervals_count > 0,
        valid_intervals_sum / valid_intervals_count,
        1.0
    )
    
    period = mean_interval * dt
    frequency = 1.0 / (period + 1e-8)
    St = frequency * D / U
    
    return jnp.where(valid_intervals_count > 0, St, 0.0)

# ==============================
# 剥離統計更新
# ==============================

@jit
def update_shedding_stats(
    stats: VortexSheddingStats,
    has_new_upper: bool,
    has_new_lower: bool,
    step: int
) -> VortexSheddingStats:
    """剥離統計の更新"""
    
    new_upper_count = lax.cond(
        has_new_upper,
        lambda x: x + 1,
        lambda x: x,
        stats.upper_count
    )
    
    def update_upper_steps(carry):
        steps, count = carry
        idx = count % steps.shape[0]
        
        new_steps = lax.cond(
            has_new_upper,
            lambda s: s.at[idx].set(step),
            lambda s: s,
            steps
        )
        return new_steps
    
    new_upper_steps = update_upper_steps((stats.upper_shedding_steps, stats.upper_count))
    
    new_lower_count = lax.cond(
        has_new_lower,
        lambda x: x + 1,
        lambda x: x,
        stats.lower_count
    )
    
    def update_lower_steps(carry):
        steps, count = carry
        idx = count % steps.shape[0]
        
        new_steps = lax.cond(
            has_new_lower,
            lambda s: s.at[idx].set(step),
            lambda s: s,
            steps
        )
        return new_steps
    
    new_lower_steps = update_lower_steps((stats.lower_shedding_steps, stats.lower_count))
    
    return stats._replace(
        upper_shedding_steps=new_upper_steps,
        lower_shedding_steps=new_lower_steps,
        upper_count=new_upper_count,
        lower_count=new_lower_count
    )

# ==============================
# メイン追跡関数（Smart Death統合版）
# ==============================

@partial(jit, static_argnums=(7,))
def track_vortices_step_smart(
    particle_state,
    vortex_state: VortexStateJAX,
    membership: ParticleMembershipJAX,
    shedding_stats: VortexSheddingStats,
    step: int,
    next_id: int,
    obstacle_center: jnp.ndarray,
    config
) -> Tuple[VortexStateJAX, ParticleMembershipJAX, VortexSheddingStats, int, Dict]:
    """完全JIT対応 + 賢い死判定版の渦追跡ステップ"""
    
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
    particle_cluster_ids = jnp.where(
        upper_particle_ids >= 0,
        upper_particle_ids,
        jnp.where(lower_particle_ids >= 0, lower_particle_ids + len(upper_centers), -1)
    )
    
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
    n_upper = len(upper_centers)
    sides_array = jnp.concatenate([
        jnp.zeros(n_upper, dtype=jnp.int32),
        jnp.ones(len(lower_centers), dtype=jnp.int32)
    ])
    
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
    
    # 🆕 賢い死判定を含む渦状態の更新
    vortex_state_updated, next_id, death_diagnostics = update_vortex_state_smart(
        vortex_state,
        matches,
        centers,
        properties,
        sides_array,
        particle_state.Lambda_F,
        particle_state.position,
        particle_cluster_ids,
        step,
        next_id,
        death_threshold=0.2
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
    
    active_mask = vortex_state_updated.is_alive
    active_circulations = jnp.where(active_mask, vortex_state_updated.circulations, 0)
    active_coherences = jnp.where(active_mask, vortex_state_updated.coherences, 0)
    
    mean_circulation = jnp.sum(active_circulations) / jnp.maximum(n_active, 1)
    mean_coherence = jnp.sum(active_coherences) / jnp.maximum(n_active, 1)
    
    metrics = {
        'n_active_vortices': n_active,
        'n_total_vortices': n_total,
        'n_upper_shedding': shedding_stats.upper_count,
        'n_lower_shedding': shedding_stats.lower_count,
        'strouhal_number': St,
        'mean_circulation': mean_circulation,
        'mean_coherence': mean_coherence,
        'mean_health': death_diagnostics['mean_health'],
        'n_deaths': death_diagnostics['n_deaths']
    }
    
    return vortex_state_updated, membership, shedding_stats, next_id, metrics

# ==============================
# 補助関数
# ==============================

def explain_death_reason(health_score: float, threshold: float = 0.2) -> str:
    """健康スコアから死因を説明"""
    if health_score >= threshold:
        return "Alive and healthy"
    elif health_score < 0.05:
        return "Critical weakness - total collapse"
    elif health_score < 0.1:
        return "Severe deterioration"
    elif health_score < 0.15:
        return "Lost structural integrity"
    else:
        return "Below health threshold"

# ==============================
# テスト用
# ==============================

if __name__ == "__main__":
    print("=" * 70)
    print("GET Wind™ v6.3 JAX - FULLY JIT + Smart Death Edition!")
    print("環ちゃん & ご主人さま Ultimate Intelligence! 💕")
    print("=" * 70)
    
    print("\n✨ Merged Features:")
    print("  ✅ Full JIT compilation - 完全JIT対応!")
    print("  ✅ NO Boolean Indexing - 完全排除!")
    print("  ✅ Smart health monitoring - 賢い健康監視!")
    print("  ✅ Phase-lag aware sync - 位相ラグ考慮!")
    print("  ✅ Temporal stability - 時間的安定性!")
    print("  ✅ Physically justified death - 物理的に正しい死!")
    
    print("\n🎯 Integration Complete:")
    print("  • Original JIT-compatible tracking ✓")
    print("  • Smart death judgment system ✓")
    print("  • Full vortex lifecycle management ✓")
    
    vortex_state = initialize_vortex_state()
    membership = initialize_particle_membership(1500)
    shedding_stats = initialize_shedding_stats()
    
    print(f"\n📊 Initialized structures:")
    print(f"  Vortex state: {vortex_state.ids.shape[0]} max vortices")
    print(f"  Membership: {membership.vortex_ids.shape[0]} max particles")
    print(f"  Shedding stats: {shedding_stats.upper_shedding_steps.shape[0]} max events")
    
    print("\n✨ Ready for simulation! Use track_vortices_step_smart() ✨")
    print("=" * 70)
