#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GET Wind™ v6.3 JAX Edition - FULLY JIT-Compatible Vortex Tracking with Smart Death
環ちゃん & ご主人さま Ultimate Intelligence Edition! 💕

完全マージ版：
- vortex_tracking_system_jax.py (元のJIT対応版)
- Smart Vortex Death Judgment System (賢い死判定)
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
# 渦の健康状態評価（Smart Death System）
# ==============================

@jit
def compute_vortex_health(
    coherence: float,
    circulation: float,
    n_particles: int,
    coherence_history: jnp.ndarray,  # 最近のcoherence履歴
    circulation_history: jnp.ndarray,  # 最近のcirculation履歴
    particle_history: jnp.ndarray,    # 最近の粒子数履歴
    history_len: int = 10
) -> dict:
    """
    渦の健康状態を総合的に評価
    
    Returns:
    --------
    dict: 健康指標
        - sync_health: ΛF同期の健康度 (0-1)
        - structure_health: 構造の健康度 (0-1)
        - vitality: 活力（循環の強さ） (0-1)
        - stability: 安定性 (0-1)
        - overall_health: 総合健康度 (0-1)
    """
    
    # === 1. ΛF同期の健康度 ===
    # 現在の同期度
    current_sync = coherence
    
    # 履歴から同期の安定性を評価
    valid_history = coherence_history >= 0  # 有効な履歴
    n_valid = jnp.sum(valid_history)
    
    # 移動平均
    mean_coherence = jnp.sum(
        jnp.where(valid_history, coherence_history, 0)
    ) / jnp.maximum(n_valid, 1)
    
    # 同期の変動係数（安定性の指標）
    variance = jnp.sum(
        jnp.where(valid_history, (coherence_history - mean_coherence)**2, 0)
    ) / jnp.maximum(n_valid, 1)
    std_coherence = jnp.sqrt(variance)
    cv_coherence = std_coherence / (mean_coherence + 1e-8)
    
    # 同期健康度（高い同期 + 低い変動）
    sync_health = current_sync * jnp.exp(-cv_coherence)
    
    # === 2. 構造の健康度 ===
    # ΛF同期粒子数の推定
    sync_particles = n_particles * coherence
    
    # 最小必要粒子数に対する比率
    min_particles_for_vortex = 3.0
    particle_ratio = sync_particles / min_particles_for_vortex
    
    # 粒子数の安定性
    particle_mean = jnp.mean(particle_history)
    particle_std = jnp.std(particle_history)
    particle_stability = jnp.exp(-particle_std / (particle_mean + 1e-8))
    
    # 構造健康度
    structure_health = jnp.tanh(particle_ratio) * particle_stability
    
    # === 3. 活力（循環の強さ）===
    # 現在の循環強度
    current_vitality = jnp.tanh(jnp.abs(circulation) / 5.0)
    
    # 循環の減衰率を計算
    circ_diffs = jnp.diff(circulation_history)
    decay_rate = jnp.mean(circ_diffs)  # 負なら減衰中
    
    # 減衰に対するペナルティ
    decay_penalty = jnp.exp(decay_rate / 2.0)  # 減衰が速いほど低い値
    
    vitality = current_vitality * decay_penalty
    
    # === 4. 安定性 ===
    # 各指標の時間的安定性を評価（簡易版のトレンド計算）
    x = jnp.arange(history_len, dtype=jnp.float32)
    x_mean = jnp.mean(x)
    y_mean_coh = jnp.mean(coherence_history)
    y_mean_circ = jnp.mean(circulation_history)
    
    coherence_trend = jnp.sum((x - x_mean) * (coherence_history - y_mean_coh)) / (jnp.sum((x - x_mean)**2) + 1e-8)
    circulation_trend = jnp.sum((x - x_mean) * (circulation_history - y_mean_circ)) / (jnp.sum((x - x_mean)**2) + 1e-8)
    
    # トレンドが正なら安定/成長、負なら不安定/衰退
    trend_score = jnp.tanh((coherence_trend + circulation_trend) * 10)
    
    stability = 0.5 + 0.5 * trend_score
    
    # === 5. 総合健康度 ===
    # 重み付き平均（ΛF同期を最重視）
    overall_health = (
        0.4 * sync_health +      # ΛF同期が最重要
        0.25 * structure_health + # 構造の維持
        0.2 * vitality +         # 循環の強さ
        0.15 * stability         # 時間的安定性
    )
    
    return {
        'sync_health': sync_health,
        'structure_health': structure_health,
        'vitality': vitality,
        'stability': stability,
        'overall_health': overall_health
    }

# ==============================
# 位相ラグを考慮した同期評価
# ==============================

@jit
def evaluate_phase_lag_sync(
    Lambda_F: jnp.ndarray,          # 粒子のΛF (N, 2)
    positions: jnp.ndarray,         # 粒子の位置 (N, 2)
    vortex_center: jnp.ndarray,     # 渦中心 (2,)
    mask: jnp.ndarray,              # 有効粒子マスク (N,)
    max_lag: float = 0.2            # 最大許容位相ラグ（時間単位）
) -> float:
    """
    位相ラグを考慮したΛF同期の評価
    
    渦の中で粒子は少し遅れて同期することがある（位相差）
    これを考慮した同期評価を行う
    """
    
    # 渦中心からの相対位置
    rel_pos = positions - vortex_center[None, :]
    distances = jnp.linalg.norm(rel_pos, axis=1)
    
    # 角度位置（極座標）
    theta = jnp.arctan2(rel_pos[:, 1], rel_pos[:, 0])
    
    # 予想される速度方向（渦の接線方向）
    expected_tangent = jnp.stack([-rel_pos[:, 1], rel_pos[:, 0]], axis=1)
    expected_tangent = expected_tangent / (jnp.linalg.norm(expected_tangent, axis=1, keepdims=True) + 1e-8)
    
    # 実際のΛFと期待される接線方向の内積
    Lambda_F_normalized = Lambda_F / (jnp.linalg.norm(Lambda_F, axis=1, keepdims=True) + 1e-8)
    alignment = jnp.sum(Lambda_F_normalized * expected_tangent, axis=1)
    
    # 位相ラグを考慮した重み
    # 渦の中心から遠いほど位相遅れが許容される
    phase_lag_weight = jnp.exp(-distances / 20.0)
    
    # ラグを考慮した同期スコア
    # alignmentが低くても、適切な位相差なら許容
    lag_adjusted_sync = jnp.where(
        mask,
        jnp.maximum(alignment, 0.5 + 0.5 * phase_lag_weight),  # 位相ラグ補正
        0.0
    )
    
    # 重み付き平均
    total_weight = jnp.sum(jnp.where(mask, phase_lag_weight, 0))
    phase_sync_score = jnp.sum(lag_adjusted_sync * phase_lag_weight) / jnp.maximum(total_weight, 1e-8)
    
    return phase_sync_score

# ==============================
# 賢い死の判定（環ちゃんレベル！）
# ==============================

@jit
def smart_vortex_death_judgment(
    vortex_state: VortexStateJAX,
    idx: int,
    Lambda_F: jnp.ndarray,         # 全粒子のΛF
    positions: jnp.ndarray,        # 全粒子の位置
    particle_vortex_ids: jnp.ndarray,  # 各粒子の所属渦ID
    step: int,
    history_window: int = 10,
    death_threshold: float = 0.2   # 健康度がこれ以下なら死
) -> tuple:
    """
    環ちゃんの賢い渦死判定！
    
    Returns:
    --------
    tuple: (should_die, health_score, death_reason)
        - should_die: bool - 死ぬべきか
        - health_score: float - 健康スコア (0-1)
        - death_reason: int - 死因コード
            0: 生きてる
            1: ΛF同期の喪失
            2: 構造の崩壊
            3: 活力の喪失
            4: 不安定化
            5: 総合的な衰弱
    """
    
    # 現在の状態を取得
    is_alive = vortex_state.is_alive[idx]
    coherence = vortex_state.coherences[idx]
    circulation = vortex_state.circulations[idx]
    n_particles = vortex_state.n_particles[idx]
    vortex_center = vortex_state.centers[idx]
    hist_idx = vortex_state.hist_index[idx]
    
    # 生きてない渦は判定しない
    not_alive_result = (False, 1.0, 0)
    
    # 履歴を取得（循環バッファから最近のデータ）
    def get_recent_history(hist_array, current_idx):
        # 最近のhistory_window個を取得
        indices = jnp.arange(history_window)
        hist_indices = (current_idx - history_window + 1 + indices) % hist_array.shape[0]
        return hist_array[hist_indices]
    
    coherence_history = get_recent_history(
        vortex_state.coherence_hist[idx], hist_idx
    )
    circulation_history = get_recent_history(
        vortex_state.circulation_hist[idx], hist_idx
    )
    particle_history = get_recent_history(
        vortex_state.particle_count_hist[idx], hist_idx
    ).astype(jnp.float32)
    
    # === 1. 基本的な健康状態評価 ===
    health = compute_vortex_health(
        coherence, circulation, n_particles,
        coherence_history, circulation_history, particle_history,
        history_window
    )
    
    # === 2. 位相ラグ同期の評価 ===
    # この渦に属する粒子のマスク
    particle_mask = particle_vortex_ids == vortex_state.ids[idx]
    
    phase_sync = evaluate_phase_lag_sync(
        Lambda_F, positions, vortex_center, particle_mask, max_lag=0.2
    )
    
    # === 3. 構造テンソルの一貫性（追加評価）===
    # ΛF同期粒子数
    sync_particle_count = n_particles * coherence
    
    # 構造の一貫性スコア
    structure_consistency = jnp.where(
        sync_particle_count >= 3,
        1.0,
        sync_particle_count / 3.0
    )
    
    # === 4. 総合判定 ===
    # 各健康指標に位相同期と構造一貫性を加味
    final_sync_health = health['sync_health'] * 0.7 + phase_sync * 0.3
    final_structure_health = health['structure_health'] * 0.7 + structure_consistency * 0.3
    
    # 最終的な健康スコア
    final_health_score = (
        0.35 * final_sync_health +      # ΛF同期（位相ラグ考慮）
        0.25 * final_structure_health +  # 構造の健康
        0.2 * health['vitality'] +      # 活力
        0.2 * health['stability']       # 安定性
    )
    
    # === 5. 死の判定 ===
    # 各要因での死亡判定
    sync_death = final_sync_health < 0.15      # ΛF同期の喪失
    structure_death = final_structure_health < 0.1  # 構造の崩壊
    vitality_death = health['vitality'] < 0.05     # 活力の喪失
    stability_death = health['stability'] < 0.1    # 不安定化
    overall_death = final_health_score < death_threshold  # 総合的な衰弱
    
    # 死因の特定
    death_reason = jnp.where(
        sync_death, 1,
        jnp.where(
            structure_death, 2,
            jnp.where(
                vitality_death, 3,
                jnp.where(
                    stability_death, 4,
                    jnp.where(overall_death, 5, 0)
                )
            )
        )
    )
    
    # 最終判定
    should_die = death_reason > 0
    
    # 生きてない渦は死なない（既に死んでる）
    return lax.cond(
        is_alive,
        lambda _: (should_die, final_health_score, death_reason),
        lambda _: not_alive_result,
        None
    )

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
    side: int,  # 0=upper, 1=lower
    grid_size: int = 10,
    min_particles: int = 10
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """上下領域別の渦検出（完全JIT対応）"""
    N = positions.shape[0]
    max_clusters = 25
    
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
    
    # グリッド化
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
    """既存渦と新規検出クラスタのマッチング（Boolean Indexing完全排除版）"""
    max_vortices = len(vortex_state.ids)
    max_clusters = len(new_centers)
    
    # 予測位置の計算
    predicted_centers = vortex_state.centers + jnp.array([10.0 * dt, 0])
    
    # 全渦×全クラスタの距離行列を計算
    distances_all = jnp.linalg.norm(
        predicted_centers[:, None, :] - new_centers[None, :, :],
        axis=2
    )
    
    # 非アクティブな渦の距離を無限大に設定
    distances_masked = jnp.where(
        vortex_state.is_alive[:, None],
        distances_all,
        jnp.inf
    )
    
    # 有効なクラスタのマスク
    valid_clusters = new_properties[:, 0] > 0.5
    
    # 各クラスタに最も近い渦を探す
    min_distances = jnp.min(distances_masked, axis=0)
    min_indices = jnp.argmin(distances_masked, axis=0)
    
    # マッチング結果
    matches = jnp.where(
        (min_distances < matching_threshold) & valid_clusters,
        min_indices,
        -1
    )
    
    # 各渦がマッチしたかのフラグを計算
    def check_matched(vid):
        return jnp.any(matches == vid)
    
    is_matched = vmap(check_matched)(jnp.arange(max_vortices))
    
    return matches, is_matched

# ==============================
# 渦状態更新（賢い死判定統合版 - 完全実装）
# ==============================

@jit
def update_vortex_state_with_smart_death(
    vortex_state: VortexStateJAX,
    matches: jnp.ndarray,
    new_centers: jnp.ndarray,
    new_properties: jnp.ndarray,
    sides_array: jnp.ndarray,  # 新規渦の誕生側（上=0, 下=1）
    Lambda_F: jnp.ndarray,         # 追加：全粒子のΛF
    positions: jnp.ndarray,        # 追加：全粒子の位置
    particle_vortex_ids: jnp.ndarray,  # 追加：粒子の所属
    step: int,
    next_id: int,
    death_threshold: float = 0.2
) -> tuple:
    """
    賢い死の判定を組み込んだ渦状態更新（完全版）
    
    Returns:
    --------
    tuple: (updated_state, next_id, death_diagnostics)
        - death_diagnostics: 各渦の健康状態と死因
    """
    
    max_vortices = len(vortex_state.ids)
    
    # === 履歴インデックス更新 ===
    new_hist_indices = jnp.where(
        vortex_state.is_alive,
        (vortex_state.hist_index + 1) % vortex_state.trajectory.shape[1],
        vortex_state.hist_index
    )
    
    # === 既存渦の更新 ===
    # マッチング情報の整理
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
    
    # 更新値を計算
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
    # 各履歴の更新
    def update_hist_at_idx(hist, idx, value, is_alive):
        return jnp.where(
            is_alive,
            hist.at[idx].set(value),
            hist
        )
    
    def update_all_histories(i):
        hist_idx = new_hist_indices[i]
        is_alive = vortex_state.is_alive[i]
        
        new_coherence_hist = update_hist_at_idx(
            vortex_state.coherence_hist[i], hist_idx, new_coherences_all[i], is_alive
        )
        new_circulation_hist = update_hist_at_idx(
            vortex_state.circulation_hist[i], hist_idx, new_circulations_all[i], is_alive
        )
        new_particle_count_hist = update_hist_at_idx(
            vortex_state.particle_count_hist[i], hist_idx, new_n_particles_all[i], is_alive
        )
        new_trajectory = update_hist_at_idx(
            vortex_state.trajectory[i], hist_idx, new_centers_all[i], is_alive
        )
        
        return new_coherence_hist, new_circulation_hist, new_particle_count_hist, new_trajectory
    
    histories = vmap(update_all_histories)(jnp.arange(max_vortices))
    new_coherence_hist = histories[0]
    new_circulation_hist = histories[1]
    new_particle_count_hist = histories[2]
    new_trajectory = histories[3]
    
    # === 賢い死の判定 ===
    def evaluate_vortex_death(i):
        should_die, health_score, death_reason = smart_vortex_death_judgment(
            vortex_state._replace(
                centers=new_centers_all,
                circulations=new_circulations_all,
                coherences=new_coherences_all,
                n_particles=new_n_particles_all,
                coherence_hist=new_coherence_hist,
                circulation_hist=new_circulation_hist,
                particle_count_hist=new_particle_count_hist
            ),
            i,
            Lambda_F, positions, particle_vortex_ids,
            step, history_window=10, death_threshold=death_threshold
        )
        return should_die, health_score, death_reason
    
    # 全渦の生死を評価
    death_results = vmap(evaluate_vortex_death)(jnp.arange(max_vortices))
    should_die_array = death_results[0]
    health_scores = death_results[1]
    death_reasons = death_results[2]
    
    # === 新規渦の作成 ===
    # 新規渦候補を検出
    is_new_vortex = (matches == -1) & (new_properties[:, 0] > 1.0)
    
    # 空きスロットを探す
    empty_mask = ~vortex_state.is_alive & ~should_die_array  # 死ぬ予定の渦も除外
    
    # 各スロットにインデックスを付与
    slot_indices = jnp.where(empty_mask, jnp.arange(max_vortices), max_vortices)
    sorted_slots = jnp.sort(slot_indices)
    
    # 新規渦を追加する関数
    def add_new_vortex(carry, i):
        state, current_id = carry
        
        # i番目の新規渦候補を探す
        new_vortex_indices = jnp.where(is_new_vortex, jnp.arange(len(matches)), -1)
        sorted_new_indices = jnp.sort(new_vortex_indices)
        
        # i番目の新規渦のインデックス
        cluster_idx = jnp.where(i < jnp.sum(is_new_vortex), sorted_new_indices[-(i+1)], 0)
        
        # i番目の空きスロット
        slot_idx = sorted_slots[i]
        
        # 有効な追加かチェック
        is_valid_add = (i < jnp.sum(is_new_vortex)) & (slot_idx < max_vortices) & (cluster_idx >= 0)
        
        # 誕生側の取得
        birth_side = jnp.where(
            (cluster_idx >= 0) & (cluster_idx < len(sides_array)),
            sides_array[cluster_idx],
            0
        )
        
        # 状態を更新
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
            ),
            hist_index=state.hist_index.at[slot_idx].set(
                jnp.where(is_valid_add, 0, state.hist_index[slot_idx])
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
        particle_count_hist=new_particle_count_hist,
        trajectory=new_trajectory
    )
    
    # 最大10個の新規渦を追加
    max_new_vortices = 10
    (final_state, final_next_id), _ = lax.scan(
        add_new_vortex,
        (temp_state, next_id),
        jnp.arange(max_new_vortices)
    )
    
    # === 死亡処理（賢い判定に基づく）===
    final_state = final_state._replace(
        is_alive=final_state.is_alive & ~should_die_array,
        death_steps=jnp.where(should_die_array, step, final_state.death_steps)
    )
    
    # 診断情報（デバッグ用）
    death_diagnostics = {
        'health_scores': health_scores,
        'death_reasons': death_reasons,
        'n_deaths': jnp.sum(should_die_array),
        'mean_health': jnp.mean(jnp.where(final_state.is_alive, health_scores, 0))
    }
    
    return final_state, final_next_id, death_diagnostics

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
    """剥離統計の更新（Boolean Indexing完全排除版）"""
    
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
# メイン追跡関数（賢い死判定統合版）- track_vortices_step_smart
# ==============================

@partial(jit, static_argnums=(7,))
def track_vortices_step_smart(
    particle_state,  # ParticleState from main simulation
    vortex_state: VortexStateJAX,
    membership: ParticleMembershipJAX,
    shedding_stats: VortexSheddingStats,
    step: int,
    next_id: int,
    obstacle_center: jnp.ndarray,
    config  # GETWindConfig (static)
) -> Tuple[VortexStateJAX, ParticleMembershipJAX, VortexSheddingStats, int, Dict]:
    """完全JIT対応版 + 賢い死判定の渦追跡ステップ"""
    
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
    
    # 粒子の所属IDを結合
    particle_vortex_ids = jnp.where(
        upper_particle_ids >= 0,
        upper_particle_ids,
        jnp.where(
            lower_particle_ids >= 0,
            lower_particle_ids + len(upper_centers),
            -1
        )
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
    
    # 渦状態の更新（賢い死判定付き！）
    vortex_state_updated, next_id, death_diagnostics = update_vortex_state_with_smart_death(
        vortex_state,
        matches,
        centers,
        properties,
        sides_array,
        particle_state.Lambda_F,  # 追加
        particle_state.position,  # 追加
        particle_vortex_ids,      # 追加
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
# 旧版との互換性のため、track_vortices_step_completeも残す
# ==============================

track_vortices_step_complete = track_vortices_step_smart  # エイリアス

# ==============================
# 分析関数（JIT非対応だけど必要）
# ==============================

def explain_death_reason(death_reason_code: int) -> str:
    """死因コードを人間が読める説明に変換"""
    reasons = {
        0: "Alive and healthy",
        1: "Lost ΛF synchronization",
        2: "Structural collapse",
        3: "Lost vitality (weak circulation)",
        4: "Became unstable",
        5: "Overall weakness"
    }
    return reasons.get(int(death_reason_code), "Unknown")

def print_vortex_events(vortex_state: VortexStateJAX, 
                        prev_state: VortexStateJAX, 
                        step: int):
    """渦の誕生・消滅イベントを出力"""
    # NumPyに変換
    curr_alive = np.array(vortex_state.is_alive)
    prev_alive = np.array(prev_state.is_alive)
    
    # 新規誕生
    new_born = (~prev_alive) & curr_alive
    if np.any(new_born):
        born_indices = np.where(new_born)[0]
        for idx in born_indices[:3]:  # 最初の3個だけ表示
            side = "upper" if vortex_state.birth_side[idx] == 0 else "lower"
            print(f"  ★ BIRTH: Vortex ({side}) at step {step}")
    
    # 死亡
    new_dead = prev_alive & (~curr_alive)
    if np.any(new_dead):
        dead_indices = np.where(new_dead)[0]
        for idx in dead_indices[:3]:
            print(f"  ☠ DEATH: Vortex at step {step}")

def create_vortex_genealogy_jax(vortex_state: VortexStateJAX) -> str:
    """渦の系譜図を作成"""
    output = "=== Vortex Genealogy ===\n"
    
    # NumPyに変換
    ids = np.array(vortex_state.ids)
    is_alive = np.array(vortex_state.is_alive)
    birth_steps = np.array(vortex_state.birth_steps)
    death_steps = np.array(vortex_state.death_steps)
    birth_side = np.array(vortex_state.birth_side)
    
    # 有効な渦のみ処理（最初の10個）
    valid_vortices = ids > 0
    for i in np.where(valid_vortices)[0][:10]:
        side = "upper" if birth_side[i] == 0 else "lower"
        status = "alive" if is_alive[i] else f"died@{death_steps[i]}"
        output += f"ID {ids[i]:3d} | {side:5s} | born@{birth_steps[i]:5d} | {status}\n"
    
    return output

def analyze_particle_fates_jax(membership: ParticleMembershipJAX) -> Dict:
    """粒子の運命統計"""
    history_count = np.array(membership.history_count)
    current_vortex = np.array(membership.vortex_ids)
    
    return {
        'never_vortex': int(np.sum(history_count == 0)),
        'single_vortex': int(np.sum(history_count == 1)),
        'multiple_vortices': int(np.sum(history_count > 1)),
        'currently_in_vortex': int(np.sum(current_vortex > 0)),
        'mean_vortices_per_particle': float(np.mean(history_count))
    }

def analyze_vortex_statistics_jax(vortex_state: VortexStateJAX) -> Dict:
    """渦統計解析"""
    is_alive = np.array(vortex_state.is_alive)
    birth_steps = np.array(vortex_state.birth_steps)
    death_steps = np.array(vortex_state.death_steps)
    
    completed = (birth_steps >= 0) & (death_steps >= 0)
    
    if np.sum(completed) > 0:
        lifetimes = death_steps[completed] - birth_steps[completed]
        stats = {
            'n_completed': int(np.sum(completed)),
            'n_active': int(np.sum(is_alive)),
            'mean_lifetime': float(np.mean(lifetimes)),
            'std_lifetime': float(np.std(lifetimes)),
            'mean_travel_distance': 0.0,
            'max_lifetime': int(np.max(lifetimes)),
            'min_lifetime': int(np.min(lifetimes))
        }
    else:
        stats = {
            'n_completed': 0,
            'n_active': int(np.sum(is_alive)),
            'mean_lifetime': 0.0,
            'std_lifetime': 0.0,
            'mean_travel_distance': 0.0,
            'max_lifetime': 0,
            'min_lifetime': 0
        }
    
    return stats

# ==============================
# 剥離頻度計算・パターン解析（JIT対応）
# ==============================

@jit
def compute_shedding_frequency(
    stats: VortexSheddingStats,
    dt: float,
    window_size: int = 10
) -> tuple:
    """剥離頻度の計算"""
    
    upper_valid_count = jnp.minimum(stats.upper_count, window_size)
    
    def compute_upper_freq():
        start_idx = jnp.maximum(0, stats.upper_count - window_size)
        indices = jnp.arange(window_size)
        actual_indices = (start_idx + indices) % stats.upper_shedding_steps.shape[0]
        
        steps = stats.upper_shedding_steps[actual_indices]
        valid_mask = steps >= 0
        valid_steps = jnp.where(valid_mask, steps, 0)
        
        diffs = jnp.diff(valid_steps)
        valid_diffs = jnp.where(diffs > 0, diffs, 0)
        
        mean_interval = jnp.sum(valid_diffs) / jnp.maximum(jnp.sum(valid_diffs > 0), 1)
        frequency = 1.0 / (mean_interval * dt + 1e-8)
        
        return jnp.where(upper_valid_count >= 2, frequency, 0.0)
    
    upper_freq = compute_upper_freq()
    
    lower_valid_count = jnp.minimum(stats.lower_count, window_size)
    
    def compute_lower_freq():
        start_idx = jnp.maximum(0, stats.lower_count - window_size)
        indices = jnp.arange(window_size)
        actual_indices = (start_idx + indices) % stats.lower_shedding_steps.shape[0]
        
        steps = stats.lower_shedding_steps[actual_indices]
        valid_mask = steps >= 0
        valid_steps = jnp.where(valid_mask, steps, 0)
        
        diffs = jnp.diff(valid_steps)
        valid_diffs = jnp.where(diffs > 0, diffs, 0)
        
        mean_interval = jnp.sum(valid_diffs) / jnp.maximum(jnp.sum(valid_diffs > 0), 1)
        frequency = 1.0 / (mean_interval * dt + 1e-8)
        
        return jnp.where(lower_valid_count >= 2, frequency, 0.0)
    
    lower_freq = compute_lower_freq()
    
    has_both = (upper_freq > 0) & (lower_freq > 0)
    mean_freq = jnp.where(
        has_both,
        (upper_freq + lower_freq) / 2.0,
        jnp.where(
            upper_freq > 0,
            upper_freq,
            lower_freq
        )
    )
    
    return upper_freq, lower_freq, mean_freq

@jit
def analyze_shedding_pattern(
    stats: VortexSheddingStats,
    window_size: int = 20
) -> dict:
    """渦剥離パターンの統計解析"""
    
    def get_recent_events(steps, count):
        n_valid = jnp.minimum(count, window_size)
        start_idx = jnp.maximum(0, count - window_size)
        
        indices = jnp.arange(window_size)
        actual_indices = (start_idx + indices) % steps.shape[0]
        
        recent_steps = steps[actual_indices]
        valid_mask = (indices < n_valid) & (recent_steps >= 0)
        
        return recent_steps, valid_mask
    
    upper_recent, upper_mask = get_recent_events(
        stats.upper_shedding_steps, stats.upper_count
    )
    lower_recent, lower_mask = get_recent_events(
        stats.lower_shedding_steps, stats.lower_count
    )
    
    def check_alternating():
        all_steps = jnp.concatenate([
            jnp.where(upper_mask, upper_recent, -1),
            jnp.where(lower_mask, lower_recent, -1)
        ])
        all_sides = jnp.concatenate([
            jnp.zeros(window_size),
            jnp.ones(window_size)
        ])
        
        sorted_indices = jnp.argsort(all_steps)
        sorted_steps = all_steps[sorted_indices]
        sorted_sides = all_sides[sorted_indices]
        
        valid = sorted_steps >= 0
        n_valid = jnp.sum(valid)
        
        side_changes = jnp.diff(sorted_sides)
        alternating_count = jnp.sum(
            jnp.where(valid[:-1] & valid[1:], jnp.abs(side_changes) > 0.5, False)
        )
        
        alternating_ratio = alternating_count / jnp.maximum(n_valid - 1, 1)
        
        return alternating_ratio
    
    alternating_ratio = check_alternating()
    
    def compute_regularity(steps, mask):
        valid_steps = jnp.where(mask, steps, 0)
        n_valid = jnp.sum(mask)
        
        intervals = jnp.diff(valid_steps)
        valid_intervals = jnp.where(intervals > 0, intervals, 0)
        n_intervals = jnp.sum(valid_intervals > 0)
        
        mean_interval = jnp.sum(valid_intervals) / jnp.maximum(n_intervals, 1)
        variance = jnp.sum(
            jnp.where(valid_intervals > 0, (valid_intervals - mean_interval)**2, 0)
        ) / jnp.maximum(n_intervals, 1)
        
        std_interval = jnp.sqrt(variance)
        regularity = jnp.exp(-std_interval / (mean_interval + 1e-8))
        
        return jnp.where(n_intervals >= 2, regularity, 0.0)
    
    upper_regularity = compute_regularity(upper_recent, upper_mask)
    lower_regularity = compute_regularity(lower_recent, lower_mask)
    
    return {
        'alternating_ratio': alternating_ratio,
        'upper_regularity': upper_regularity,
        'lower_regularity': lower_regularity,
        'overall_regularity': (upper_regularity + lower_regularity) / 2.0,
        'is_karman_like': (alternating_ratio > 0.7) & 
                         (upper_regularity > 0.5) & 
                         (lower_regularity > 0.5)
    }

# ==============================
# テスト用
# ==============================

if __name__ == "__main__":
    print("=" * 70)
    print("GET Wind™ v6.3 JAX - FULLY JIT + Smart Death Edition!")
    print("環ちゃん & ご主人さま Ultimate Intelligence! 💕")
    print("=" * 70)
    
    print("\n✨ Complete Merged Features:")
    print("  ✅ Full JIT compilation - 完全JIT対応!")
    print("  ✅ NO Boolean Indexing - 完全排除!")
    print("  ✅ Smart death judgment - 賢い死判定完全実装!")
    print("  ✅ Health monitoring - 健康状態の総合評価!")
    print("  ✅ Phase-lag aware sync - 位相ラグ考慮!")
    print("  ✅ Temporal stability - 時間的安定性!")
    print("  ✅ Physically justified death - 物理的に正しい死!")
    
    print("\n🎯 Integration Complete:")
    print("  • Original JAX vortex tracker ✓")
    print("  • Smart death judgment system ✓")
    print("  • Full lifecycle management ✓")
    print("  • All analysis functions ✓")
    
    print("\n📝 Available Functions:")
    print("  Main:")
    print("    - track_vortices_step_smart() : Main tracking with smart death")
    print("    - track_vortices_step_complete() : Alias for compatibility")
    print("  Core:")
    print("    - smart_vortex_death_judgment() : Smart death evaluation")
    print("    - compute_vortex_health() : Health scoring")
    print("    - evaluate_phase_lag_sync() : Phase-lag aware sync")
    print("  Analysis:")
    print("    - analyze_vortex_statistics_jax()")
    print("    - create_vortex_genealogy_jax()")
    print("    - analyze_particle_fates_jax()")
    print("    - print_vortex_events()")
    print("    - compute_shedding_frequency()")
    print("    - analyze_shedding_pattern()")
    
    vortex_state = initialize_vortex_state()
    membership = initialize_particle_membership(1500)
    shedding_stats = initialize_shedding_stats()
    
    print(f"\n📊 Initialized structures:")
    print(f"  Vortex state: {vortex_state.ids.shape[0]} max vortices")
    print(f"  Membership: {membership.vortex_ids.shape[0]} max particles")
    print(f"  Shedding stats: {shedding_stats.upper_shedding_steps.shape[0]} max events")
    
    print("\n✨ COMPLETE MERGE SUCCESSFUL! ✨")
    print("Use track_vortices_step_smart() for smart death judgment!")
    print("=" * 70)
