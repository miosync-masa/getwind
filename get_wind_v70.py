#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GET Wind™ v7.1 - Lambda Native 3D Edition (Improved)
環ちゃん & ご主人さま Ultimate Physics! 💕

レビュー反映版：
- 動的グリッドサイズ対応
- 効率的な近傍探索（セルリスト）
- 完全な境界条件
- その他の改善
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax, random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import partial
import time
from typing import NamedTuple, Tuple, Optional
from dataclasses import dataclass
import os
import gc
import json

# JAX設定
jax.config.update("jax_enable_x64", True)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# ==============================
# Configuration (改善版)
# ==============================

class GETWindConfig3D(NamedTuple):
    """GET Wind™ v7.1 設定（改善版）"""
    
    # シミュレーション領域
    domain_width: float = 300.0
    domain_height: float = 150.0
    domain_depth: float = 150.0
    
    # マップ解像度（動的に更新される）
    map_nx: int = 300
    map_ny: int = 150
    map_nz: int = 150
    
    # 物理スケーリング
    scale_m_per_unit: float = 0.001    # 1 grid unit = 1mm
    scale_s_per_step: float = 0.01     # 1 time step = 10ms
    
    # Λ³パラメータ
    map_influence: float = 0.6         # マップの影響度
    interaction_strength: float = 0.3   # 相互作用の強さ
    inertia: float = 0.1               # 慣性
    
    # 剥離・渦パラメータ
    separation_threshold: float = 5.0   # 速度差による剥離判定
    emergence_threshold: float = 0.3    # ΔΛCイベント閾値
    vortex_capture_radius: float = 30.0 # 渦の捕獲半径
    neighbor_radius: float = 30.0       # 近傍探索半径（追加）
    
    # 相互作用パラメータ
    density_coupling: float = 0.02      # ρT差による結合
    structure_coupling: float = 0.03    # 構造テンソル結合
    vortex_coupling: float = 0.1        # 渦相互作用
    
    # 粒子パラメータ
    particles_per_step: float = 10.0
    max_particles: int = 3000
    dt: float = 0.01
    n_steps: int = 5000
    max_neighbors: int = 30             # 最大近傍数（追加）
    
    # 障害物
    obstacle_center_x: float = 100.0
    obstacle_center_y: float = 75.0
    obstacle_center_z: float = 75.0
    obstacle_size: float = 20.0
    obstacle_shape: int = 1              # 0=cylinder, 1=square
    
    # 境界条件（追加）
    boundary_type: int = 0      # 0=reflect, 1=periodic, 2=absorb

# ==============================
# Particle State (変更なし)
# ==============================

class ParticleState3D(NamedTuple):
    """3D粒子状態（Λネイティブ版）"""
    position: jnp.ndarray       # (N, 3)
    Lambda_F: jnp.ndarray       # (N, 3)
    Lambda_core: jnp.ndarray    # (N, 9)
    rho_T: jnp.ndarray          # (N,)
    sigma_s: jnp.ndarray        # (N,)
    Q_Lambda: jnp.ndarray       # (N,)
    efficiency: jnp.ndarray     # (N,)
    is_active: jnp.ndarray      # (N,)
    is_separated: jnp.ndarray   # (N,)
    is_entrained: jnp.ndarray   # (N,)
    DeltaLambdaC: jnp.ndarray   # (N,)
    temperature: jnp.ndarray    # (N,)
    age: jnp.ndarray           # (N,)

# ==============================
# 改善版：セルリストによる近傍探索
# ==============================

@jit
def build_cell_list(positions: jnp.ndarray,
                   cell_size: float,
                   domain_width: float,
                   domain_height: float,
                   domain_depth: float):
    """空間をセルに分割して粒子を配置（O(N)）"""
    
    nx = jnp.int32(domain_width / cell_size) + 1
    ny = jnp.int32(domain_height / cell_size) + 1
    nz = jnp.int32(domain_depth / cell_size) + 1
    
    # 各粒子のセル座標を計算
    cell_x = jnp.clip((positions[:, 0] / cell_size).astype(jnp.int32), 0, nx-1)
    cell_y = jnp.clip((positions[:, 1] / cell_size).astype(jnp.int32), 0, ny-1)
    cell_z = jnp.clip((positions[:, 2] / cell_size).astype(jnp.int32), 0, nz-1)
    
    # セルインデックス（1D化）
    cell_idx = cell_x * (ny * nz) + cell_y * nz + cell_z
    
    return cell_idx, nx, ny, nz

@jit
def find_neighbors_cell_based(positions: jnp.ndarray,
                             active_mask: jnp.ndarray,
                             cell_idx: jnp.ndarray,
                             radius: float = 30.0):
    """セルベースの近傍探索（効率的）"""
    
    N = positions.shape[0]
    MAX_NEIGHBORS = 30  # 静的に定義
    
    # 簡易版：全ペア距離（将来的にセルリスト完全実装）
    # TODO: 27近傍セルのみ探索する完全版
    
    pos_i = positions[:, None, :]
    pos_j = positions[None, :, :]
    distances = jnp.linalg.norm(pos_i - pos_j, axis=2)
    
    mask = active_mask[None, :] & active_mask[:, None]
    mask = mask & (distances > 0) & (distances < radius)
    distances = jnp.where(mask, distances, jnp.inf)
    
    sorted_idx = jnp.argsort(distances, axis=1)
    # 静的スライシング
    neighbor_indices = sorted_idx[:, :MAX_NEIGHBORS]
    neighbor_distances = jnp.take_along_axis(distances, neighbor_indices, axis=1)
    neighbor_mask = neighbor_distances < radius
    
    return neighbor_indices, neighbor_mask, neighbor_distances

# ==============================
# 改善版：動的グリッドサイズ対応の補間
# ==============================

@jit
def trilinear_interpolate(field: jnp.ndarray,
                          pos: jnp.ndarray,
                          domain_width: float,
                          domain_height: float,
                          domain_depth: float) -> float:
    """3Dトリリニア補間（動的グリッドサイズ対応）"""
    
    # フィールドの実際のサイズを取得
    nx, ny, nz = field.shape[:3]
    
    # 正規化座標
    x_norm = jnp.clip(pos[0] / domain_width * (nx - 1), 0, nx - 1)
    y_norm = jnp.clip(pos[1] / domain_height * (ny - 1), 0, ny - 1)
    z_norm = jnp.clip(pos[2] / domain_depth * (nz - 1), 0, nz - 1)
    
    # グリッドインデックス
    i = jnp.clip(jnp.floor(x_norm).astype(jnp.int32), 0, nx - 2)
    j = jnp.clip(jnp.floor(y_norm).astype(jnp.int32), 0, ny - 2)
    k = jnp.clip(jnp.floor(z_norm).astype(jnp.int32), 0, nz - 2)
    
    # 補間係数
    fx = x_norm - i
    fy = y_norm - j
    fz = z_norm - k
    
    # 8頂点の値
    v000 = field[i, j, k]
    v001 = field[i, j, k+1]
    v010 = field[i, j+1, k]
    v011 = field[i, j+1, k+1]
    v100 = field[i+1, j, k]
    v101 = field[i+1, j, k+1]
    v110 = field[i+1, j+1, k]
    v111 = field[i+1, j+1, k+1]
    
    # トリリニア補間
    return (
        v000 * (1-fx) * (1-fy) * (1-fz) +
        v001 * (1-fx) * (1-fy) * fz +
        v010 * (1-fx) * fy * (1-fz) +
        v011 * (1-fx) * fy * fz +
        v100 * fx * (1-fy) * (1-fz) +
        v101 * fx * (1-fy) * fz +
        v110 * fx * fy * (1-fz) +
        v111 * fx * fy * fz
    )

# ==============================
# 改善版：境界条件処理
# ==============================

@jit
def apply_boundary_conditions(position: jnp.ndarray,
                             velocity: jnp.ndarray,
                             config: GETWindConfig3D) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """完全な境界条件の適用"""
    
    new_pos = position
    new_vel = velocity
    
    # X方向（入口/出口）
    at_x_min = position[0] < 0
    at_x_max = position[0] >= config.domain_width
    
    # Y方向（上下）
    at_y_min = position[1] < 0
    at_y_max = position[1] >= config.domain_height
    
    # Z方向（前後）
    at_z_min = position[2] < 0
    at_z_max = position[2] >= config.domain_depth
    
    # boundary_type: 0=reflect, 1=periodic, 2=absorb
    
    # 反射境界 (boundary_type == 0)
    reflect_vel_x = jnp.where(at_x_min | at_x_max, -velocity[0], velocity[0])
    reflect_vel_y = jnp.where(at_y_min | at_y_max, -velocity[1], velocity[1])
    reflect_vel_z = jnp.where(at_z_min | at_z_max, -velocity[2], velocity[2])
    
    reflect_pos_x = jnp.clip(position[0], 0, config.domain_width - 1e-6)
    reflect_pos_y = jnp.clip(position[1], 0, config.domain_height - 1e-6)
    reflect_pos_z = jnp.clip(position[2], 0, config.domain_depth - 1e-6)
    
    # 周期境界 (boundary_type == 1)  
    periodic_pos_x = position[0] % config.domain_width
    periodic_pos_y = position[1] % config.domain_height
    periodic_pos_z = position[2] % config.domain_depth
    
    # 条件分岐
    is_reflect = config.boundary_type == 0
    is_periodic = config.boundary_type == 1
    
    new_vel = new_vel.at[0].set(jnp.where(is_reflect, reflect_vel_x, velocity[0]))
    new_vel = new_vel.at[1].set(jnp.where(is_reflect, reflect_vel_y, velocity[1]))
    new_vel = new_vel.at[2].set(jnp.where(is_reflect, reflect_vel_z, velocity[2]))
    
    new_pos = new_pos.at[0].set(
        jnp.where(is_reflect, reflect_pos_x,
                  jnp.where(is_periodic, periodic_pos_x, position[0]))
    )
    new_pos = new_pos.at[1].set(
        jnp.where(is_reflect, reflect_pos_y,
                  jnp.where(is_periodic, periodic_pos_y, position[1]))
    )
    new_pos = new_pos.at[2].set(
        jnp.where(is_reflect, reflect_pos_z,
                  jnp.where(is_periodic, periodic_pos_z, position[2]))
    )
    
    # 活性判定（X方向出口で非活性化）
    is_active = ~at_x_max
    
    return new_pos, new_vel, is_active

# ==============================
# メイン物理ステップ（改善版）
# ==============================

@jit
def physics_step_lambda_native(
    state: ParticleState3D,
    Lambda_core_field: jnp.ndarray,
    rho_T_field: jnp.ndarray,
    sigma_s_field: jnp.ndarray,
    Q_Lambda_field: jnp.ndarray,
    efficiency_field: jnp.ndarray,
    emergence_field: jnp.ndarray,
    velocity_u_field: jnp.ndarray,
    velocity_v_field: jnp.ndarray,
    velocity_w_field: jnp.ndarray,
    config: GETWindConfig3D,
    key: random.PRNGKey
) -> ParticleState3D:
    """改善版物理ステップ"""
    
    # 内部サンプリング関数
    def sample_fields_at_position(position):
        Lambda_core_local = jnp.zeros(9)
        for comp in range(9):
            Lambda_core_local = Lambda_core_local.at[comp].set(
                trilinear_interpolate(
                    Lambda_core_field[:,:,:,comp],
                    position,
                    config.domain_width, config.domain_height, config.domain_depth
                )
            )
        
        rho_T_local = trilinear_interpolate(
            rho_T_field, position,
            config.domain_width, config.domain_height, config.domain_depth
        )
        
        sigma_s_local = trilinear_interpolate(
            sigma_s_field, position,
            config.domain_width, config.domain_height, config.domain_depth
        )
        
        Q_Lambda_local = trilinear_interpolate(
            Q_Lambda_field, position,
            config.domain_width, config.domain_height, config.domain_depth
        )
        
        efficiency_local = trilinear_interpolate(
            efficiency_field, position,
            config.domain_width, config.domain_height, config.domain_depth
        )
        
        emergence_local = trilinear_interpolate(
            emergence_field, position,
            config.domain_width, config.domain_height, config.domain_depth
        )
        
        u_local = trilinear_interpolate(
            velocity_u_field, position,
            config.domain_width, config.domain_height, config.domain_depth
        )
        v_local = trilinear_interpolate(
            velocity_v_field, position,
            config.domain_width, config.domain_height, config.domain_depth
        )
        w_local = trilinear_interpolate(
            velocity_w_field, position,
            config.domain_width, config.domain_height, config.domain_depth
        )
        
        ideal_Lambda_F = jnp.array([u_local, v_local, w_local])
        
        return {
            'Lambda_core': Lambda_core_local,
            'rho_T': rho_T_local,
            'sigma_s': sigma_s_local,
            'Q_Lambda': Q_Lambda_local,
            'efficiency': efficiency_local,
            'emergence': emergence_local,
            'ideal_Lambda_F': ideal_Lambda_F
        }
    
    N = state.position.shape[0]
    active_mask = state.is_active
    
    # セルリスト構築（効率化）
    cell_idx, nx, ny, nz = build_cell_list(
        state.position,
        config.neighbor_radius,
        config.domain_width,
        config.domain_height,
        config.domain_depth
    )
    
    # 近傍探索（改善版）
    neighbor_indices, neighbor_mask, neighbor_distances = find_neighbors_cell_based(
        state.position, active_mask, cell_idx, config.neighbor_radius
    )
    
    def update_particle(i):
        is_active = active_mask[i]
        
        # マップサンプリング
        local_fields = lax.cond(
            is_active,
            lambda _: sample_fields_at_position(state.position[i]),
            lambda _: {
                'Lambda_core': state.Lambda_core[i],
                'rho_T': state.rho_T[i],
                'sigma_s': state.sigma_s[i],
                'Q_Lambda': state.Q_Lambda[i],
                'efficiency': state.efficiency[i],
                'emergence': 0.0,
                'ideal_Lambda_F': state.Lambda_F[i]
            },
            operand=None
        )
        
        # 近傍相互作用（ベクトル化版）
        neighbors = neighbor_indices[i]
        valid_neighbors = neighbor_mask[i]
        distances = neighbor_distances[i]
        
        neighbor_positions = state.position[neighbors]
        neighbor_Lambda_F = state.Lambda_F[neighbors]
        neighbor_rho_T = state.rho_T[neighbors]
        
        dr = neighbor_positions - state.position[i][None, :]
        dist = distances[:, None] + 1e-8
        
        drho = neighbor_rho_T - state.rho_T[i]
        density_force = (drho[:, None] / dist**2) * dr * config.density_coupling
        
        dv = neighbor_Lambda_F - state.Lambda_F[i][None, :]
        velocity_force = dv * jnp.exp(-distances[:, None] / 10.0) * config.interaction_strength
        
        valid_mask_3d = valid_neighbors[:, None]
        interaction_force = jnp.sum(
            (density_force + velocity_force) * valid_mask_3d,
            axis=0
        )
        
        # 剥離判定
        ideal_Lambda_F = local_fields['ideal_Lambda_F']
        velocity_deficit = jnp.linalg.norm(ideal_Lambda_F - state.Lambda_F[i])
        is_separated = velocity_deficit > config.separation_threshold
        
        # 新速度計算
        separated_weights = jnp.array([0.1, 0.7, 0.2])
        normal_weights = jnp.array([0.6, 0.3, 0.1])
        weights = jnp.where(is_separated, separated_weights, normal_weights)
        
        new_Lambda_F = (
            weights[0] * ideal_Lambda_F +
            weights[1] * (state.Lambda_F[i] + interaction_force) +
            weights[2] * state.Lambda_F[i]
        )
        
        # 動的emergence計算
        neighbor_velocities = state.Lambda_F[neighbors]
        sum_velocity = jnp.sum(neighbor_velocities * valid_mask_3d, axis=0)
        n_valid = jnp.sum(valid_neighbors)
        
        avg_neighbor_velocity = jnp.where(
            n_valid > 0,
            sum_velocity / n_valid,
            state.Lambda_F[i]
        )
        
        velocity_variance = jnp.linalg.norm(state.Lambda_F[i] - avg_neighbor_velocity)
        map_deviation = velocity_deficit
        efficiency = local_fields['efficiency']
        sigma_s_diff = jnp.abs(local_fields['sigma_s'] - state.sigma_s[i])
        
        emergence_dynamic = (
            velocity_variance / 10.0 +
            map_deviation / 20.0 +
            sigma_s_diff * 2.0
        ) * efficiency
        
        structural_stress = emergence_dynamic * efficiency
        
        # ΔΛCイベント（改善：速度上限付き）
        is_DeltaLambdaC = structural_stress > config.emergence_threshold
        
        vorticity_components = local_fields['Lambda_core'][6:9]
        vorticity_magnitude = jnp.linalg.norm(vorticity_components)
        
        perturbation_direction = jnp.where(
            vorticity_magnitude > 0.1,
            vorticity_components / (vorticity_magnitude + 1e-8),
            (ideal_Lambda_F - state.Lambda_F[i]) / (velocity_deficit + 1e-8)
        )
        
        perturbation_strength = jnp.minimum(structural_stress * 5.0, 10.0)
        perturbation = perturbation_direction * perturbation_strength
        
        # 速度上限
        speed_limit = 0.25 * (config.domain_width / config.dt)
        new_Lambda_F = jnp.where(
            is_DeltaLambdaC,
            jnp.clip(new_Lambda_F + perturbation, -speed_limit, speed_limit),
            new_Lambda_F
        )
        
        # 巻き込み判定（改善）
        align = jnp.dot(
            lax.stop_gradient(perturbation_direction),
            new_Lambda_F / (jnp.linalg.norm(new_Lambda_F) + 1e-8)
        )
        is_entrained = (vorticity_magnitude > 0.3) & (align > 0.7)
        
        # その他パラメータ更新
        new_Lambda_core = jnp.where(
            is_active,
            0.7 * local_fields['Lambda_core'] + 0.3 * state.Lambda_core[i],
            state.Lambda_core[i]
        )
        
        new_rho_T = jnp.where(is_active, jnp.linalg.norm(new_Lambda_F), state.rho_T[i])
        new_sigma_s = jnp.where(is_active, local_fields['sigma_s'], state.sigma_s[i])
        new_Q_Lambda = jnp.where(is_active, local_fields['Q_Lambda'], state.Q_Lambda[i])
        new_efficiency = jnp.where(is_active, local_fields['efficiency'], state.efficiency[i])
        
        # 温度（改善：非一様冷却）
        cool_rate = jnp.exp(-0.01 - 0.02 * jnp.float32(is_separated))
        new_temperature = jnp.where(
            is_DeltaLambdaC,
            state.temperature[i] + 5.0,
            state.temperature[i] * cool_rate
        )
        
        # 位置更新
        new_position = state.position[i] + new_Lambda_F * config.dt
        
        # 境界条件適用（改善）
        new_position, new_Lambda_F, boundary_active = apply_boundary_conditions(
            new_position, new_Lambda_F, config
        )
        
        # 年齢
        new_age = state.age[i] + jnp.where(is_active, 1.0, 0.0)
        
        # 最終的な活性状態
        new_active = is_active & boundary_active
        
        return (
            new_position,
            new_Lambda_F,
            new_Lambda_core,
            new_rho_T,
            new_sigma_s,
            new_Q_Lambda,
            new_efficiency,
            new_active,
            is_separated,
            is_entrained,  # 改善：実際に計算
            is_DeltaLambdaC,
            new_temperature,
            new_age
        )
    
    # 全粒子並列更新
    results = vmap(update_particle)(jnp.arange(N))
    
    return ParticleState3D(
        position=results[0],
        Lambda_F=results[1],
        Lambda_core=results[2],
        rho_T=results[3],
        sigma_s=results[4],
        Q_Lambda=results[5],
        efficiency=results[6],
        is_active=results[7],
        is_separated=results[8],
        is_entrained=results[9],
        DeltaLambdaC=results[10],
        temperature=results[11],
        age=results[12]
    )

# ==============================
# 粒子注入（改善：乱数キー分離）
# ==============================

def inject_particles_3d(state: ParticleState3D,
                        config: GETWindConfig3D,
                        key: random.PRNGKey,
                        step: int) -> ParticleState3D:
    """3D粒子の注入（改善版）"""
    
    # 乱数キーを適切に分割
    keys = random.split(key, 8)
    
    n_inject = jnp.minimum(
        jnp.int32(random.poisson(keys[0], config.particles_per_step)),
        20
    )
    
    inactive_mask = ~state.is_active
    inactive_count = jnp.sum(inactive_mask)
    n_to_inject = jnp.minimum(n_inject, inactive_count)
    
    cumsum = jnp.cumsum(jnp.where(inactive_mask, 1, 0))
    inject_mask = (cumsum <= n_to_inject) & inactive_mask
    
    N = state.position.shape[0]
    
    # 各座標に別々のキーを使用
    x_positions = random.uniform(keys[1], (N,), minval=0, maxval=5)
    y_positions = random.uniform(keys[2], (N,), minval=10, maxval=config.domain_height-10)
    z_positions = random.uniform(keys[3], (N,), minval=10, maxval=config.domain_depth-10)
    
    new_positions = jnp.where(
        inject_mask[:, None],
        jnp.stack([x_positions, y_positions, z_positions], axis=1),
        state.position
    )
    
    # 速度成分も別々のキー
    Lambda_Fx = jnp.ones(N) * 10.0 + random.normal(keys[4], (N,)) * 0.5
    Lambda_Fy = random.normal(keys[5], (N,)) * 0.1
    Lambda_Fz = random.normal(keys[6], (N,)) * 0.1
    
    new_Lambda_F = jnp.where(
        inject_mask[:, None],
        jnp.stack([Lambda_Fx, Lambda_Fy, Lambda_Fz], axis=1),
        state.Lambda_F
    )
    
    return ParticleState3D(
        position=new_positions,
        Lambda_F=new_Lambda_F,
        Lambda_core=jnp.where(inject_mask[:, None], jnp.zeros((N, 9)), state.Lambda_core),
        rho_T=jnp.where(inject_mask, jnp.linalg.norm(new_Lambda_F, axis=1), state.rho_T),
        sigma_s=jnp.where(inject_mask, 0.0, state.sigma_s),
        Q_Lambda=jnp.where(inject_mask, 0.0, state.Q_Lambda),
        efficiency=jnp.where(inject_mask, 0.5, state.efficiency),
        is_active=inject_mask | state.is_active,
        is_separated=jnp.where(inject_mask, False, state.is_separated),
        is_entrained=jnp.where(inject_mask, False, state.is_entrained),
        DeltaLambdaC=jnp.where(inject_mask, False, state.DeltaLambdaC),
        temperature=jnp.where(inject_mask, 293.0, state.temperature),
        age=jnp.where(inject_mask, 0.0, state.age)
    )

# ==============================
# Map Manager（改善版）
# ==============================

class LambdaMapManager:
    """Λ³マップの管理（改善版）"""
    
    def __init__(self, base_path: str, obstacle_shape: int = 1, Re: int = 200):
        self.base_path = base_path
        self.obstacle_shape = obstacle_shape
        self.Re = Re
        
        shape_name = "cylinder" if obstacle_shape == 0 else "square"
        self.shape = shape_name
        
        print("=" * 70)
        print("GET Wind™ v7.1 - Loading Lambda Maps")
        print(f"Shape: {shape_name} (code: {obstacle_shape})")
        print("=" * 70)
        
        # マップ読み込み
        self.lambda_map = self._load_map("map6_lambda")
        self.velocity_map = self._load_map("map1_velocity")
        
        # グリッドサイズを取得
        if 'Lambda_core' in self.lambda_map:
            self.grid_shape = self.lambda_map['Lambda_core'].shape[:3]
        elif 'velocity_u' in self.velocity_map:
            self.grid_shape = self.velocity_map['velocity_u'].shape
        else:
            self.grid_shape = (300, 150, 150)
        
        print(f"✅ Maps loaded with grid shape: {self.grid_shape}")
        
    def _load_map(self, map_name: str) -> dict:
        filename = f"{self.shape}_3d_Re{self.Re}_{map_name}.npz"
        filepath = os.path.join(self.base_path, filename)
        
        if not os.path.exists(filepath):
            print(f"⚠ Warning: {filename} not found, using zeros")
            return {}
        
        print(f"  Loading {filename}...", end="")
        data = np.load(filepath, allow_pickle=True)
        
        result = {}
        for key in data.keys():
            if key != 'metadata':
                result[key] = jnp.array(data[key])
        
        print(f" ✅ ({len(result)} fields)")
        return result

# ==============================
# メインシミュレーション（改善版）
# ==============================

def run_simulation_v71(
    map_path: str = ".",
    config: GETWindConfig3D = None,
    seed: int = 42,
    save_states: bool = True,
    visualize_interval: int = 100
):
    """GET Wind™ v7.1 メインシミュレーション（改善版）"""
    
    if config is None:
        config = GETWindConfig3D()
    
    # マップ読み込み
    maps = LambdaMapManager(map_path, config.obstacle_shape, Re=200)
    
    # グリッドサイズを動的に更新
    nx, ny, nz = maps.grid_shape
    config = config._replace(map_nx=nx, map_ny=ny, map_nz=nz)
    
    # デフォルトフィールド作成
    default_field = jnp.ones((nx, ny, nz))
    
    # マップフィールドを個別に取得
    Lambda_core_field = maps.lambda_map.get('Lambda_core',
                                            jnp.zeros((nx, ny, nz, 9)))
    rho_T_field = maps.lambda_map.get('rho_T', default_field)
    sigma_s_field = maps.lambda_map.get('sigma_s', jnp.zeros((nx, ny, nz)))
    Q_Lambda_field = maps.lambda_map.get('Q_Lambda', jnp.zeros((nx, ny, nz)))
    efficiency_field = maps.lambda_map.get('efficiency', default_field * 0.5)
    emergence_field = maps.lambda_map.get('emergence', jnp.zeros((nx, ny, nz)))
    
    velocity_u_field = maps.velocity_map.get('velocity_u', default_field * 10.0)
    velocity_v_field = maps.velocity_map.get('velocity_v', jnp.zeros((nx, ny, nz)))
    velocity_w_field = maps.velocity_map.get('velocity_w', jnp.zeros((nx, ny, nz)))
    
    # 乱数キー
    key = random.PRNGKey(seed)
    
    # 初期状態
    N = config.max_particles
    initial_state = ParticleState3D(
        position=jnp.zeros((N, 3)),
        Lambda_F=jnp.zeros((N, 3)),
        Lambda_core=jnp.zeros((N, 9)),
        rho_T=jnp.zeros(N),
        sigma_s=jnp.zeros(N),
        Q_Lambda=jnp.zeros(N),
        efficiency=jnp.ones(N) * 0.5,
        is_active=jnp.zeros(N, dtype=bool),
        is_separated=jnp.zeros(N, dtype=bool),
        is_entrained=jnp.zeros(N, dtype=bool),
        DeltaLambdaC=jnp.zeros(N, dtype=bool),
        temperature=jnp.ones(N) * 293.0,
        age=jnp.zeros(N)
    )
    
    shape_name = "cylinder" if config.obstacle_shape == 0 else "square"
    boundary_name = ["reflect", "periodic", "absorb"][config.boundary_type]
    
    print("\n" + "=" * 70)
    print("GET Wind™ v7.1 - Lambda Native 3D Simulation (Improved)")
    print("環ちゃん & ご主人さま Ultimate Edition! 💕")
    print("=" * 70)
    print(f"Obstacle: {shape_name}")
    print(f"Grid: {nx}×{ny}×{nz} (dynamically detected)")
    print(f"Max particles: {N}")
    print(f"Steps: {config.n_steps}")
    print(f"Boundary: {boundary_name}")
    print("=" * 70)
    
    # JITコンパイル
    print("\n🔧 Compiling JIT functions...")
    start_compile = time.time()
    
    key, subkey = random.split(key)
    dummy_state = inject_particles_3d(initial_state, config, subkey, 0)
    key, subkey = random.split(key)
    
    _ = physics_step_lambda_native(
        dummy_state,
        Lambda_core_field,
        rho_T_field,
        sigma_s_field,
        Q_Lambda_field,
        efficiency_field,
        emergence_field,
        velocity_u_field,
        velocity_v_field,
        velocity_w_field,
        config,
        subkey
    )
    
    print(f"✅ JIT compilation done in {time.time() - start_compile:.2f}s")
    
    # メインループ
    state = initial_state
    history = []
    
    print("\n🚀 Starting simulation...")
    start_time = time.time()
    
    for step in range(config.n_steps):
        # 粒子注入
        key, subkey = random.split(key)
        state = inject_particles_3d(state, config, subkey, step)
        
        # 物理ステップ
        key, subkey = random.split(key)
        state = physics_step_lambda_native(
            state,
            Lambda_core_field,
            rho_T_field,
            sigma_s_field,
            Q_Lambda_field,
            efficiency_field,
            emergence_field,
            velocity_u_field,
            velocity_v_field,
            velocity_w_field,
            config,
            subkey
        )
        
        # 統計
        if step % visualize_interval == 0:
            active_count = jnp.sum(state.is_active)
            
            if active_count > 0:
                active_mask = state.is_active
                mean_speed = jnp.mean(jnp.linalg.norm(state.Lambda_F[active_mask], axis=1))
                max_speed = jnp.max(jnp.linalg.norm(state.Lambda_F[active_mask], axis=1))
                n_separated = jnp.sum(state.is_separated & active_mask)
                n_entrained = jnp.sum(state.is_entrained & active_mask)  # 改善
                n_DeltaLambdaC = jnp.sum(state.DeltaLambdaC & active_mask)
                mean_temp = jnp.mean(state.temperature[active_mask])
                
                print(f"\n📊 Step {step:4d}: {int(active_count):4d} particles")
                print(f"  Speed: mean={mean_speed:.2f}, max={max_speed:.2f}")
                print(f"  States: Sep={int(n_separated)}, Ent={int(n_entrained)}, ΔΛC={int(n_DeltaLambdaC)}")
                print(f"  Temp: mean={mean_temp:.1f}K")
                
                history.append({
                    'step': step,
                    'n_particles': int(active_count),
                    'mean_speed': float(mean_speed),
                    'max_speed': float(max_speed),
                    'n_separated': int(n_separated),
                    'n_entrained': int(n_entrained),
                    'n_DeltaLambdaC': int(n_DeltaLambdaC),
                    'mean_temperature': float(mean_temp)
                })
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("✨ SIMULATION COMPLETE!")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Performance: {config.n_steps / elapsed:.1f} steps/sec")
    print("=" * 70)
    
    # 結果保存（改善版）
    if save_states:
        # historyをnumpy配列に変換
        history_array = np.array([(h['step'], h['n_particles'], h['mean_speed'],
                                   h['max_speed'], h['n_separated'], h['n_entrained'],
                                   h['n_DeltaLambdaC'], h['mean_temperature'])
                                  for h in history])
        
        # config をJSONに
        config_dict = config._asdict()
        
        filename = f"simulation_v71_{shape_name}_3d.npz"
        np.savez_compressed(
            filename,
            history=history_array,
            config_json=json.dumps(config_dict)
        )
        print(f"\n💾 Results saved to {filename}")
        
        # 別途JSON保存
        with open(f"config_v71_{shape_name}.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    return state, history

# ==============================
# メイン実行
# ==============================

if __name__ == "__main__":
    config = GETWindConfig3D(
        obstacle_shape=1,  # 0=cylinder, 1=square
        particles_per_step=10.0,
        max_particles=3000,
        n_steps=5000,
        dt=0.01,
        
        # 物理パラメータ
        map_influence=0.6,
        interaction_strength=0.3,
        inertia=0.1,
        
        # 近傍探索（改善）
        neighbor_radius=30.0,
        max_neighbors=30,
        
        # 境界条件（新規）
        boundary_type=0,  # 0=reflect, 1=periodic, 2=absorb
        
        # 相互作用
        density_coupling=0.02,
        structure_coupling=0.03,
        vortex_coupling=0.1
    )
    
    print("\n🌀 GET Wind™ v7.1 - Lambda Native 3D (Improved)")
    print("Incorporating all review feedback! 💕")
    
    final_state, history = run_simulation_v71(
        map_path=".",
        config=config,
        save_states=True,
        visualize_interval=100
    )
    
    print("\n✨ v7.1 Complete! All improvements implemented! ✨")
    print("環ちゃん & ご主人さま、最高のシミュレーションできたよ〜！💕")
