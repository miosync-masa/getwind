#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GET Wind™ v6.3 JAX Edition - Geometric Bernoulli Map Integration [FIXED]
環ちゃん & ご主人さま Ultimate Geometry Edition! 💕

v6.3 Fixed Edition:
  - 座標変換とグリッド補間の修正
  - 物理勾配計算の正確化
  - compute_efficiencyの行列射影修正
  - ΛF更新の一本化
  - lax.condによる分岐最適化
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax, random
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import time
from typing import NamedTuple, Tuple, Dict

# JAX設定
jax.config.update("jax_enable_x64", True)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# 形状定数
SHAPE_CYLINDER = 0
SHAPE_SQUARE = 1

# ==============================
# Configuration
# ==============================

from typing import NamedTuple

class GETWindConfig(NamedTuple):
    """GET Wind™ v6.3 設定（幾何MAP対応・物理単位版）"""
    # 障害物形状
    obstacle_shape: int = 0  # 0=cylinder, 1=square
    
    # シミュレーション領域
    domain_width: float = 300.0
    domain_height: float = 150.0
    
    # マップ解像度
    map_nx: int = 300
    map_ny: int = 150
    
    # ✨ 物理スケーリング（NEW!）
    scale_m_per_unit: float = 0.001    # 1 grid unit = 1mm = 0.001m
    scale_s_per_step: float = 0.01     # 1 time step = 0.01s = 10ms
    
    # Λ³パラメータ（コアは維持！）
    Lambda_F_inlet: float = 10.0       # [unit/step] → 1.0 m/s with default scaling
    thermal_alpha: float = 0.008       # 温度勾配の重み
    density_beta: float = 0.015        # 密度勾配の重み
    structure_coupling: float = 0.025  # 構造結合強度
    viscosity_factor: float = 40.0     # 粘性係数
    interaction_strength: float = 0.1  # 粒子間相互作用強度
    
    # 効率パラメータ
    efficiency_threshold: float = 0.1
    efficiency_weight: float = 0.5
    
    # トポロジカルパラメータ
    topological_threshold: float = 0.1
    sync_threshold: float = 0.05
    
    # 渦検出パラメータ
    coherence_threshold: float = 0.6
    circulation_threshold: float = 1.0
    min_particles_per_region: int = 20
    vortex_grid_size: float = 10.0
    
    # 幾何MAP用パラメータ
    shear_instability_strength: float = 1.0   # せん断層不安定性
    vortex_formation_noise: float = 1.5       # 渦形成領域の乱流強度
    wake_turbulence_factor: float = 0.8       # 後流乱流係数
    
    # 粒子パラメータ
    particles_per_step: float = 5.0
    max_particles: int = 1500
    dt: float = 0.05                   # [step] → 0.5ms with default scaling
    n_steps: int = 3000
    
    # 物理定数
    obstacle_center_x: float = 100.0   # [unit] → 100mm with default scaling
    obstacle_center_y: float = 75.0    # [unit] → 75mm with default scaling
    obstacle_size: float = 20.0        # [unit] radius → 20mm with default scaling
    
    # === 物理量計算用プロパティ ===
    
    @property
    def domain_width_m(self) -> float:
        """領域幅 [m]"""
        return self.domain_width * self.scale_m_per_unit
    
    @property
    def domain_height_m(self) -> float:
        """領域高さ [m]"""
        return self.domain_height * self.scale_m_per_unit
    
    @property
    def obstacle_diameter_m(self) -> float:
        """障害物直径 [m]"""
        return 2 * self.obstacle_size * self.scale_m_per_unit
    
    @property
    def obstacle_radius_m(self) -> float:
        """障害物半径 [m]"""
        return self.obstacle_size * self.scale_m_per_unit
    
    @property
    def dt_s(self) -> float:
        """時間ステップ [s]"""
        return self.dt * self.scale_s_per_step
    
    @property
    def inlet_velocity_ms(self) -> float:
        """入口流速 [m/s]"""
        return self.Lambda_F_inlet * self.scale_m_per_unit / self.scale_s_per_step
    
    @property
    def Reynolds_number(self) -> float:
        """物理的Reynolds数 (空気 @ 20°C)"""
        D = self.obstacle_diameter_m
        U = self.inlet_velocity_ms
        nu = 1.5e-5  # 空気の動粘性係数 [m²/s] @ 20°C
        return U * D / nu
    
    @property
    def blockage_ratio(self) -> float:
        """ブロッケージ比（障害物直径/領域高さ）"""
        return self.obstacle_diameter_m / self.domain_height_m
    
    @property
    def expected_Strouhal(self) -> float:
        """期待されるStrouhal数（Re依存）"""
        Re = self.Reynolds_number
        if Re < 100:
            return 0.16
        elif Re < 250:
            return 0.195  # カルマン渦の典型値
        elif Re < 1000:
            return 0.20
        else:
            return 0.21
    
    @property
    def expected_shedding_frequency(self) -> float:
        """期待される渦放出周波数 [Hz]"""
        St = self.expected_Strouhal
        U = self.inlet_velocity_ms
        D = self.obstacle_diameter_m
        return St * U / D
    
    @property
    def simulation_time_s(self) -> float:
        """総シミュレーション時間 [s]"""
        return self.n_steps * self.dt_s
    
    @property
    def vortex_shedding_periods(self) -> float:
        """シミュレーション中の渦放出周期数"""
        f = self.expected_shedding_frequency
        if f > 0:
            return self.simulation_time_s * f
        return 0.0
    
    def print_physical_summary(self):
        """物理パラメータのサマリーを表示"""
        print("=" * 70)
        print("GET Wind™ v6.3 - Physical Parameters Summary")
        print("=" * 70)
        
        print("\n📏 SCALES:")
        print(f"  Length scale: {self.scale_m_per_unit*1000:.3f} mm/unit")
        print(f"  Time scale: {self.scale_s_per_step*1000:.1f} ms/step")
        
        print("\n🏛️ GEOMETRY:")
        print(f"  Domain: {self.domain_width_m*1000:.0f} × {self.domain_height_m*1000:.0f} mm")
        print(f"  Obstacle: {self.obstacle_diameter_m*1000:.1f} mm diameter")
        print(f"  Center: ({self.obstacle_center_x*self.scale_m_per_unit*1000:.0f}, "
              f"{self.obstacle_center_y*self.scale_m_per_unit*1000:.0f}) mm")
        print(f"  Blockage: {self.blockage_ratio:.1%}")
        
        print("\n💨 FLOW:")
        print(f"  Inlet velocity: {self.inlet_velocity_ms:.3f} m/s")
        print(f"  Reynolds number: {self.Reynolds_number:.0f}")
        print(f"  Expected Strouhal: {self.expected_Strouhal:.3f}")
        print(f"  Expected shedding freq: {self.expected_shedding_frequency:.3f} Hz")
        
        print("\n⏱️ TIME:")
        print(f"  Time step: {self.dt_s*1000:.1f} ms")
        print(f"  Total steps: {self.n_steps}")
        print(f"  Simulation time: {self.simulation_time_s:.2f} s")
        print(f"  Vortex periods: ~{self.vortex_shedding_periods:.0f}")
        
        print("\n🎯 VALIDATION:")
        if self.blockage_ratio > 0.2:
            print(f"  ⚠ High blockage ({self.blockage_ratio:.1%}) may affect results")
        if self.vortex_shedding_periods < 10:
            print(f"  ⚠ Short simulation ({self.vortex_shedding_periods:.1f} periods)")
        if 150 < self.Reynolds_number < 250:
            print(f"  ✅ Reynolds in optimal range for vortex shedding")
        
        print("=" * 70)

# ==============================
# Map Manager (v6.3: 幾何MAP対応)
# ==============================

class GeometricMapData:
    """幾何学的ベルヌーイマップデータの管理"""
    
    def __init__(self, npz_file: str):
        """NPZファイルから幾何MAPを読み込み"""
        print(f"Loading geometric map from {npz_file}...")
        data = np.load(npz_file)
        
        # 幾何学的フィールド
        self.stream_function = jnp.array(data['stream_function'])
        self.pressure = jnp.array(data['pressure'])
        self.density = jnp.array(data['density'])
        self.separation = jnp.array(data['separation'])
        self.vortex_formation = jnp.array(data['vortex_formation'])
        self.wake_structure = jnp.array(data['wake_structure'])
        self.shear_layer = jnp.array(data['shear_layer'])
        
        # グリッド情報（環の修正：shape順序を明確化）
        self.nx, self.ny = self.pressure.shape  # (cols, rows)
        
        print(f"Geometric map loaded: {self.nx}x{self.ny} (cols x rows)")
        print(f"  Available fields: stream_function, pressure, density,")
        print(f"                   separation, vortex_formation,")
        print(f"                   wake_structure, shear_layer")

# ==============================
# Particle State（変更なし）
# ==============================

class ParticleState(NamedTuple):
    """粒子状態（v6.3: 変更なし）"""
    # 基本状態
    position: jnp.ndarray       # (N, 2) 位置
    Lambda_F: jnp.ndarray       # (N, 2) 進行ベクトル
    Lambda_FF: jnp.ndarray      # (N, 2) 加速度
    prev_Lambda_F: jnp.ndarray  # (N, 2) 前の進行ベクトル
    
    # Λ³構造テンソル
    Lambda_core: jnp.ndarray    # (N, 4) 速度勾配テンソル（2x2を平坦化）
    rho_T: jnp.ndarray          # (N,) テンション密度
    sigma_s: jnp.ndarray        # (N,) 同期率
    prev_sigma_s: jnp.ndarray   # (N,) 前の同期率
    Q_Lambda: jnp.ndarray       # (N,) トポロジカル不変量
    prev_Q_Lambda: jnp.ndarray  # (N,) 前のトポロジカル不変量
    
    # 効率と評価
    efficiency: jnp.ndarray     # (N,) 構造効率
    emergence: jnp.ndarray      # (N,) 創発度
    
    # 物理量
    temperature: jnp.ndarray    # (N,) 温度
    density: jnp.ndarray        # (N,) 密度
    vorticity: jnp.ndarray      # (N,) 渦度
    Q_criterion: jnp.ndarray    # (N,) Q判定基準
    
    # イベント検出
    DeltaLambdaC: jnp.ndarray   # (N,) ΔΛCイベント
    event_score: jnp.ndarray    # (N,) イベントスコア
    
    # 管理用
    age: jnp.ndarray           # (N,) 年齢
    is_active: jnp.ndarray     # (N,) アクティブフラグ
    is_separated: jnp.ndarray   # (N,) 剥離フラグ
    near_wall: jnp.ndarray      # (N,) 壁近傍フラグ

# ==============================
# 補間処理（環の修正版：座標変換と行列順序対応）
# ==============================

@jit
def world_to_grid(x: float, y: float, 
                  domain_w: float, domain_h: float, 
                  nx: int, ny: int) -> Tuple[float, float]:
    """物理座標→グリッド座標変換"""
    gx = (x / (domain_w + 1e-8)) * (nx - 1)
    gy = (y / (domain_h + 1e-8)) * (ny - 1)
    gx = jnp.clip(gx, 0.0, nx - 1.0)
    gy = jnp.clip(gy, 0.0, ny - 1.0)
    return gx, gy

@jit
def bilinear_interpolate_rc(field: jnp.ndarray, 
                            gx: float, gy: float, 
                            nx: int, ny: int) -> float:
    """バイリニア補間（row-column順序対応）"""
    # gx, gy は [0, nx-1], [0, ny-1] のグリッド座標
    j = jnp.clip(jnp.floor(gx).astype(int), 0, nx - 2)  # col (=x)
    i = jnp.clip(jnp.floor(gy).astype(int), 0, ny - 2)  # row (=y)
    
    fx = gx - j
    fy = gy - i
    
    # field[row, col] = field[y, x] の順序に注意
    v00 = field[j, i]      # field[x, y]
    v10 = field[j + 1, i]
    v01 = field[j, i + 1]
    v11 = field[j + 1, i + 1]
    
    return (1-fx)*(1-fy)*v00 + fx*(1-fy)*v10 + (1-fx)*fy*v01 + fx*fy*v11

@jit
def gradient_from_map_rc(field: jnp.ndarray, 
                        x: float, y: float,
                        domain_w: float, domain_h: float, 
                        nx: int, ny: int) -> jnp.ndarray:
    """物理勾配計算（正確なスケーリング付き）"""
    dx = domain_w / (nx - 1)
    dy = domain_h / (ny - 1)
    
    gx, gy = world_to_grid(x, y, domain_w, domain_h, nx, ny)
    
    # 中心差分を格子空間で
    val_xp = bilinear_interpolate_rc(field, 
                                     jnp.minimum(gx + 1.0, nx - 1.0), gy, 
                                     nx, ny)
    val_xm = bilinear_interpolate_rc(field, 
                                     jnp.maximum(gx - 1.0, 0.0), gy, 
                                     nx, ny)
    val_yp = bilinear_interpolate_rc(field, 
                                     gx, jnp.minimum(gy + 1.0, ny - 1.0), 
                                     nx, ny)
    val_ym = bilinear_interpolate_rc(field, 
                                     gx, jnp.maximum(gy - 1.0, 0.0), 
                                     nx, ny)
    
    dfdx = (val_xp - val_xm) / (2.0 * dx)
    dfdy = (val_yp - val_ym) / (2.0 * dy)
    
    return jnp.array([dfdx, dfdy])

# ==============================
# Λ³構造テンソル計算（環の修正版：効率計算改良）
# ==============================

@jit
def compute_Lambda_gradient(Lambda_F_i: jnp.ndarray, pos_i: jnp.ndarray,
                           neighbor_Lambda_F: jnp.ndarray,
                           neighbor_positions: jnp.ndarray,
                           neighbor_mask: jnp.ndarray) -> jnp.ndarray:
    """ΛF勾配テンソル（Lambda_core）"""
    dr = neighbor_positions - pos_i
    dLambda = neighbor_Lambda_F - Lambda_F_i
    
    valid = neighbor_mask & (jnp.linalg.norm(dr, axis=1) > 0.01)
    
    # 最小二乗法で勾配を推定
    A = jnp.where(valid[:, None], dr, 0)
    b_u = jnp.where(valid, dLambda[:, 0], 0)
    b_v = jnp.where(valid, dLambda[:, 1], 0)
    
    ATA = A.T @ A + jnp.eye(2) * 1e-8
    grad_u = jnp.linalg.solve(ATA, A.T @ b_u)
    grad_v = jnp.linalg.solve(ATA, A.T @ b_v)
    
    return jnp.array([[grad_u[0], grad_u[1]], [grad_v[0], grad_v[1]]])

@jit
def compute_vortex_quantities(grad_Lambda: jnp.ndarray) -> Tuple[float, float, float]:
    """渦量計算（S, Ω, Q, λ2）"""
    S = 0.5 * (grad_Lambda + grad_Lambda.T)  # 歪み速度テンソル
    Omega = 0.5 * (grad_Lambda - grad_Lambda.T)  # 渦度テンソル
    
    # Q判定基準
    Q = 0.5 * (jnp.trace(Omega @ Omega.T) - jnp.trace(S @ S.T))
    
    # λ2基準
    S2_Omega2 = S @ S + Omega @ Omega
    eigenvalues = jnp.linalg.eigvalsh(S2_Omega2)
    lambda2 = eigenvalues[0]
    
    # 渦度
    vorticity = grad_Lambda[1, 0] - grad_Lambda[0, 1]
    
    return Q, lambda2, vorticity

@jit
def compute_efficiency(Lambda_core_flat: jnp.ndarray, Lambda_F: jnp.ndarray) -> float:
    """構造の効率計算（環の修正版：行列-ベクトル射影）"""
    G = Lambda_core_flat.reshape(2, 2)
    v = Lambda_F
    vn = jnp.linalg.norm(v) + 1e-8
    
    # Gがvをどれだけ伸縮/回転させるかのv方向成分
    Gv = G @ v
    proj_mag = jnp.dot(Gv, v) / vn
    
    # 構造の一貫性
    coherence = jnp.exp(-jnp.var(G))
    
    return jnp.abs(proj_mag) * coherence

@jit
def compute_sigma_s(rho_T_i: float, Lambda_F_i: jnp.ndarray,
                   neighbor_rho_T: jnp.ndarray,
                   neighbor_positions: jnp.ndarray, pos_i: jnp.ndarray,
                   neighbor_mask: jnp.ndarray) -> float:
    """同期率σₛの計算"""
    dr = neighbor_positions - pos_i
    distances = jnp.linalg.norm(dr, axis=1) + 1e-8
    
    valid = neighbor_mask & (distances < 10.0)
    
    # テンション密度の勾配
    drho = neighbor_rho_T - rho_T_i
    grad_rho_T = jnp.sum(
        jnp.where(valid[:, None], (drho[:, None] / distances[:, None]**2) * dr, 0),
        axis=0
    ) / jnp.maximum(jnp.sum(valid), 1)
    
    # ΛFとの同期
    numerator = jnp.dot(grad_rho_T, Lambda_F_i)
    denominator = jnp.linalg.norm(grad_rho_T) * jnp.linalg.norm(Lambda_F_i) + 1e-8
    
    return numerator / denominator

@jit
def compute_local_Q_Lambda(Lambda_F_i: jnp.ndarray, pos_i: jnp.ndarray,
                          neighbor_Lambda_F: jnp.ndarray,
                          neighbor_positions: jnp.ndarray,
                          neighbor_mask: jnp.ndarray) -> float:
    """局所トポロジカルチャージQ_Λ（循環の計算）"""
    valid = neighbor_mask & (jnp.linalg.norm(neighbor_positions - pos_i, axis=1) < 10.0)
    
    rel_pos = neighbor_positions - pos_i
    
    def compute_contribution(idx):
        is_valid = valid[idx]
        Lambda_vec = neighbor_Lambda_F[idx]
        r_vec = rel_pos[idx]
        r_norm = jnp.linalg.norm(r_vec) + 1e-8
        
        # 接線方向（反時計回り）
        tangent = jnp.array([-r_vec[1], r_vec[0]]) / r_norm
        
        # 循環への寄与（速度と接線の内積）
        circulation_contrib = jnp.dot(Lambda_vec, tangent)
        
        # 角度の重み（近い粒子ほど重要）
        weight = jnp.exp(-r_norm / 5.0)
        
        return jnp.where(is_valid, circulation_contrib * weight, 0.0)
    
    # 重み付き循環
    weighted_circulation = jnp.sum(vmap(compute_contribution)(jnp.arange(len(neighbor_positions))))
    
    # 重みの合計で正規化
    total_weight = jnp.sum(jnp.where(valid, jnp.exp(-jnp.linalg.norm(rel_pos, axis=1) / 5.0), 0.0))
    
    # 循環を正規化（-π〜πの範囲）
    normalized_circulation = jnp.where(
        total_weight > 0.1,
        weighted_circulation / (total_weight + 1e-8),
        0.0
    )
    
    # 角度として返す（-π〜π）
    return jnp.clip(normalized_circulation, -jnp.pi, jnp.pi)

# ==============================
# ΔΛC検出（変更なし）
# ==============================

@jit
def detect_DeltaLambdaC(efficiency: float, prev_efficiency: float,
                       sigma_s: float, prev_sigma_s: float,
                       Q_Lambda: float, prev_Q_Lambda: float,
                       Q: float, lambda2: float, vorticity: float,
                       config: GETWindConfig) -> Tuple[bool, float]:
    """ΔΛC検出（構造変化点）"""
    score = 0.0
    
    # 効率の急変
    eff_change = jnp.abs(efficiency - prev_efficiency) / (jnp.abs(prev_efficiency) + 1e-8)
    score += jnp.where(eff_change > 0.5, 2.0, 0.0)
    
    # 同期率の急変
    sigma_jump = jnp.abs(sigma_s - prev_sigma_s)
    score += jnp.where(sigma_jump > config.sync_threshold, 1.5, 0.0)
    
    # トポロジカルジャンプ
    Q_jump = jnp.abs(Q_Lambda - prev_Q_Lambda)
    score += jnp.where(Q_jump > config.topological_threshold, 2.0, 0.0)
    
    # 渦判定
    score += jnp.where(Q > 0.1, 1.0, 0.0)
    score += jnp.where(lambda2 < -0.01, 1.0, 0.0)
    score += jnp.where(jnp.abs(vorticity) > 0.5, 1.0, 0.0)
    
    # ΔΛCイベント判定
    is_event = score >= 2.0
    
    return is_event, score

# ==============================
# 構造間相互作用（変更なし！重要！）
# ==============================

@jit
def compute_structure_interaction(Lambda_F_i: jnp.ndarray, pos_i: jnp.ndarray,
                                 Lambda_core_i: jnp.ndarray,
                                 rho_T_i: float, sigma_s_i: float,
                                 neighbor_Lambda_F: jnp.ndarray,
                                 neighbor_positions: jnp.ndarray,
                                 neighbor_Lambda_core: jnp.ndarray,
                                 neighbor_rho_T: jnp.ndarray,
                                 neighbor_sigma_s: jnp.ndarray,
                                 neighbor_mask: jnp.ndarray,
                                 config: GETWindConfig) -> jnp.ndarray:
    """構造間相互作用（Λ³ Enhanced + 段階的減衰）"""
    
    # 障害物からの距離
    distance_from_obstacle_x = pos_i[0] - config.obstacle_center_x
    
    # 段階的減衰
    decay_factor = jnp.where(
        distance_from_obstacle_x < 100.0,
        1.0,
        jnp.where(
            distance_from_obstacle_x < 200.0,
            jnp.exp(-(distance_from_obstacle_x - 100.0) / 50.0),
            0.0
        )
    )
    
    dr = neighbor_positions - pos_i
    distances = jnp.linalg.norm(dr, axis=1) + 1e-8
    
    # 相互作用範囲
    near_range = neighbor_mask & (distances < 15.0)
    far_range = neighbor_mask & (distances < 30.0)
    
    # 近傍粒子も減衰を考慮
    neighbor_decay = jnp.where(
        (neighbor_positions[:, 0] - config.obstacle_center_x) < 100.0,
        1.0,
        jnp.where(
            (neighbor_positions[:, 0] - config.obstacle_center_x) < 200.0,
            jnp.exp(-((neighbor_positions[:, 0] - config.obstacle_center_x) - 100.0) / 50.0),
            0.0
        )
    )
    
    # 平均減衰率を計算
    combined_decay = jnp.sqrt(decay_factor * neighbor_decay)
    
    # === 1. テンション密度の勾配による力 ===
    drho = neighbor_rho_T - rho_T_i
    grad_rho_force = jnp.sum(
        jnp.where(near_range[:, None], 
                  (drho[:, None] / distances[:, None]**2) * dr * config.density_beta * combined_decay[:, None],
                  0),
        axis=0
    )
    
    # === 2. 構造テンソルの差による力 ===
    Lambda_core_2x2 = Lambda_core_i.reshape(2, 2)
    
    def compute_tensor_force(idx):
        neighbor_core_2x2 = neighbor_Lambda_core[idx].reshape(2, 2)
        
        # テンソル差のノルム
        tensor_diff = neighbor_core_2x2 - Lambda_core_2x2
        diff_norm = jnp.linalg.norm(tensor_diff, 'fro')
        
        # 構造の不一致による反発/引力
        direction = dr[idx] / distances[idx]
        force_mag = diff_norm * jnp.exp(-distances[idx] / 15.0)
        
        # 同期率で重み付け
        sync_weight = 1.0 + (neighbor_sigma_s[idx] - sigma_s_i)
        
        force = direction * force_mag * sync_weight * config.structure_coupling
        
        return jnp.where(near_range[idx], force, jnp.zeros(2))
    
    tensor_forces = vmap(compute_tensor_force)(jnp.arange(len(neighbor_positions)))
    tensor_force = jnp.sum(tensor_forces, axis=0)
    
    # === 3. 渦的相互作用 ===
    vorticity_i = Lambda_core_2x2[1, 0] - Lambda_core_2x2[0, 1]
    
    # 3a. 基本的な渦の回転力（近距離）
    tangent = jnp.stack([-dr[:, 1], dr[:, 0]], axis=1) / distances[:, None]
    
    vortex_rotation = jnp.sum(
        jnp.where(
            near_range[:, None],
            tangent * vorticity_i * jnp.exp(-distances[:, None] / 15.0) * 0.2,
            0
        ),
        axis=0
    )
    
    # 3b. 同回転渦の結合力（遠距離まで作用）
    def compute_vortex_merging(idx):
        # 近傍の渦度
        neighbor_vorticity = neighbor_Lambda_core[idx].reshape(2, 2)[1, 0] - \
                           neighbor_Lambda_core[idx].reshape(2, 2)[0, 1]
        
        # 同じ回転方向かチェック
        same_rotation = vorticity_i * neighbor_vorticity > 0
        
        # 渦度の強さに比例した引力（同回転のみ）
        attraction = jnp.abs(neighbor_vorticity * vorticity_i) * same_rotation
        
        # 距離に応じた減衰
        r = distances[idx]
        force_mag = attraction * jnp.exp(-r / 25.0) * (1 - jnp.exp(-r / 3.0))
        
        # 引力の方向
        direction = dr[idx] / r
        
        return jnp.where(far_range[idx] & same_rotation, direction * force_mag * 0.15, jnp.zeros(2))
    
    vortex_merging = jnp.sum(
        vmap(compute_vortex_merging)(jnp.arange(len(neighbor_positions))),
        axis=0
    )
    
    # 渦力の合計
    vortex_force = vortex_rotation + vortex_merging
    
    # === 4. 粘性的相互作用 ===
    mean_Lambda_F = jnp.sum(
        jnp.where(near_range[:, None], neighbor_Lambda_F, 0),
        axis=0
    ) / jnp.maximum(jnp.sum(near_range), 1)
    
    # 粘性を渦度に応じて調整
    vorticity_factor = jnp.exp(-jnp.abs(vorticity_i) / 2.0)
    effective_viscosity = jnp.minimum(config.viscosity_factor * 0.05 * vorticity_factor, 0.2)
    viscous_force = effective_viscosity * (mean_Lambda_F - Lambda_F_i)
    
    # === 5. 全体の力を合成 ===
    total_interaction = grad_rho_force + tensor_force + vortex_force + viscous_force
    
    # 全体に減衰を適用
    total_interaction = total_interaction * decay_factor
    
    # 相互作用力の大きさを制限
    max_interaction = 5.0
    interaction_norm = jnp.linalg.norm(total_interaction)
    total_interaction = jnp.where(
        interaction_norm > max_interaction,
        total_interaction * max_interaction / interaction_norm,
        total_interaction
    )
    
    return total_interaction

# ==============================
# トポロジカル保存フィードバック（変更なし）
# ==============================

@jit
def apply_topological_feedback(upper_DQ: float, lower_DQ: float,
                              Lambda_F: jnp.ndarray, 
                              efficiency: float,
                              y: float, center_y: float,
                              is_separated: bool,
                              config: GETWindConfig,
                              key: random.PRNGKey) -> jnp.ndarray:
    """トポロジカル保存フィードバック（改良版）"""
    # トポロジカルインバランス
    Q_imbalance = upper_DQ + lower_DQ
    
    # インバランスが大きい場合に補正
    strong_imbalance = jnp.abs(Q_imbalance) > 0.5
    
    y_rel = y - center_y
    
    # 効率が低い場合は補正を強める
    efficiency_factor = jnp.where(efficiency < config.efficiency_threshold, 1.5, 1.0)
    
    # 補正が必要な条件
    should_correct_upper = (
        strong_imbalance & ~is_separated & 
        (Q_imbalance > 0) & (y_rel > 0)
    )
    should_correct_lower = (
        strong_imbalance & ~is_separated & 
        (Q_imbalance < 0) & (y_rel < 0)
    )
    
    # 補正の強さ
    correction_strength = jnp.tanh(jnp.abs(Q_imbalance) / jnp.pi) * efficiency_factor
    
    # 速度の向きを調整
    y_correction = jnp.where(
        should_correct_upper,
        -correction_strength * 2.0,
        jnp.where(
            should_correct_lower,
            correction_strength * 2.0,
            0.0
        )
    )
    
    # ランダムな摂動も追加
    random_factor = random.normal(key, (2,)) * 0.02
    
    new_Lambda_F = Lambda_F + jnp.array([0.0, y_correction]) + random_factor
    
    # 速度の大きさは保存
    original_norm = jnp.linalg.norm(Lambda_F) + 1e-8
    new_norm = jnp.linalg.norm(new_Lambda_F) + 1e-8
    new_Lambda_F = new_Lambda_F * (original_norm / new_norm)
    
    return new_Lambda_F

# ==============================
# 近傍探索（変更なし）
# ==============================

@partial(jit, static_argnums=(2,))
def find_neighbors(positions: jnp.ndarray, active_mask: jnp.ndarray,
                   max_neighbors: int = 20):
    """近傍粒子を探索"""
    N = positions.shape[0]
    
    pos_i = positions[:, None, :]
    pos_j = positions[None, :, :]
    distances = jnp.linalg.norm(pos_i - pos_j, axis=2)
    
    mask = active_mask[None, :] & active_mask[:, None]
    mask = mask & (distances > 0)
    distances = jnp.where(mask, distances, jnp.inf)
    
    sorted_indices = jnp.argsort(distances, axis=1)
    sorted_distances = jnp.sort(distances, axis=1)
    
    neighbor_indices = sorted_indices[:, :max_neighbors]
    neighbor_distances = sorted_distances[:, :max_neighbors]
    neighbor_mask = neighbor_distances < 15.0
    
    return neighbor_indices, neighbor_mask

# ==============================
# メイン物理ステップ（v6.3 Fixed: 環の修正適用）
# ==============================

@partial(jit, static_argnums=(7, 8))
def physics_step_v63(state: ParticleState,
                    pressure_map: jnp.ndarray,
                    density_map: jnp.ndarray,
                    separation_map: jnp.ndarray,
                    vortex_formation_map: jnp.ndarray,
                    wake_structure_map: jnp.ndarray,
                    shear_layer_map: jnp.ndarray,
                    map_nx: int, map_ny: int,
                    config: GETWindConfig,
                    key: random.PRNGKey) -> ParticleState:
    """v6.3の物理ステップ（幾何MAP駆動・修正版）"""
    
    active_mask = state.is_active
    N = state.position.shape[0]
    
    # 近傍探索
    neighbor_indices, neighbor_mask = find_neighbors(state.position, active_mask)
    
    # トポロジカル統計
    y_rel_all = state.position[:, 1] - config.obstacle_center_y
    upper_sep = state.is_separated & active_mask & (y_rel_all > 0)
    lower_sep = state.is_separated & active_mask & (y_rel_all <= 0)
    
    upper_DQ = jnp.sum(jnp.where(upper_sep, state.Q_Lambda, 0.0))
    lower_DQ = jnp.sum(jnp.where(lower_sep, state.Q_Lambda, 0.0))
    
    # 近傍データ準備
    all_neighbor_positions = state.position[neighbor_indices]
    all_neighbor_Lambda_F = state.Lambda_F[neighbor_indices]
    all_neighbor_Lambda_core = state.Lambda_core[neighbor_indices]
    all_neighbor_rho_T = state.rho_T[neighbor_indices]
    all_neighbor_sigma_s = state.sigma_s[neighbor_indices]
    
    def update_particle(i):
        """各粒子の更新"""
        is_active = active_mask[i]
        pos = state.position[i]
        
        # === 1. 座標変換と幾何MAPから場を取得（環の修正） ===
        gx, gy = world_to_grid(pos[0], pos[1], 
                               config.domain_width, config.domain_height, 
                               map_nx, map_ny)
        
        local_pressure = bilinear_interpolate_rc(pressure_map, gx, gy, map_nx, map_ny)
        local_density = bilinear_interpolate_rc(density_map, gx, gy, map_nx, map_ny)
        local_separation = bilinear_interpolate_rc(separation_map, gx, gy, map_nx, map_ny)
        local_vortex_formation = bilinear_interpolate_rc(vortex_formation_map, gx, gy, map_nx, map_ny)
        local_wake = bilinear_interpolate_rc(wake_structure_map, gx, gy, map_nx, map_ny)
        local_shear = bilinear_interpolate_rc(shear_layer_map, gx, gy, map_nx, map_ny)
        
        # 物理勾配（環の修正：物理単位で計算）
        grad_pressure = gradient_from_map_rc(pressure_map, pos[0], pos[1],
                                            config.domain_width, config.domain_height,
                                            map_nx, map_ny)
        grad_density = gradient_from_map_rc(density_map, pos[0], pos[1],
                                           config.domain_width, config.domain_height,
                                           map_nx, map_ny)
        
        # === 2. Λ³構造テンソルの計算（lax.condで最適化） ===
        neighbor_pos = all_neighbor_positions[i]
        neighbor_Lambda_F = all_neighbor_Lambda_F[i]
        neighbor_Lambda_core = all_neighbor_Lambda_core[i]
        neighbor_rho_T = all_neighbor_rho_T[i]
        neighbor_sigma_s = all_neighbor_sigma_s[i]
        neighbor_valid = neighbor_mask[i]
        
        # 重い計算はlax.condで分岐
        grad_Lambda = lax.cond(
            is_active,
            lambda _: compute_Lambda_gradient(
                state.Lambda_F[i], pos,
                neighbor_Lambda_F, neighbor_pos, neighbor_valid
            ),
            lambda _: jnp.eye(2),
            operand=None
        )
        Lambda_core = grad_Lambda.reshape(-1)[:4]
        
        # 渦量計算
        Q_active, lambda2_active, vorticity_active = compute_vortex_quantities(grad_Lambda)
        Q = jnp.where(is_active, Q_active, 0.0)
        lambda2 = jnp.where(is_active, lambda2_active, 0.0)
        vorticity = jnp.where(is_active, vorticity_active, 0.0)
        
        # テンション密度
        rho_T = jnp.where(is_active, jnp.linalg.norm(state.Lambda_F[i]), state.rho_T[i])
        
        # 同期率
        sigma_s = lax.cond(
            is_active,
            lambda _: compute_sigma_s(
                state.rho_T[i], state.Lambda_F[i],
                neighbor_rho_T, neighbor_pos, pos,
                neighbor_valid
            ),
            lambda _: state.sigma_s[i],
            operand=None
        )
        
        # トポロジカル不変量
        Q_Lambda = lax.cond(
            is_active,
            lambda _: compute_local_Q_Lambda(
                state.Lambda_F[i], pos,
                neighbor_Lambda_F, neighbor_pos,
                neighbor_valid
            ),
            lambda _: state.Q_Lambda[i],
            operand=None
        )
        
        # 効率（環の修正版を使用）
        efficiency = jnp.where(
            is_active,
            compute_efficiency(Lambda_core, state.Lambda_F[i]),
            state.efficiency[i]
        )
        
        # === 3. 構造間相互作用 ===
        structure_force = lax.cond(
            is_active,
            lambda _: compute_structure_interaction(
                state.Lambda_F[i], pos, Lambda_core,
                rho_T, sigma_s,
                neighbor_Lambda_F, neighbor_pos,
                neighbor_Lambda_core, neighbor_rho_T, neighbor_sigma_s,
                neighbor_valid, config
            ),
            lambda _: jnp.zeros(2),
            operand=None
        )
        
        # === 4. ΔΛC検出 ===
        is_DeltaLambdaC_active, event_score_active = detect_DeltaLambdaC(
            efficiency, state.efficiency[i],
            sigma_s, state.sigma_s[i],
            Q_Lambda, state.Q_Lambda[i],
            Q, lambda2, vorticity,
            config
        )
        is_DeltaLambdaC = jnp.where(is_active, is_DeltaLambdaC_active, False)
        event_score = jnp.where(is_active, event_score_active, 0.0)
        
        # === 5. ΛF更新（環の修正：一本化） ===
        # 1) 基本的な力
        base_force = -config.thermal_alpha * grad_pressure - config.density_beta * grad_density
        
        # 2) 合成
        new_Lambda_F = state.Lambda_F[i] + base_force + structure_force
        
        # 3) ΔΛCでのみノイズ注入
        subkey = random.fold_in(key, i * 1000)
        DeltaLambdaC_noise = random.normal(subkey, (2,)) * 0.5
        new_Lambda_F = jnp.where(
            is_DeltaLambdaC,
            new_Lambda_F + DeltaLambdaC_noise,
            new_Lambda_F
        )
        
        # === 6. 幾何情報から物理パラメータを再計算 ===
        # 剥離判定
        is_separated = jnp.where(
            local_separation > 0.5,
            True,
            state.is_separated[i]
        )
        
        # トポロジカル保存フィードバック
        feedback_key = random.fold_in(key, i * 6000)
        new_Lambda_F = lax.cond(
            is_active,
            lambda _: apply_topological_feedback(
                upper_DQ, lower_DQ,
                new_Lambda_F, efficiency,
                pos[1], config.obstacle_center_y,
                is_separated, config,
                feedback_key
            ),
            lambda _: state.Lambda_F[i],
            operand=None
        )
        
        # 速度制限
        max_velocity = config.Lambda_F_inlet * 1.5
        new_Lambda_F = jnp.clip(new_Lambda_F, -max_velocity, max_velocity)
        
        # ΛFF（加速度）
        new_Lambda_FF = jnp.where(
            is_active,
            (new_Lambda_F - state.Lambda_F[i]) / config.dt,
            state.Lambda_FF[i]
        )
        
        # === 7. 物理量更新 ===
        temp_noise = random.normal(random.fold_in(key, i * 7000)) * 2.0
        new_temperature = state.temperature[i] + jnp.where(
            is_DeltaLambdaC,
            temp_noise,
            0.0
        )
        new_temperature = jnp.clip(new_temperature, 285.0, 305.0)
        
        new_density = jnp.where(
            is_active,
            local_density * (1.0 + 0.1 * jnp.sin(Q_Lambda)),
            state.density[i]
        )
        
        # 創発度（幾何構造からの逸脱）
        emergence = jnp.where(
            is_active,
            jnp.tanh(jnp.linalg.norm(structure_force) / 5.0),
            state.emergence[i]
        )
        
        # 障害物からの距離
        particle_dx = pos[0] - config.obstacle_center_x
        particle_dy = pos[1] - config.obstacle_center_y
        particle_r = jnp.sqrt(particle_dx**2 + particle_dy**2)
        
        near_wall = (particle_r > config.obstacle_size) & (particle_r < config.obstacle_size + 5.0)
        
        return (
            jnp.where(is_active, new_Lambda_F, state.Lambda_F[i]),
            jnp.where(is_active, new_Lambda_FF, state.Lambda_FF[i]),
            jnp.where(is_active, state.Lambda_F[i], state.prev_Lambda_F[i]),
            jnp.where(is_active, Lambda_core, state.Lambda_core[i]),
            jnp.where(is_active, rho_T, state.rho_T[i]),
            jnp.where(is_active, sigma_s, state.sigma_s[i]),
            jnp.where(is_active, Q_Lambda, state.Q_Lambda[i]),
            jnp.where(is_active, efficiency, state.efficiency[i]),
            jnp.where(is_active, emergence, state.emergence[i]),
            jnp.where(is_active, new_temperature, state.temperature[i]),
            jnp.where(is_active, new_density, state.density[i]),
            jnp.where(is_active, vorticity, state.vorticity[i]),
            jnp.where(is_active, Q, state.Q_criterion[i]),
            jnp.where(is_active, is_DeltaLambdaC, state.DeltaLambdaC[i]),
            jnp.where(is_active, event_score, state.event_score[i]),
            jnp.where(is_active, is_separated, state.is_separated[i]),
            jnp.where(is_active, near_wall, state.near_wall[i])
        )
    
    # 全粒子を並列更新
    results = vmap(update_particle)(jnp.arange(N))
    
    # 結果を展開
    new_Lambda_F = results[0]
    new_Lambda_FF = results[1]
    new_prev_Lambda_F = results[2]
    new_Lambda_core = results[3]
    new_rho_T = results[4]
    new_sigma_s = results[5]
    new_Q_Lambda = results[6]
    new_efficiency = results[7]
    new_emergence = results[8]
    new_temperature = results[9]
    new_density = results[10]
    new_vorticity = results[11]
    new_Q_criterion = results[12]
    new_DeltaLambdaC = results[13]
    new_event_score = results[14]
    new_is_separated = results[15]
    new_near_wall = results[16]
    
    # 位置更新
    new_positions = state.position + new_Lambda_F * config.dt
    
    # 年齢更新
    new_age = state.age + jnp.where(active_mask, 1.0, 0.0)
    
    # 境界チェック
    new_active = active_mask & (new_positions[:, 0] < config.domain_width)
    
    return ParticleState(
        position=new_positions,
        Lambda_F=new_Lambda_F,
        Lambda_FF=new_Lambda_FF,
        prev_Lambda_F=new_prev_Lambda_F,
        Lambda_core=new_Lambda_core,
        rho_T=new_rho_T,
        sigma_s=new_sigma_s,
        prev_sigma_s=state.sigma_s,
        Q_Lambda=new_Q_Lambda,
        prev_Q_Lambda=state.Q_Lambda,
        efficiency=new_efficiency,
        emergence=new_emergence,
        temperature=new_temperature,
        density=new_density,
        vorticity=new_vorticity,
        Q_criterion=new_Q_criterion,
        DeltaLambdaC=new_DeltaLambdaC,
        event_score=new_event_score,
        age=new_age,
        is_active=new_active,
        is_separated=new_is_separated,
        near_wall=new_near_wall
    )

# ==============================
# 粒子注入（変更なし）
# ==============================

def inject_particles(state: ParticleState, config: GETWindConfig,
                    key: random.PRNGKey, step: int) -> ParticleState:
    """新粒子の注入"""
    key1, key2, key3 = random.split(key, 3)
    
    n_inject_float = random.poisson(key1, config.particles_per_step)
    n_inject = jnp.minimum(jnp.int32(n_inject_float), 10)
    
    inactive_mask = ~state.is_active
    inactive_count = jnp.sum(inactive_mask)
    
    n_to_inject = jnp.minimum(n_inject, inactive_count)
    
    cumsum = jnp.cumsum(jnp.where(inactive_mask, 1, 0))
    inject_mask = (cumsum <= n_to_inject) & inactive_mask
    
    N = state.position.shape[0]
    
    # 新しい位置とΛF
    y_positions = random.uniform(key2, (N,), minval=5, maxval=config.domain_height-5)
    x_positions = random.uniform(key3, (N,), minval=0, maxval=5)
    
    Lambda_Fx = jnp.ones(N) * config.Lambda_F_inlet + random.normal(key2, (N,)) * 0.1
    Lambda_Fy = random.normal(key3, (N,)) * 0.1
    
    # 初期温度
    temperatures = 293.0 + 5.0 * (1 - y_positions / config.domain_height)
    
    # 更新
    new_positions = jnp.where(
        inject_mask[:, None],
        jnp.stack([x_positions, y_positions], axis=1),
        state.position
    )
    
    new_Lambda_F = jnp.where(
        inject_mask[:, None],
        jnp.stack([Lambda_Fx, Lambda_Fy], axis=1),
        state.Lambda_F
    )
    
    return ParticleState(
        position=new_positions,
        Lambda_F=new_Lambda_F,
        Lambda_FF=jnp.where(inject_mask[:, None], jnp.zeros((N, 2)), state.Lambda_FF),
        prev_Lambda_F=new_Lambda_F,
        Lambda_core=jnp.where(inject_mask[:, None], jnp.zeros((N, 4)), state.Lambda_core),
        rho_T=jnp.where(inject_mask, jnp.linalg.norm(new_Lambda_F, axis=1), state.rho_T),
        sigma_s=jnp.where(inject_mask, 0.0, state.sigma_s),
        prev_sigma_s=jnp.where(inject_mask, 0.0, state.prev_sigma_s),
        Q_Lambda=jnp.where(inject_mask, 0.0, state.Q_Lambda),
        prev_Q_Lambda=jnp.where(inject_mask, 0.0, state.prev_Q_Lambda),
        efficiency=jnp.where(inject_mask, 0.5, state.efficiency),
        emergence=jnp.where(inject_mask, 0.0, state.emergence),
        temperature=jnp.where(inject_mask, temperatures, state.temperature),
        density=jnp.where(inject_mask, 1.225, state.density),
        vorticity=jnp.where(inject_mask, 0.0, state.vorticity),
        Q_criterion=jnp.where(inject_mask, 0.0, state.Q_criterion),
        DeltaLambdaC=jnp.where(inject_mask, False, state.DeltaLambdaC),
        event_score=jnp.where(inject_mask, 0.0, state.event_score),
        age=jnp.where(inject_mask, 0.0, state.age),
        is_active=inject_mask | state.is_active,
        is_separated=jnp.where(inject_mask, False, state.is_separated),
        near_wall=jnp.where(inject_mask, False, state.near_wall)
    )

# ==============================
# メインシミュレーション（履歴保存を間引き）
# ==============================

def run_simulation_v63(map_file: str, config: GETWindConfig, 
                      seed: int = 42, save_states: bool = True,
                      snapshot_interval: int = 50):
    """GET Wind™ v6.3 メインシミュレーション（幾何MAP駆動・修正版）"""
    
    # 幾何マップ読み込み
    map_data = GeometricMapData(map_file)
    
    # 乱数キー
    key = random.PRNGKey(seed)
    
    # 初期状態
    N = config.max_particles
    initial_state = ParticleState(
        position=jnp.zeros((N, 2)),
        Lambda_F=jnp.zeros((N, 2)),
        Lambda_FF=jnp.zeros((N, 2)),
        prev_Lambda_F=jnp.zeros((N, 2)),
        Lambda_core=jnp.zeros((N, 4)),
        rho_T=jnp.zeros(N),
        sigma_s=jnp.zeros(N),
        prev_sigma_s=jnp.zeros(N),
        Q_Lambda=jnp.zeros(N),
        prev_Q_Lambda=jnp.zeros(N),
        efficiency=jnp.ones(N) * 0.5,
        emergence=jnp.zeros(N),
        temperature=jnp.ones(N) * 293.0,
        density=jnp.ones(N) * 1.225,
        vorticity=jnp.zeros(N),
        Q_criterion=jnp.zeros(N),
        DeltaLambdaC=jnp.zeros(N, dtype=bool),
        event_score=jnp.zeros(N),
        age=jnp.zeros(N),
        is_active=jnp.zeros(N, dtype=bool),
        is_separated=jnp.zeros(N, dtype=bool),
        near_wall=jnp.zeros(N, dtype=bool)
    )
    
    print("=" * 70)
    print("GET Wind™ v6.3 - Geometric Bernoulli Integration [FIXED EDITION]")
    print("環ちゃん & ご主人さま Ultimate Geometry Edition! 💕")
    print(f"Map: {map_file}")
    print(f"Max particles: {N}")
    print(f"Steps: {config.n_steps}")
    print(f"Snapshot interval: {snapshot_interval}")
    print("Features: Fixed coordinate transforms, optimized branches, proper gradients")
    print("=" * 70)
    
    # JITコンパイル
    print("Compiling JIT functions...")
    start_compile = time.time()
    
    key, subkey = random.split(key)
    dummy_state = inject_particles(initial_state, config, subkey, 0)
    key, subkey = random.split(key)
    _ = physics_step_v63(
        dummy_state,
        map_data.pressure,
        map_data.density,
        map_data.separation,
        map_data.vortex_formation,
        map_data.wake_structure,
        map_data.shear_layer,
        map_data.nx,
        map_data.ny,
        config,
        subkey
    )
    
    print(f"JIT compilation done in {time.time() - start_compile:.2f}s")
    
    # メインループ
    state = initial_state
    history = []
    state_history = []
    
    print("\nStarting simulation...")
    start_time = time.time()
    
    for step in range(config.n_steps):
        # 粒子注入
        key, subkey = random.split(key)
        state = inject_particles(state, config, subkey, step)
        
        # 物理ステップ
        key, subkey = random.split(key)
        state = physics_step_v63(
            state,
            map_data.pressure,
            map_data.density,
            map_data.separation,
            map_data.vortex_formation,
            map_data.wake_structure,
            map_data.shear_layer,
            map_data.nx,
            map_data.ny,
            config,
            subkey
        )
        
        # 状態保存（間引き）
        if save_states:
            state_history.append({
                'step': step,
                'position': np.array(state.position),
                'Lambda_F': np.array(state.Lambda_F),
                'vorticity': np.array(state.vorticity),
                'Q_criterion': np.array(state.Q_criterion),
                'is_active': np.array(state.is_active),
                'is_separated': np.array(state.is_separated)
            })
        
        # 定期的な統計出力
        if step % 100 == 0 or step == config.n_steps - 1:
            active_count = jnp.sum(state.is_active)
            
            if active_count > 0:
                active_mask = state.is_active
                
                mean_Lambda_F = jnp.mean(jnp.linalg.norm(state.Lambda_F[active_mask], axis=1))
                mean_vorticity = jnp.mean(jnp.abs(state.vorticity[active_mask]))
                max_vorticity = jnp.max(jnp.abs(state.vorticity[active_mask]))
                mean_emergence = jnp.mean(state.emergence[active_mask])
                n_separated = jnp.sum(state.is_separated & active_mask)
                
                print(f"\nStep {step:4d}: {int(active_count):4d} particles")
                print(f"  |ΛF|={mean_Lambda_F:.2f}, |ω|={mean_vorticity:.3f} (max={max_vorticity:.3f})")
                print(f"  Emergence={mean_emergence:.3f}, Separated={int(n_separated)}")
                
                history.append({
                    'step': step,
                    'n_particles': int(active_count),
                    'mean_Lambda_F': float(mean_Lambda_F),
                    'mean_vorticity': float(mean_vorticity),
                    'max_vorticity': float(max_vorticity),
                    'mean_emergence': float(mean_emergence),
                    'n_separated': int(n_separated)
                })
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE!")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Performance: {config.n_steps / elapsed:.1f} steps/sec")
    print(f"Saved {len(state_history)} snapshots")
    print("=" * 70)
    
    # 結果保存
    if save_states:
        shape_name = "cylinder" if config.obstacle_shape == 0 else "square"
        filename = f"simulation_results_v63_fixed_{shape_name}.npz"
        np.savez_compressed(filename,
                           states=state_history,
                           history=history,
                           config=config._asdict())
        print(f"\nResults saved to {filename} (compressed)")
    
    return state, history

# ==============================
# メイン実行
# ==============================

if __name__ == "__main__":
    # 設定
    config = GETWindConfig(
        obstacle_shape=0,  # 0=cylinder, 1=square
        particles_per_step=5.0,
        max_particles=1500,
        n_steps=5000,
        dt=0.02,
        
        # 流れパラメータ
        Lambda_F_inlet=10.0,
        
        # Λ³パラメータ（調整済み）
        thermal_alpha=0.01,
        density_beta=0.02,
        structure_coupling=0.03,
        viscosity_factor=0.1,
        interaction_strength=0.1,
        
        # 幾何MAP用パラメータ
        shear_instability_strength=0.8,
        vortex_formation_noise=1.2,
        wake_turbulence_factor=1.0,
        
        # 障害物設定
        obstacle_center_x=100.0,
        obstacle_center_y=75.0,
        obstacle_size=20.0
    )
    
    # マップファイル（幾何MAP）
    shape_name = "cylinder" if config.obstacle_shape == 0 else "square"
    map_file = f"{shape_name}_Re200_geometric.npz"
    
    print("\n" + "=" * 70)
    print("GET Wind™ v6.3 Fixed - Geometric Bernoulli Edition 🌀")
    print("With 環's complete patch applied!")
    print("=" * 70)
    
    try:
        final_state, history = run_simulation_v63(map_file, config, 
                                                  save_states=True, 
                                                  snapshot_interval=50)
        print("\n✨ v6.3 Fixed Complete! Physics emerges from geometry! ✨")
        
    except FileNotFoundError:
        print(f"\n⚠ Map file '{map_file}' not found!")
        print("Please run the Geometric Bernoulli Map Generator first.")
