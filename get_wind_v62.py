#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GET Wind™ v6.2 JAX Edition - Spatial Coherence Vortex Detection
環ちゃん & ご主人さま Ultimate Physics Fix Edition! 💕

v6.2の改良点：
  - 空間的速度場同期による物理的に正しい渦検出
  - ΔΛCの過検出問題を解決
  - カルマン渦列の安定性評価を改善
  - トポロジカル保存則はそのまま維持
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

# ==============================
# Configuration
# ==============================

class GETWindConfig(NamedTuple):
    """GET Wind™ v6.2 設定（Λ³ Enhanced + Spatial Coherence）"""
    # シミュレーション領域
    domain_width: float = 300.0
    domain_height: float = 150.0
    
    # マップ解像度
    map_nx: int = 300
    map_ny: int = 150
    
    # Λ³パラメータ
    Lambda_F_inlet: float = 10.0
    thermal_alpha: float = 0.008      # 温度勾配の重み
    density_beta: float = 0.015       # 密度勾配の重み
    structure_coupling: float = 0.025  # 構造結合強度
    viscosity_factor: float = 0.1      # 粘性係数
    interaction_strength: float = 0.1  # 粒子間相互作用強度
    
    # 効率パラメータ
    efficiency_threshold: float = 0.1
    efficiency_weight: float = 0.5
    
    # トポロジカルパラメータ
    topological_threshold: float = 0.1  # Q_Λジャンプの閾値
    sync_threshold: float = 0.05        # σₛジャンプの閾値
    
    # 渦検出パラメータ（v6.2新規）
    coherence_threshold: float = 0.6    # 速度場同期度の閾値
    circulation_threshold: float = 1.0   # 循環の最小値
    min_particles_per_region: int = 20  # 領域あたり最小粒子数
    vortex_grid_size: float = 10.0     # 渦検出グリッドサイズ
    
    # 粒子パラメータ
    particles_per_step: float = 5.0
    max_particles: int = 1500
    dt: float = 0.05
    n_steps: int = 3000
    
    # 物理定数
    obstacle_center_x: float = 100.0
    obstacle_center_y: float = 75.0
    obstacle_size: float = 20.0

# ==============================
# Map Manager
# ==============================

class MapData:
    """マップデータの管理"""
    
    def __init__(self, npz_file: str):
        """NPZファイルからマップを読み込み"""
        print(f"Loading map from {npz_file}...")
        data = np.load(npz_file)
        
        # 各フィールドをJAX配列に変換
        self.density = jnp.array(data['density'])
        self.pressure = jnp.array(data['pressure'])
        self.separation = jnp.array(data['separation'])
        self.vorticity_potential = jnp.array(data['vorticity_potential'])
        self.wake_region = jnp.array(data['wake_region'])
        
        # 速度場（参考用）
        self.velocity_u = jnp.array(data['velocity_u'])
        self.velocity_v = jnp.array(data['velocity_v'])
        
        # グリッド情報
        self.nx, self.ny = self.density.shape
        
        print(f"Map loaded: {self.nx}x{self.ny} grid")

# ==============================
# Particle State（Λ³ Enhanced）
# ==============================

class ParticleState(NamedTuple):
    """粒子状態（v6.2: 変更なし）"""
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
# 補間処理
# ==============================

@jit
def bilinear_interpolate(field: jnp.ndarray, x: float, y: float, 
                         nx: int, ny: int) -> float:
    """バイリニア補間"""
    i = jnp.clip(jnp.floor(x).astype(int), 0, nx-2)
    j = jnp.clip(jnp.floor(y).astype(int), 0, ny-2)
    
    fx = x - i
    fy = y - j
    
    v00 = field[i, j]
    v10 = field[jnp.minimum(i+1, nx-1), j]
    v01 = field[i, jnp.minimum(j+1, ny-1)]
    v11 = field[jnp.minimum(i+1, nx-1), jnp.minimum(j+1, ny-1)]
    
    return (1-fx)*(1-fy)*v00 + fx*(1-fy)*v10 + (1-fx)*fy*v01 + fx*fy*v11

@jit
def compute_gradient_from_map(field: jnp.ndarray, x: float, y: float,
                              nx: int, ny: int) -> jnp.ndarray:
    """マップから勾配を計算"""
    h = 1.0
    
    fx_plus = bilinear_interpolate(field, jnp.minimum(x+h, nx-1), y, nx, ny)
    fx_minus = bilinear_interpolate(field, jnp.maximum(x-h, 0), y, nx, ny)
    grad_x = (fx_plus - fx_minus) / (2*h)
    
    fy_plus = bilinear_interpolate(field, x, jnp.minimum(y+h, ny-1), nx, ny)
    fy_minus = bilinear_interpolate(field, x, jnp.maximum(y-h, 0), nx, ny)
    grad_y = (fy_plus - fy_minus) / (2*h)
    
    return jnp.array([grad_x, grad_y])

# ==============================
# Λ³構造テンソル計算
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
def compute_efficiency(Lambda_core: jnp.ndarray, Lambda_F: jnp.ndarray) -> float:
    """構造の効率計算"""
    norm_LF = jnp.linalg.norm(Lambda_F) + 1e-8
    
    # Lambda_coreの最初の2成分をΛFに射影
    proj = jnp.dot(Lambda_core[:2], Lambda_F) / norm_LF
    
    # 構造の一貫性
    coherence = jnp.exp(-jnp.var(Lambda_core))
    
    return jnp.abs(proj) * coherence

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
# ΔΛC検出（構造変化点）
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
# 構造間相互作用（Λ³の本質！）
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
    """構造間相互作用（Λ³ Enhanced + Vortex Merging）"""
    
    dr = neighbor_positions - pos_i
    distances = jnp.linalg.norm(dr, axis=1) + 1e-8
    
    # 🔧 相互作用範囲を拡大！（渦の結合のため）
    near_range = neighbor_mask & (distances < 15.0)   # 近距離
    far_range = neighbor_mask & (distances < 30.0)    # 遠距離（渦結合用）
    
    # === 1. テンション密度の勾配による力（変更なし） ===
    drho = neighbor_rho_T - rho_T_i
    grad_rho_force = jnp.sum(
        jnp.where(near_range[:, None], 
                  (drho[:, None] / distances[:, None]**2) * dr * config.density_beta,
                  0),
        axis=0
    )
    
    # === 2. 構造テンソルの差による力（範囲拡大） ===
    Lambda_core_2x2 = Lambda_core_i.reshape(2, 2)
    
    def compute_tensor_force(idx):
        neighbor_core_2x2 = neighbor_Lambda_core[idx].reshape(2, 2)
        
        # テンソル差のノルム
        tensor_diff = neighbor_core_2x2 - Lambda_core_2x2
        diff_norm = jnp.linalg.norm(tensor_diff, 'fro')
        
        # 構造の不一致による反発/引力
        direction = dr[idx] / distances[idx]
        force_mag = diff_norm * jnp.exp(-distances[idx] / 15.0)  # 10→15
        
        # 同期率で重み付け
        sync_weight = 1.0 + (neighbor_sigma_s[idx] - sigma_s_i)
        
        force = direction * force_mag * sync_weight * config.structure_coupling
        
        return jnp.where(near_range[idx], force, jnp.zeros(2))
    
    tensor_forces = vmap(compute_tensor_force)(jnp.arange(len(neighbor_positions)))
    tensor_force = jnp.sum(tensor_forces, axis=0)
    
    # === 3. 渦的相互作用（強化版！） ===
    vorticity_i = Lambda_core_2x2[1, 0] - Lambda_core_2x2[0, 1]
    
    # 3a. 基本的な渦の回転力（近距離）
    tangent = jnp.stack([-dr[:, 1], dr[:, 0]], axis=1) / distances[:, None]
    
    vortex_rotation = jnp.sum(
        jnp.where(
            near_range[:, None],
            tangent * vorticity_i * jnp.exp(-distances[:, None] / 15.0) * 0.2,  # 0.1→0.2
            0
        ),
        axis=0
    )
    
    # 🆕 3b. 同回転渦の結合力（遠距離まで作用）
    def compute_vortex_merging(idx):
        # 近傍の渦度
        neighbor_vorticity = neighbor_Lambda_core[idx].reshape(2, 2)[1, 0] - \
                           neighbor_Lambda_core[idx].reshape(2, 2)[0, 1]
        
        # 同じ回転方向かチェック
        same_rotation = vorticity_i * neighbor_vorticity > 0
        
        # 渦度の強さに比例した引力（同回転のみ）
        attraction = jnp.abs(neighbor_vorticity * vorticity_i) * same_rotation
        
        # 距離に応じた減衰（でも遠くまで届く）
        r = distances[idx]
        force_mag = attraction * jnp.exp(-r / 25.0) * (1 - jnp.exp(-r / 3.0))  # 近すぎると弱い
        
        # 引力の方向
        direction = dr[idx] / r
        
        return jnp.where(far_range[idx] & same_rotation, direction * force_mag * 0.15, jnp.zeros(2))
    
    vortex_merging = jnp.sum(
        vmap(compute_vortex_merging)(jnp.arange(len(neighbor_positions))),
        axis=0
    )
    
    # 渦力の合計
    vortex_force = vortex_rotation + vortex_merging
    
    # === 4. 粘性的相互作用（調整版） ===
    mean_Lambda_F = jnp.sum(
        jnp.where(near_range[:, None], neighbor_Lambda_F, 0),
        axis=0
    ) / jnp.maximum(jnp.sum(near_range), 1)
    
    # 🔧 粘性を渦度に応じて調整（渦が強い時は粘性下げる）
    vorticity_factor = jnp.exp(-jnp.abs(vorticity_i) / 2.0)  # 渦が強いと粘性減
    effective_viscosity = jnp.minimum(config.viscosity_factor * 0.05 * vorticity_factor, 0.2)
    viscous_force = effective_viscosity * (mean_Lambda_F - Lambda_F_i)
    
    # === 5. 全体の力を合成 ===
    total_interaction = grad_rho_force + tensor_force + vortex_force + viscous_force
    
    # 相互作用力の大きさを制限（少し緩める）
    max_interaction = 5.0  # 3.0→5.0
    interaction_norm = jnp.linalg.norm(total_interaction)
    total_interaction = jnp.where(
        interaction_norm > max_interaction,
        total_interaction * max_interaction / interaction_norm,
        total_interaction
    )
    
    return total_interaction

# ==============================
# トポロジカル保存フィードバック
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
# 近傍探索
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
# 渦検出（v6.2: 空間同期ベース）
# ==============================

@jit
def compute_region_coherence(Lambda_F: jnp.ndarray, mask: jnp.ndarray) -> float:
    """領域内でのΛF（速度場）の同期度を計算"""
    n_particles = jnp.sum(mask)
    
    # 少なすぎる場合は0
    too_few = n_particles < 3
    
    # 平均速度ベクトル
    mean_Lambda_F = jnp.sum(
        jnp.where(mask[:, None], Lambda_F, 0), axis=0
    ) / jnp.maximum(n_particles, 1)
    
    # 各粒子と平均のコサイン類似度
    dots = jnp.sum(Lambda_F * mean_Lambda_F[None, :], axis=1)
    norms = jnp.linalg.norm(Lambda_F, axis=1) * jnp.linalg.norm(mean_Lambda_F) + 1e-8
    similarities = dots / norms
    
    # マスクされた粒子のみの平均類似度
    coherence = jnp.sum(
        jnp.where(mask, similarities, 0)
    ) / jnp.maximum(n_particles, 1)
    
    return jnp.where(too_few, 0.0, coherence)

@jit
def compute_region_circulation(Lambda_F: jnp.ndarray, 
                              positions: jnp.ndarray,
                              mask: jnp.ndarray,
                              center_y: float) -> float:
    """領域内での循環（渦の回転強度）を計算"""
    n_particles = jnp.sum(mask)
    
    # 重心を計算
    center_x = jnp.sum(jnp.where(mask, positions[:, 0], 0)) / jnp.maximum(n_particles, 1)
    center = jnp.array([center_x, center_y])
    
    # 各粒子の相対位置
    rel_pos = positions - center[None, :]
    
    # 接線方向の速度成分（r × v）のz成分
    cross_product_z = rel_pos[:, 0] * Lambda_F[:, 1] - rel_pos[:, 1] * Lambda_F[:, 0]
    distances = jnp.linalg.norm(rel_pos, axis=1) + 1e-8
    
    # 重み付き循環
    weights = jnp.exp(-distances / 20.0)
    
    circulation = jnp.sum(
        jnp.where(mask, cross_product_z * weights / distances, 0)
    ) / jnp.maximum(jnp.sum(jnp.where(mask, weights, 0)), 1e-8)
    
    return circulation

@jit
def detect_karman_vortex_v2(state: ParticleState, config: GETWindConfig) -> Tuple[bool, float, dict]:
    """
    カルマン渦列の検出（v6.2: 空間同期版）
    
    原理：
    1. 後流領域を上下に分割
    2. 各領域での速度場の同期度を計算
    3. 循環（渦の回転）を計算
    4. 上下で反対回転の渦が交互に並んでいればカルマン渦列
    """
    active = state.is_active
    
    # === 1. 後流領域の粒子を抽出 ===
    wake_start = config.obstacle_center_x + 20.0
    wake_mask = active & (state.position[:, 0] > wake_start)
    n_wake_particles = jnp.sum(wake_mask)
    
    # 粒子が少なすぎる場合
    too_few = n_wake_particles < config.min_particles_per_region * 2
    
    # === 2. 上下領域に分割 ===
    center_y = config.obstacle_center_y
    upper_mask = wake_mask & (state.position[:, 1] > center_y)
    lower_mask = wake_mask & (state.position[:, 1] <= center_y)
    
    n_upper = jnp.sum(upper_mask)
    n_lower = jnp.sum(lower_mask)
    
    # === 3. 各領域での同期度と循環を計算 ===
    
    # 上側領域
    upper_coherence = compute_region_coherence(state.Lambda_F, upper_mask)
    upper_circulation = compute_region_circulation(
        state.Lambda_F, state.position, upper_mask, center_y + 20.0
    )
    
    # 下側領域
    lower_coherence = compute_region_coherence(state.Lambda_F, lower_mask)
    lower_circulation = compute_region_circulation(
        state.Lambda_F, state.position, lower_mask, center_y - 20.0
    )
    
    # === 4. カルマン渦列の判定 ===
    
    # 条件1: 上下両方に高同期領域がある
    has_upper_vortex = (
        (upper_coherence > config.coherence_threshold) & 
        (jnp.abs(upper_circulation) > config.circulation_threshold) &
        (n_upper >= config.min_particles_per_region)
    )
    has_lower_vortex = (
        (lower_coherence > config.coherence_threshold) & 
        (jnp.abs(lower_circulation) > config.circulation_threshold) &
        (n_lower >= config.min_particles_per_region)
    )
    
    # 条件2: 循環が反対向き（渦が反対回転）
    opposite_rotation = (upper_circulation * lower_circulation) < 0
    
    # 条件3: 位相差がある（簡易判定）
    upper_x_mean = jnp.sum(jnp.where(upper_mask, state.position[:, 0], 0)) / jnp.maximum(n_upper, 1)
    lower_x_mean = jnp.sum(jnp.where(lower_mask, state.position[:, 0], 0)) / jnp.maximum(n_lower, 1)
    x_diff = jnp.abs(upper_x_mean - lower_x_mean)
    has_phase_diff = x_diff > 3.0  # 位相差の最小値
    
    # === 5. 最終判定 ===
    is_karman = (
        ~too_few &  # 十分な粒子がある
        has_upper_vortex & has_lower_vortex &  # 上下に渦
        opposite_rotation  # 反対回転
        # has_phase_diffは厳しすぎるので一旦外す
    )
    
    # === 6. 安定性スコア ===
    # 同期度の平均
    avg_coherence = (upper_coherence + lower_coherence) / 2.0
    
    # 循環の強さのバランス
    circulation_balance = 1.0 - jnp.abs(
        jnp.abs(upper_circulation) - jnp.abs(lower_circulation)
    ) / (jnp.abs(upper_circulation) + jnp.abs(lower_circulation) + 1e-8)
    
    # 粒子数のバランス
    particle_balance = 1.0 - jnp.abs(n_upper - n_lower) / (n_upper + n_lower + 1e-8)
    
    # 総合安定性
    stability = jnp.where(
        is_karman,
        avg_coherence * 0.4 + circulation_balance * 0.3 + particle_balance * 0.3,
        0.0
    )
    
    # === 7. 詳細メトリクス（JIT互換版）===
    # JIT内ではint()やfloat()が使えないので、そのまま返す
    metrics = {
        'n_wake': n_wake_particles,  # JIT内ではそのまま
        'upper_coherence': upper_coherence,
        'lower_coherence': lower_coherence,
        'upper_circulation': upper_circulation,
        'lower_circulation': lower_circulation,
        'phase_diff': x_diff,
        'circulation_balance': circulation_balance,
        'particle_balance': particle_balance
    }
    
    return is_karman, stability, metrics

# ==============================
# メイン物理ステップ（変更なし）
# ==============================

@partial(jit, static_argnums=(8,))
def physics_step_v62(state: ParticleState,
                    density_map: jnp.ndarray,
                    pressure_map: jnp.ndarray,
                    separation_map: jnp.ndarray,
                    velocity_u_map: jnp.ndarray,
                    velocity_v_map: jnp.ndarray,
                    map_nx: int, map_ny: int,
                    config: GETWindConfig,
                    key: random.PRNGKey) -> ParticleState:
    """v6.2の物理ステップ（Map-Driven + Λ³）"""
    
    active_mask = state.is_active
    N = state.position.shape[0]
    obstacle_center = jnp.array([config.obstacle_center_x, config.obstacle_center_y])
    
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
    all_neighbor_density = state.density[neighbor_indices]
    all_neighbor_temperature = state.temperature[neighbor_indices]
    
    def update_particle(i):
        """各粒子の更新"""
        is_active = active_mask[i]
        
        pos = state.position[i]
        grid_x = pos[0]
        grid_y = pos[1]
        
        # === 1. マップから基本場を取得 ===
        local_density = bilinear_interpolate(density_map, grid_x, grid_y, map_nx, map_ny)
        local_pressure = bilinear_interpolate(pressure_map, grid_x, grid_y, map_nx, map_ny)
        local_separation = bilinear_interpolate(separation_map, grid_x, grid_y, map_nx, map_ny)
        expected_u = bilinear_interpolate(velocity_u_map, grid_x, grid_y, map_nx, map_ny)
        expected_v = bilinear_interpolate(velocity_v_map, grid_x, grid_y, map_nx, map_ny)
        
        # 勾配
        grad_pressure = compute_gradient_from_map(pressure_map, grid_x, grid_y, map_nx, map_ny)
        grad_density = compute_gradient_from_map(density_map, grid_x, grid_y, map_nx, map_ny)
        
        # === 2. Λ³構造テンソルの計算 ===
        neighbor_pos = all_neighbor_positions[i]
        neighbor_Lambda_F = all_neighbor_Lambda_F[i]
        neighbor_Lambda_core = all_neighbor_Lambda_core[i]
        neighbor_rho_T = all_neighbor_rho_T[i]
        neighbor_sigma_s = all_neighbor_sigma_s[i]
        neighbor_valid = neighbor_mask[i]
        
        grad_Lambda = jnp.where(
            is_active,
            compute_Lambda_gradient(
                state.Lambda_F[i], pos,
                neighbor_Lambda_F, neighbor_pos, neighbor_valid
            ),
            jnp.eye(2)
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
        sigma_s = jnp.where(
            is_active,
            compute_sigma_s(
                state.rho_T[i], state.Lambda_F[i],
                neighbor_rho_T, neighbor_pos, pos,
                neighbor_valid
            ),
            state.sigma_s[i]
        )
        
        # トポロジカル不変量
        Q_Lambda = jnp.where(
            is_active,
            compute_local_Q_Lambda(
                state.Lambda_F[i], pos,
                neighbor_Lambda_F, neighbor_pos,
                neighbor_valid
            ),
            state.Q_Lambda[i]
        )
        
        # 効率
        efficiency = jnp.where(
            is_active,
            compute_efficiency(Lambda_core, state.Lambda_F[i]),
            state.efficiency[i]
        )
        
        # === 3. 構造間相互作用 ===
        structure_force = jnp.where(
            is_active,
            compute_structure_interaction(
                state.Lambda_F[i], pos, Lambda_core,
                rho_T, sigma_s,
                neighbor_Lambda_F, neighbor_pos,
                neighbor_Lambda_core, neighbor_rho_T, neighbor_sigma_s,
                neighbor_valid, config
            ),
            jnp.zeros(2)
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
        
        # === 5. ΛF更新 ===
        base_force = -config.thermal_alpha * grad_pressure - config.density_beta * grad_density
        new_Lambda_F_base = state.Lambda_F[i] + base_force + structure_force
        
        # ΔΛCイベント時の追加処理
        subkey = random.fold_in(key, i * 1000)
        DeltaLambdaC_noise = random.normal(subkey, (2,)) * 0.5
        new_Lambda_F = jnp.where(
            is_DeltaLambdaC,
            new_Lambda_F_base + DeltaLambdaC_noise,
            new_Lambda_F_base
        )
        
        # 剥離領域での処理
        sep_key = random.fold_in(key, i * 2000)
        sep_noise = random.normal(sep_key, (2,)) * local_separation
        new_Lambda_F = jnp.where(
            local_separation > 0.2,
            new_Lambda_F + sep_noise,
            new_Lambda_F
        )
        
        # 剥離フラグ更新
        is_separated = jnp.where(
            local_separation > 0.2,
            True,
            state.is_separated[i]
        )
        
        # トポロジカル保存フィードバック
        feedback_key = random.fold_in(key, i * 3000)
        new_Lambda_F = jnp.where(
            is_active,
            apply_topological_feedback(
                upper_DQ, lower_DQ,
                new_Lambda_F, efficiency,
                pos[1], config.obstacle_center_y,
                is_separated, config,
                feedback_key
            ),
            state.Lambda_F[i]
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
        
        # === 6. 物理量更新 ===
        temp_noise = random.normal(random.fold_in(key, i * 4000)) * 2.0
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
        
        expected = jnp.array([expected_u, expected_v])
        emergence = jnp.where(
            is_active,
            jnp.tanh(jnp.linalg.norm(new_Lambda_F - expected) / 10.0),
            state.emergence[i]
        )
        
        dist_to_obstacle = jnp.linalg.norm(pos - obstacle_center) - config.obstacle_size
        near_wall = (dist_to_obstacle > 0) & (dist_to_obstacle < 5.0)
        
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
# 粒子注入
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
    
    new_Lambda_FF = jnp.where(
        inject_mask[:, None],
        jnp.zeros((N, 2)),
        state.Lambda_FF
    )
    
    new_prev_Lambda_F = jnp.where(
        inject_mask[:, None],
        jnp.stack([Lambda_Fx, Lambda_Fy], axis=1),
        state.prev_Lambda_F
    )
    
    # Λ³構造の初期化
    new_Lambda_core = jnp.where(
        inject_mask[:, None],
        jnp.zeros((N, 4)),
        state.Lambda_core
    )
    
    new_rho_T = jnp.where(
        inject_mask,
        jnp.linalg.norm(new_Lambda_F, axis=1),
        state.rho_T
    )
    
    new_sigma_s = jnp.where(inject_mask, 0.0, state.sigma_s)
    new_prev_sigma_s = jnp.where(inject_mask, 0.0, state.prev_sigma_s)
    new_Q_Lambda = jnp.where(inject_mask, 0.0, state.Q_Lambda)
    new_prev_Q_Lambda = jnp.where(inject_mask, 0.0, state.prev_Q_Lambda)
    
    new_efficiency = jnp.where(inject_mask, 0.5, state.efficiency)
    new_emergence = jnp.where(inject_mask, 0.0, state.emergence)
    
    new_temperature = jnp.where(inject_mask, temperatures, state.temperature)
    new_density = jnp.where(inject_mask, 1.225, state.density)
    
    new_vorticity = jnp.where(inject_mask, 0.0, state.vorticity)
    new_Q_criterion = jnp.where(inject_mask, 0.0, state.Q_criterion)
    new_DeltaLambdaC = jnp.where(inject_mask, False, state.DeltaLambdaC)
    new_event_score = jnp.where(inject_mask, 0.0, state.event_score)
    
    new_age = jnp.where(inject_mask, 0.0, state.age)
    new_is_active = inject_mask | state.is_active
    new_is_separated = jnp.where(inject_mask, False, state.is_separated)
    new_near_wall = jnp.where(inject_mask, False, state.near_wall)
    
    return ParticleState(
        position=new_positions,
        Lambda_F=new_Lambda_F,
        Lambda_FF=new_Lambda_FF,
        prev_Lambda_F=new_prev_Lambda_F,
        Lambda_core=new_Lambda_core,
        rho_T=new_rho_T,
        sigma_s=new_sigma_s,
        prev_sigma_s=new_prev_sigma_s,
        Q_Lambda=new_Q_Lambda,
        prev_Q_Lambda=new_prev_Q_Lambda,
        efficiency=new_efficiency,
        emergence=new_emergence,
        temperature=new_temperature,
        density=new_density,
        vorticity=new_vorticity,
        Q_criterion=new_Q_criterion,
        DeltaLambdaC=new_DeltaLambdaC,
        event_score=new_event_score,
        age=new_age,
        is_active=new_is_active,
        is_separated=new_is_separated,
        near_wall=new_near_wall
    )

# ==============================
# メインシミュレーション
# ==============================

def run_simulation_v62(map_file: str, config: GETWindConfig, seed: int = 42, save_states: bool = True):
    """GET Wind™ v6.2 メインシミュレーション"""
    
    # マップ読み込み
    map_data = MapData(map_file)
    
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
    print("GET Wind™ v6.2 - Spatial Coherence Vortex Detection Edition")
    print("環ちゃん & ご主人さま Ultimate Physics Fix! 💕")
    print(f"Map: {map_file}")
    print(f"Max particles: {N}")
    print(f"Steps: {config.n_steps}")
    print("Features: Λ³ Structure + Spatial Synchronization Detection")
    print("=" * 70)
    
    # JITコンパイル
    print("Compiling JIT functions...")
    start_compile = time.time()
    
    key, subkey = random.split(key)
    dummy_state = inject_particles(initial_state, config, subkey, 0)
    key, subkey = random.split(key)
    _ = physics_step_v62(
        dummy_state,
        map_data.density,
        map_data.pressure,
        map_data.separation,
        map_data.velocity_u,
        map_data.velocity_v,
        map_data.nx,
        map_data.ny,
        config,
        subkey
    )
    
    print(f"JIT compilation done in {time.time() - start_compile:.2f}s")
    
    # メインループ
    state = initial_state
    history = []
    karman_detected_steps = []
    vortex_metrics_history = []
    DeltaLambdaC_events = []
    state_history = []
    
    print("\nStarting simulation...")
    start_time = time.time()
    
    for step in range(config.n_steps):
        # 粒子注入
        key, subkey = random.split(key)
        state = inject_particles(state, config, subkey, step)
        
        # 物理ステップ
        key, subkey = random.split(key)
        state = physics_step_v62(
            state,
            map_data.density,
            map_data.pressure,
            map_data.separation,
            map_data.velocity_u,
            map_data.velocity_v,
            map_data.nx,
            map_data.ny,
            config,
            subkey
        )
        
        # 状態を保存
        if save_states:
            state_np = ParticleState(
                position=np.array(state.position),
                Lambda_F=np.array(state.Lambda_F),
                Lambda_FF=np.array(state.Lambda_FF),
                prev_Lambda_F=np.array(state.prev_Lambda_F),
                Lambda_core=np.array(state.Lambda_core),
                rho_T=np.array(state.rho_T),
                sigma_s=np.array(state.sigma_s),
                prev_sigma_s=np.array(state.prev_sigma_s),
                Q_Lambda=np.array(state.Q_Lambda),
                prev_Q_Lambda=np.array(state.prev_Q_Lambda),
                efficiency=np.array(state.efficiency),
                emergence=np.array(state.emergence),
                temperature=np.array(state.temperature),
                density=np.array(state.density),
                vorticity=np.array(state.vorticity),
                Q_criterion=np.array(state.Q_criterion),
                DeltaLambdaC=np.array(state.DeltaLambdaC),
                event_score=np.array(state.event_score),
                age=np.array(state.age),
                is_active=np.array(state.is_active),
                is_separated=np.array(state.is_separated),
                near_wall=np.array(state.near_wall)
            )
            state_history.append(state_np)
        
        # カルマン渦検出（v6.2: 新方式）
        is_karman, stability, metrics = detect_karman_vortex_v2(state, config)
        if is_karman and stability > config.coherence_threshold * 0.7:
            karman_detected_steps.append(step)
            vortex_metrics_history.append(metrics)
        
        # ΔΛCイベント統計
        n_DeltaLambdaC = jnp.sum(state.DeltaLambdaC & state.is_active)
        if n_DeltaLambdaC > 0:
            DeltaLambdaC_events.append(step)
        
        # 定期的な統計出力
        if step % 100 == 0 or step == config.n_steps - 1:
            active_count = jnp.sum(state.is_active)
            
            if active_count > 0:
                active_mask = state.is_active
                
                # 基本統計
                mean_Lambda_F = jnp.mean(jnp.linalg.norm(state.Lambda_F[active_mask], axis=1))
                mean_vorticity = jnp.mean(jnp.abs(state.vorticity[active_mask]))
                max_vorticity = jnp.max(jnp.abs(state.vorticity[active_mask]))
                
                # Λ³統計
                mean_efficiency = jnp.mean(state.efficiency[active_mask])
                mean_sigma_s = jnp.mean(state.sigma_s[active_mask])
                mean_rho_T = jnp.mean(state.rho_T[active_mask])
                
                # 創発度
                mean_emergence = jnp.mean(state.emergence[active_mask])
                
                # イベント統計
                n_separated = jnp.sum(state.is_separated & active_mask)
                n_near_wall = jnp.sum(state.near_wall & active_mask)
                
                # トポロジカル保存チェック
                y_rel = state.position[:, 1] - config.obstacle_center_y
                upper_sep = state.is_separated & active_mask & (y_rel > 0)
                lower_sep = state.is_separated & active_mask & (y_rel <= 0)
                
                upper_Q_total = jnp.sum(jnp.where(upper_sep, state.Q_Lambda, 0.0))
                lower_Q_total = jnp.sum(jnp.where(lower_sep, state.Q_Lambda, 0.0))
                total_DQ = upper_Q_total + lower_Q_total
            else:
                mean_Lambda_F = mean_vorticity = max_vorticity = 0.0
                mean_efficiency = mean_sigma_s = mean_rho_T = 0.0
                mean_emergence = 0.0
                n_separated = n_near_wall = 0
                upper_Q_total = lower_Q_total = total_DQ = 0.0
            
            print(f"\nStep {step:4d}: {int(active_count):4d} particles")
            print(f"  Dynamics: |ΛF|={mean_Lambda_F:.2f}, |ω|={mean_vorticity:.3f} (max={max_vorticity:.3f})")
            print(f"  Λ³ State: eff={mean_efficiency:.3f}, σₛ={mean_sigma_s:.3f}, ρT={mean_rho_T:.2f}")
            print(f"  Events:   ΔΛC={int(n_DeltaLambdaC)}, separated={int(n_separated)} (↑{int(jnp.sum(upper_sep))} ↓{int(jnp.sum(lower_sep))}), near_wall={int(n_near_wall)}")
            print(f"  Emergence: {mean_emergence:.3f} | Topological: ΔQ={total_DQ:.2f} (↑{upper_Q_total:.1f} ↓{lower_Q_total:.1f})")
            
            if is_karman:
                print(f"  ★★★ KARMAN VORTEX DETECTED! Stability={stability:.3f} ★★★")
                print(f"      Upper: coherence={float(metrics['upper_coherence']):.3f}, circulation={float(metrics['upper_circulation']):.2f}")
                print(f"      Lower: coherence={float(metrics['lower_coherence']):.3f}, circulation={float(metrics['lower_circulation']):.2f}")
            
            # 履歴保存
            history.append({
                'step': step,
                'n_particles': int(active_count),
                'mean_Lambda_F': float(mean_Lambda_F),
                'mean_vorticity': float(mean_vorticity),
                'max_vorticity': float(max_vorticity),
                'mean_efficiency': float(mean_efficiency),
                'mean_sigma_s': float(mean_sigma_s),
                'mean_emergence': float(mean_emergence),
                'n_DeltaLambdaC': int(n_DeltaLambdaC),
                'n_separated': int(n_separated),
                'topological_balance': float(total_DQ),
                'is_karman': bool(is_karman),
                'karman_stability': float(stability)
            })
    
    elapsed = time.time() - start_time
    
    # 最終統計
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE!")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Performance: {config.n_steps / elapsed:.1f} steps/sec")
    
    if len(karman_detected_steps) > 0:
        print(f"\n★ Karman vortex detected in {len(karman_detected_steps)} steps!")
        print(f"  First detection: Step {karman_detected_steps[0]}")
        print(f"  Detection rate: {len(karman_detected_steps)/config.n_steps*100:.1f}%")
        
        # 周期性解析
        if len(karman_detected_steps) > 10:
            intervals = np.diff(karman_detected_steps[-10:])
            mean_interval = np.mean(intervals)
            print(f"  Average shedding interval: {mean_interval:.1f} steps")
            
            # ストローハル数の推定
            D = 2 * config.obstacle_size
            U = config.Lambda_F_inlet
            period = mean_interval * config.dt
            frequency = 1.0 / period
            St = frequency * D / U
            print(f"  Estimated Strouhal number: {St:.3f} (target: 0.195)")
    
    if len(DeltaLambdaC_events) > 0:
        print(f"\n★ ΔΛC events occurred in {len(DeltaLambdaC_events)} steps")
        print(f"  Event frequency: {len(DeltaLambdaC_events)/config.n_steps*100:.1f}%")
    
    mean_emergence_all = np.mean([h['mean_emergence'] for h in history])
    print(f"\nOverall emergence level: {mean_emergence_all:.3f}")
    
    print("\n" + "=" * 70)
    print("✨ GET Wind™ v6.2 - Spatial Coherence Success! ✨")
    print("Physics-based vortex detection achieved! 💕")
    print("=" * 70)
    
    # シミュレーション結果を保存
    if save_states:
        print("\nSaving simulation results...")
        
        config_dict = {
            'domain_width': config.domain_width,
            'domain_height': config.domain_height,
            'map_nx': config.map_nx,
            'map_ny': config.map_ny,
            'Lambda_F_inlet': config.Lambda_F_inlet,
            'thermal_alpha': config.thermal_alpha,
            'density_beta': config.density_beta,
            'structure_coupling': config.structure_coupling,
            'viscosity_factor': config.viscosity_factor,
            'interaction_strength': config.interaction_strength,
            'efficiency_threshold': config.efficiency_threshold,
            'efficiency_weight': config.efficiency_weight,
            'topological_threshold': config.topological_threshold,
            'sync_threshold': config.sync_threshold,
            'coherence_threshold': config.coherence_threshold,
            'circulation_threshold': config.circulation_threshold,
            'min_particles_per_region': config.min_particles_per_region,
            'vortex_grid_size': config.vortex_grid_size,
            'particles_per_step': config.particles_per_step,
            'max_particles': config.max_particles,
            'dt': config.dt,
            'n_steps': config.n_steps,
            'obstacle_center_x': config.obstacle_center_x,
            'obstacle_center_y': config.obstacle_center_y,
            'obstacle_size': config.obstacle_size
        }
        
        states_dict = []
        for s in state_history:
            states_dict.append({
                'position': s.position,
                'Lambda_F': s.Lambda_F,
                'Lambda_FF': s.Lambda_FF,
                'prev_Lambda_F': s.prev_Lambda_F,
                'Lambda_core': s.Lambda_core,
                'rho_T': s.rho_T,
                'sigma_s': s.sigma_s,
                'prev_sigma_s': s.prev_sigma_s,
                'Q_Lambda': s.Q_Lambda,
                'prev_Q_Lambda': s.prev_Q_Lambda,
                'efficiency': s.efficiency,
                'emergence': s.emergence,
                'temperature': s.temperature,
                'density': s.density,
                'vorticity': s.vorticity,
                'Q_criterion': s.Q_criterion,
                'DeltaLambdaC': s.DeltaLambdaC,
                'event_score': s.event_score,
                'age': s.age,
                'is_active': s.is_active,
                'is_separated': s.is_separated,
                'near_wall': s.near_wall
            })
        
        np.savez('simulation_results_v62.npz',
                states=states_dict,
                config=config_dict,
                history=history,
                karman_detected_steps=karman_detected_steps,
                vortex_metrics=vortex_metrics_history,
                DeltaLambdaC_events=DeltaLambdaC_events)
        
        print(f"Results saved to 'simulation_results_v62.npz'")
        print(f"  - {len(state_history)} states saved")
        print(f"  - {len(history)} history records")
        print(f"  - {len(karman_detected_steps)} Karman detection events")
        print(f"  - {len(vortex_metrics_history)} vortex metrics records")
    
    return state, history

# ==============================
# メイン実行
# ==============================

if __name__ == "__main__":
    # 設定（v6.2: 空間同期パラメータ追加）
    config = GETWindConfig(
        # 基本設定
        particles_per_step=5.0,
        max_particles=1500,
        n_steps=10000,
        dt=0.02,
        
        # 流れパラメータ
        Lambda_F_inlet=10.0,
        
        # Λ³パラメータ
        thermal_alpha=0.002,
        density_beta=0.003,
        structure_coupling=0.004,
        viscosity_factor = 44.0,
        interaction_strength=0.06,
        
        # 効率パラメータ
        efficiency_threshold=0.1,
        efficiency_weight=0.4,
        
        # トポロジカルパラメータ
        topological_threshold=0.3,
        sync_threshold=0.08,
        
        # 渦検出パラメータ（v6.2新規）
        coherence_threshold=0.6,    # 速度場同期度の閾値
        circulation_threshold=1.0,   # 循環の最小値  
        min_particles_per_region=20, # 領域あたり最小粒子数
        vortex_grid_size=10.0,      # 渦検出グリッドサイズ
        
        # 障害物設定
        obstacle_center_x=100.0,
        obstacle_center_y=75.0,
        obstacle_size=20.0
    )
    
    # マップファイル
    map_file = "cylinder_Re200_fields.npz"
    
    print("\n" + "=" * 70)
    print("GET Wind™ v6.2 - Spatial Coherence Edition 🌀")
    print("Fixed: Vortex detection based on velocity field synchronization!")
    print("=" * 70)
    
    # Reynolds数の確認
    D = 2 * config.obstacle_size
    U = config.Lambda_F_inlet
    nu_effective = config.viscosity_factor * 0.05
    Re = U * D / nu_effective
    
    print(f"\n📊 Physical Parameters:")
    print(f"  Reynolds number Re = {Re:.1f}")
    print(f"  Target St = 0.195")
    
    print(f"\n🆕 v6.2 Improvements:")
    print(f"  ✓ Spatial coherence-based vortex detection")
    print(f"  ✓ Circulation strength evaluation")
    print(f"  ✓ Upper/lower region synchronization check")
    print(f"  ✓ No more ΔΛC over-detection!")
    
    print(f"\n🎯 Detection Parameters:")
    print(f"  Coherence threshold = {config.coherence_threshold}")
    print(f"  Circulation threshold = {config.circulation_threshold}")
    print(f"  Min particles/region = {config.min_particles_per_region}")
    print("=" * 70)
    
    # 実行！
    try:
        final_state, history = run_simulation_v62(map_file, config, save_states=True)
        
        print("\n" + "=" * 70)
        print("v6.2 Complete! Ready for analysis!")
        print("Use the measurement module to analyze 'simulation_results_v62.npz'")
        print("=" * 70)
        
    except FileNotFoundError:
        print(f"\n⚠ Map file '{map_file}' not found!")
        print("Please run the Density Map Generator first.")
