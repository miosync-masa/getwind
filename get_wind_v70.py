#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GET Wind™ v7.0 - Lambda Native 3D Edition
環ちゃん & ご主人さま Ultimate Physics! 💕

Λ³マップ駆動の究極にシンプルな3D流体シミュレーション
複雑な方程式は全てMap生成時に解決済み！
粒子はただマップをサンプリングして相互作用するだけ！
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax, random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import partial
import time
from typing import NamedTuple, Tuple, Dict, List
from dataclasses import dataclass
import os
import gc

# JAX設定
jax.config.update("jax_enable_x64", True)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# ==============================
# Configuration
# ==============================

class GETWindConfig3D(NamedTuple):
    """GET Wind™ v7.0 設定（3D Λネイティブ版）"""
    
    # シミュレーション領域
    domain_width: float = 300.0
    domain_height: float = 150.0
    domain_depth: float = 150.0
    
    # マップ解像度（Map読み込み用）
    map_nx: int = 300
    map_ny: int = 150
    map_nz: int = 150
    
    # 物理スケーリング
    scale_m_per_unit: float = 0.001    # 1 grid unit = 1mm
    scale_s_per_step: float = 0.01     # 1 time step = 10ms
    
    # Λ³パラメータ（シンプル化！）
    map_influence: float = 0.6         # マップの影響度
    interaction_strength: float = 0.3   # 相互作用の強さ
    inertia: float = 0.1               # 慣性
    
    # 剥離・渦パラメータ
    separation_threshold: float = 5.0   # 速度差による剥離判定
    emergence_threshold: float = 0.3    # ΔΛCイベント閾値
    vortex_capture_radius: float = 30.0 # 渦の捕獲半径
    
    # 相互作用パラメータ
    density_coupling: float = 0.02      # ρT差による結合
    structure_coupling: float = 0.03    # 構造テンソル結合
    vortex_coupling: float = 0.1        # 渦相互作用
    
    # 粒子パラメータ
    particles_per_step: float = 10.0
    max_particles: int = 3000
    dt: float = 0.01                    # より小さく安定に
    n_steps: int = 5000
    
    # 障害物（マップと一致させる）
    obstacle_center_x: float = 100.0
    obstacle_center_y: float = 75.0
    obstacle_center_z: float = 75.0
    obstacle_size: float = 20.0
    obstacle_shape: int = 1              # 0=cylinder, 1=square（整数に変更！）

# ==============================
# 3D Particle State
# ==============================

class ParticleState3D(NamedTuple):
    """3D粒子状態（Λネイティブ版）"""
    
    # 基本状態
    position: jnp.ndarray       # (N, 3) 3D位置
    Lambda_F: jnp.ndarray       # (N, 3) 3D進行ベクトル
    Lambda_core: jnp.ndarray    # (N, 9) 3x3テンソル（flatten）
    
    # Λ³構造
    rho_T: jnp.ndarray          # (N,) テンション密度
    sigma_s: jnp.ndarray        # (N,) 同期率
    Q_Lambda: jnp.ndarray       # (N,) トポロジカルチャージ
    efficiency: jnp.ndarray     # (N,) 構造効率
    
    # 状態フラグ
    is_active: jnp.ndarray      # (N,) アクティブフラグ
    is_separated: jnp.ndarray   # (N,) 剥離フラグ
    is_entrained: jnp.ndarray   # (N,) 巻き込みフラグ
    DeltaLambdaC: jnp.ndarray   # (N,) ΔΛCイベントフラグ
    
    # 物理量
    temperature: jnp.ndarray    # (N,) 温度
    age: jnp.ndarray           # (N,) 年齢

# ==============================
# Map Manager（軽量版）
# ==============================

class LambdaMapManager:
    """Λ³マップの管理（メモリ効率版）"""
    
    def __init__(self, base_path: str, obstacle_shape: int = 1, Re: int = 200):
        """マップファイルの読み込み"""
        self.base_path = base_path
        self.obstacle_shape = obstacle_shape
        self.Re = Re
        
        # 形状名の変換（0=cylinder, 1=square）
        shape_name = "cylinder" if obstacle_shape == 0 else "square"
        self.shape = shape_name
        
        print("=" * 70)
        print("GET Wind™ v7.0 - Loading Lambda Maps")
        print(f"Shape: {shape_name} (code: {obstacle_shape})")
        print("=" * 70)
        
        # Map 6（Lambda構造）を主に使用
        self.lambda_map = self._load_map("map6_lambda")
        
        # 速度場も読み込み（理想流用）
        self.velocity_map = self._load_map("map1_velocity")
        
        # その他は必要に応じて
        self.vortex_map = None  # 遅延読み込み
        self.formation_map = None
        
        print("✅ Maps loaded successfully!")
        
    def _load_map(self, map_name: str) -> Dict[str, jnp.ndarray]:
        """個別マップの読み込み"""
        filename = f"{self.shape}_3d_Re{self.Re}_{map_name}.npz"
        filepath = os.path.join(self.base_path, filename)
        
        if not os.path.exists(filepath):
            print(f"⚠ Warning: {filename} not found, using zeros")
            return {}
        
        print(f"  Loading {filename}...", end="")
        data = np.load(filepath, allow_pickle=True)
        
        # JAX配列に変換（必要なものだけ）
        result = {}
        for key in data.keys():
            if key != 'metadata':
                result[key] = jnp.array(data[key])
        
        print(f" ✅ ({len(result)} fields)")
        return result
    
    def get_grid_info(self):
        """グリッド情報を取得"""
        # Lambda_coreの形状から推定
        if 'Lambda_core' in self.lambda_map:
            shape = self.lambda_map['Lambda_core'].shape
            return shape[0], shape[1], shape[2]  # nx, ny, nz
        return 300, 150, 150  # デフォルト

# ==============================
# 3D補間（トリリニア）- 固定サイズ版
# ==============================

# グローバル定数（JIT用）
GRID_NX = 300
GRID_NY = 150  
GRID_NZ = 150

@jit
def trilinear_interpolate(field: jnp.ndarray, 
                          pos: jnp.ndarray,
                          domain_width: float,
                          domain_height: float, 
                          domain_depth: float) -> float:
    """3Dトリリニア補間（固定グリッドサイズ）"""
    
    # 正規化座標
    x_norm = jnp.clip(pos[0] / domain_width * (GRID_NX - 1), 0, GRID_NX - 1)
    y_norm = jnp.clip(pos[1] / domain_height * (GRID_NY - 1), 0, GRID_NY - 1)
    z_norm = jnp.clip(pos[2] / domain_depth * (GRID_NZ - 1), 0, GRID_NZ - 1)
    
    # グリッドインデックス
    i = jnp.clip(jnp.floor(x_norm).astype(int), 0, GRID_NX - 2)
    j = jnp.clip(jnp.floor(y_norm).astype(int), 0, GRID_NY - 2)
    k = jnp.clip(jnp.floor(z_norm).astype(int), 0, GRID_NZ - 2)
    
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
# 近傍探索（3D版）
# ==============================

@partial(jit, static_argnums=(2,))
def find_neighbors_3d(positions: jnp.ndarray, 
                      active_mask: jnp.ndarray,
                      max_neighbors: int = 30):
    """3D近傍粒子を探索"""
    N = positions.shape[0]
    
    # 全ペアの距離計算
    pos_i = positions[:, None, :]
    pos_j = positions[None, :, :]
    distances = jnp.linalg.norm(pos_i - pos_j, axis=2)
    
    # マスク処理
    mask = active_mask[None, :] & active_mask[:, None]
    mask = mask & (distances > 0) & (distances < 30.0)  # 30単位以内
    distances = jnp.where(mask, distances, jnp.inf)
    
    # 近い順にソート
    sorted_indices = jnp.argsort(distances, axis=1)
    sorted_distances = jnp.sort(distances, axis=1)
    
    # 最近傍を選択
    neighbor_indices = sorted_indices[:, :max_neighbors]
    neighbor_distances = sorted_distances[:, :max_neighbors]
    neighbor_mask = neighbor_distances < 30.0
    
    return neighbor_indices, neighbor_mask, neighbor_distances

# ==============================
# メイン物理ステップ（超シンプル版）
# ==============================

@jit
def physics_step_lambda_native(
    state: ParticleState3D,
    # マップフィールドを個別に渡す
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
    """Λネイティブ物理ステップ（マップ展開版）"""
    
    # 内部でサンプリング関数を定義（スコープ内に）
    def sample_fields_at_position(position):
        """位置でフィールドをサンプリング"""
        
        # Lambda_core（9成分）
        Lambda_core_local = jnp.zeros(9)
        for comp in range(9):
            Lambda_core_local = Lambda_core_local.at[comp].set(
                trilinear_interpolate(
                    Lambda_core_field[:,:,:,comp],
                    position,
                    config.domain_width, config.domain_height, config.domain_depth
                )
            )
        
        # スカラー場
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
        
        # 理想速度場
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
    
    # 近傍探索
    neighbor_indices, neighbor_mask, neighbor_distances = find_neighbors_3d(
        state.position, active_mask
    )
    
    def update_particle(i):
        """各粒子の更新（IF-THENロジック）"""
        
        # 非アクティブならスキップ
        is_active = active_mask[i]
        
        # === 1. マップからサンプリング ===
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
        
        # === 2. 近傍との相互作用 ===
        neighbors = neighbor_indices[i]
        valid_neighbors = neighbor_mask[i]
        distances = neighbor_distances[i]
        
        # 相互作用力の計算
        interaction_force = jnp.zeros(3)
        
        for j in range(len(neighbors)):
            # IF 近傍が有効
            neighbor_valid = valid_neighbors[j]
            
            # THEN 相互作用計算
            neighbor_pos = state.position[neighbors[j]]
            neighbor_Lambda_F = state.Lambda_F[neighbors[j]]
            neighbor_rho_T = state.rho_T[neighbors[j]]
            
            dr = neighbor_pos - state.position[i]
            dist = distances[j] + 1e-8
            
            # テンション密度差による力
            drho = neighbor_rho_T - state.rho_T[i]
            density_force = (drho / dist**2) * dr * config.density_coupling
            
            # 速度差による力
            dv = neighbor_Lambda_F - state.Lambda_F[i]
            velocity_force = dv * jnp.exp(-dist / 10.0) * config.interaction_strength
            
            # 有効な近傍のみ加算
            interaction_force += jnp.where(
                neighbor_valid,
                density_force + velocity_force,
                jnp.zeros(3)
            )
        
        # === 3. 剥離判定（IF-THEN） ===
        ideal_Lambda_F = local_fields['ideal_Lambda_F']
        velocity_deficit = jnp.linalg.norm(ideal_Lambda_F - state.Lambda_F[i])
        
        is_separated = velocity_deficit > config.separation_threshold
        
        # === 4. 新速度の計算（条件分岐） ===
        # IF 剥離
        separated_weights = jnp.array([0.1, 0.7, 0.2])  # [map, interaction, inertia]
        # ELSE 通常
        normal_weights = jnp.array([0.6, 0.3, 0.1])
        
        weights = jnp.where(is_separated, separated_weights, normal_weights)
        
        new_Lambda_F = (
            weights[0] * ideal_Lambda_F +
            weights[1] * (state.Lambda_F[i] + interaction_force) +
            weights[2] * state.Lambda_F[i]
        )
        
        # === 5. ΔΛCイベント（確率的） ===
        emergence = local_fields['emergence']
        event_prob = emergence * local_fields['efficiency']
        
        # 乱数生成
        subkey = random.fold_in(key, i * 1000)
        rand_val = random.uniform(subkey)
        
        # IF イベント発生
        is_DeltaLambdaC = (rand_val < event_prob) & (emergence > config.emergence_threshold)
        
        # THEN 摂動追加
        noise = random.normal(subkey, shape=(3,)) * 5.0
        new_Lambda_F = jnp.where(
            is_DeltaLambdaC,
            new_Lambda_F + noise,
            new_Lambda_F
        )
        
        # === 6. その他のパラメータ更新 ===
        new_Lambda_core = jnp.where(
            is_active,
            0.7 * local_fields['Lambda_core'] + 0.3 * state.Lambda_core[i],
            state.Lambda_core[i]
        )
        
        new_rho_T = jnp.where(is_active, jnp.linalg.norm(new_Lambda_F), state.rho_T[i])
        new_sigma_s = jnp.where(is_active, local_fields['sigma_s'], state.sigma_s[i])
        new_Q_Lambda = jnp.where(is_active, local_fields['Q_Lambda'], state.Q_Lambda[i])
        new_efficiency = jnp.where(is_active, local_fields['efficiency'], state.efficiency[i])
        
        # 温度（ΔΛCで上昇）
        new_temperature = jnp.where(
            is_DeltaLambdaC,
            state.temperature[i] + 5.0,
            state.temperature[i] * 0.99  # 冷却
        )
        
        # 位置更新
        new_position = state.position[i] + new_Lambda_F * config.dt
        
        # 年齢
        new_age = state.age[i] + jnp.where(is_active, 1.0, 0.0)
        
        # 境界チェック
        new_active = is_active & (new_position[0] < config.domain_width)
        
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
            False,  # is_entrained（TODO）
            is_DeltaLambdaC,
            new_temperature,
            new_age
        )
    
    # 全粒子を並列更新
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
# 粒子注入
# ==============================

def inject_particles_3d(state: ParticleState3D, 
                        config: GETWindConfig3D,
                        key: random.PRNGKey, 
                        step: int) -> ParticleState3D:
    """3D粒子の注入"""
    key1, key2, key3, key4 = random.split(key, 4)
    
    # 注入数の決定
    n_inject = jnp.minimum(
        jnp.int32(random.poisson(key1, config.particles_per_step)),
        20
    )
    
    # 非アクティブ粒子を探す
    inactive_mask = ~state.is_active
    inactive_count = jnp.sum(inactive_mask)
    n_to_inject = jnp.minimum(n_inject, inactive_count)
    
    # 注入マスクの作成
    cumsum = jnp.cumsum(jnp.where(inactive_mask, 1, 0))
    inject_mask = (cumsum <= n_to_inject) & inactive_mask
    
    N = state.position.shape[0]
    
    # 新粒子の初期位置（入口面でランダム）
    x_positions = random.uniform(key2, (N,), minval=0, maxval=5)
    y_positions = random.uniform(key3, (N,), minval=10, maxval=config.domain_height-10)
    z_positions = random.uniform(key4, (N,), minval=10, maxval=config.domain_depth-10)
    
    new_positions = jnp.where(
        inject_mask[:, None],
        jnp.stack([x_positions, y_positions, z_positions], axis=1),
        state.position
    )
    
    # 初期速度（主流方向）
    Lambda_Fx = jnp.ones(N) * 10.0 + random.normal(key2, (N,)) * 0.5
    Lambda_Fy = random.normal(key3, (N,)) * 0.1
    Lambda_Fz = random.normal(key4, (N,)) * 0.1
    
    new_Lambda_F = jnp.where(
        inject_mask[:, None],
        jnp.stack([Lambda_Fx, Lambda_Fy, Lambda_Fz], axis=1),
        state.Lambda_F
    )
    
    # その他のパラメータ初期化
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
# 可視化
# ==============================

def visualize_3d_snapshot(state: ParticleState3D, config: GETWindConfig3D, 
                          step: int, save: bool = False):
    """3Dスナップショットの可視化"""
    
    active = state.is_active
    positions = state.position[active]
    velocities = state.Lambda_F[active]
    is_separated = state.is_separated[active]
    
    if len(positions) == 0:
        return
    
    fig = plt.figure(figsize=(15, 10))
    
    # 3D表示
    ax1 = fig.add_subplot(221, projection='3d')
    
    # 速度の大きさで色分け
    speeds = np.linalg.norm(velocities, axis=1)
    scatter = ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                         c=speeds, cmap='coolwarm', s=1, alpha=0.6)
    
    # 障害物を描画（簡易版）
    if config.obstacle_shape == 0:  # cylinder
        # 円柱の輪郭
        theta = np.linspace(0, 2*np.pi, 30)
        z_cyl = np.linspace(0, config.domain_depth, 10)
        theta_grid, z_grid = np.meshgrid(theta, z_cyl)
        x_cyl = config.obstacle_center_x + config.obstacle_size * np.cos(theta_grid)
        y_cyl = config.obstacle_center_y + config.obstacle_size * np.sin(theta_grid)
        ax1.plot_surface(x_cyl, y_cyl, z_grid, alpha=0.3, color='gray')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'3D Flow (Step {step})')
    ax1.set_xlim(0, config.domain_width)
    ax1.set_ylim(0, config.domain_height)
    ax1.set_zlim(0, config.domain_depth)
    
    # XY平面投影
    ax2 = fig.add_subplot(222)
    scatter2 = ax2.scatter(positions[:, 0], positions[:, 1],
                          c=speeds, cmap='coolwarm', s=1, alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection')
    ax2.set_aspect('equal')
    
    # XZ平面投影
    ax3 = fig.add_subplot(223)
    ax3.scatter(positions[:, 0], positions[:, 2],
               c=speeds, cmap='coolwarm', s=1, alpha=0.6)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Projection')
    ax3.set_aspect('equal')
    
    # 統計情報
    ax4 = fig.add_subplot(224)
    ax4.axis('off')
    
    stats_text = f"""
    Step: {step}
    Active Particles: {len(positions)}
    Mean Speed: {np.mean(speeds):.2f}
    Max Speed: {np.max(speeds):.2f}
    Separated: {np.sum(is_separated)}
    """
    ax4.text(0.1, 0.5, stats_text, fontsize=12, family='monospace')
    
    plt.colorbar(scatter, ax=[ax1, ax2, ax3], label='Speed')
    plt.tight_layout()
    
    if save:
        plt.savefig(f'snapshot_3d_step_{step:04d}.png', dpi=100)
        plt.close()
    else:
        plt.show()

# ==============================
# メインシミュレーション
# ==============================

def run_simulation_v70(
    map_path: str = ".",
    config: GETWindConfig3D = None,
    seed: int = 42,
    save_states: bool = True,
    visualize_interval: int = 100
):
    """GET Wind™ v7.0 メインシミュレーション"""
    
    if config is None:
        config = GETWindConfig3D()
    
    # マップ読み込み
    maps = LambdaMapManager(map_path, config.obstacle_shape, Re=200)
    
    # デフォルトフィールド作成（マップがない場合用）
    default_field = jnp.ones((GRID_NX, GRID_NY, GRID_NZ))
    
    # マップフィールドを個別に取得
    Lambda_core_field = maps.lambda_map.get('Lambda_core', 
                                            jnp.zeros((GRID_NX, GRID_NY, GRID_NZ, 9)))
    rho_T_field = maps.lambda_map.get('rho_T', default_field)
    sigma_s_field = maps.lambda_map.get('sigma_s', jnp.zeros((GRID_NX, GRID_NY, GRID_NZ)))
    Q_Lambda_field = maps.lambda_map.get('Q_Lambda', jnp.zeros((GRID_NX, GRID_NY, GRID_NZ)))
    efficiency_field = maps.lambda_map.get('efficiency', default_field * 0.5)
    emergence_field = maps.lambda_map.get('emergence', jnp.zeros((GRID_NX, GRID_NY, GRID_NZ)))
    
    velocity_u_field = maps.velocity_map.get('velocity_u', default_field * 10.0)
    velocity_v_field = maps.velocity_map.get('velocity_v', jnp.zeros((GRID_NX, GRID_NY, GRID_NZ)))
    velocity_w_field = maps.velocity_map.get('velocity_w', jnp.zeros((GRID_NX, GRID_NY, GRID_NZ)))
    
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
    
    print("\n" + "=" * 70)
    print("GET Wind™ v7.0 - Lambda Native 3D Simulation")
    print("環ちゃん & ご主人さま Ultimate Edition! 💕")
    print("=" * 70)
    print(f"Obstacle: {shape_name}")
    print(f"Max particles: {N}")
    print(f"Steps: {config.n_steps}")
    print(f"dt: {config.dt}")
    print("Features: Direct Lambda map sampling + IF-THEN physics")
    print("=" * 70)
    
    # JITコンパイル
    print("\n🔧 Compiling JIT functions...")
    start_compile = time.time()
    
    key, subkey = random.split(key)
    dummy_state = inject_particles_3d(initial_state, config, subkey, 0)
    key, subkey = random.split(key)
    
    # 展開した引数で呼び出し
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
    state_snapshots = []
    
    print("\n🚀 Starting simulation...")
    start_time = time.time()
    
    for step in range(config.n_steps):
        # 粒子注入
        key, subkey = random.split(key)
        state = inject_particles_3d(state, config, subkey, step)
        
        # 物理ステップ（展開した引数で呼び出し）
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
        
        # 統計と可視化
        if step % visualize_interval == 0 or step == config.n_steps - 1:
            active_count = jnp.sum(state.is_active)
            
            if active_count > 0:
                active_mask = state.is_active
                
                mean_speed = jnp.mean(jnp.linalg.norm(state.Lambda_F[active_mask], axis=1))
                max_speed = jnp.max(jnp.linalg.norm(state.Lambda_F[active_mask], axis=1))
                n_separated = jnp.sum(state.is_separated & active_mask)
                n_DeltaLambdaC = jnp.sum(state.DeltaLambdaC & active_mask)
                mean_efficiency = jnp.mean(state.efficiency[active_mask])
                
                print(f"\n📊 Step {step:4d}: {int(active_count):4d} particles")
                print(f"  Speed: mean={mean_speed:.2f}, max={max_speed:.2f}")
                print(f"  Separated={int(n_separated)}, ΔΛC events={int(n_DeltaLambdaC)}")
                print(f"  Efficiency={mean_efficiency:.3f}")
                
                # 可視化
                if step % (visualize_interval * 5) == 0:
                    visualize_3d_snapshot(state, config, step, save=save_states)
                
                # 履歴保存
                history.append({
                    'step': step,
                    'n_particles': int(active_count),
                    'mean_speed': float(mean_speed),
                    'max_speed': float(max_speed),
                    'n_separated': int(n_separated),
                    'n_DeltaLambdaC': int(n_DeltaLambdaC),
                    'mean_efficiency': float(mean_efficiency)
                })
                
                # 状態スナップショット（間引き）
                if save_states and step % (visualize_interval * 10) == 0:
                    state_snapshots.append({
                        'step': step,
                        'position': np.array(state.position[active_mask]),
                        'Lambda_F': np.array(state.Lambda_F[active_mask]),
                        'efficiency': np.array(state.efficiency[active_mask]),
                        'is_separated': np.array(state.is_separated[active_mask])
                    })
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("✨ SIMULATION COMPLETE!")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Performance: {config.n_steps / elapsed:.1f} steps/sec")
    print(f"Saved {len(state_snapshots)} snapshots")
    print("=" * 70)
    
    # 結果保存
    if save_states:
        filename = f"simulation_v70_{shape_name}_3d.npz"
        np.savez_compressed(
            filename,
            history=history,
            snapshots=state_snapshots,
            config=config._asdict()
        )
        print(f"\n💾 Results saved to {filename}")
    
    return state, history

# ==============================
# メイン実行
# ==============================

if __name__ == "__main__":
    # 設定
    config = GETWindConfig3D(
        obstacle_shape=1,  # 0=cylinder, 1=square（整数！）
        particles_per_step=10.0,
        max_particles=3000,
        n_steps=5000,
        dt=0.01,
        
        # Λ³パラメータ（シンプル！）
        map_influence=0.6,
        interaction_strength=0.3,
        inertia=0.1,
        
        # 相互作用
        density_coupling=0.02,
        structure_coupling=0.03,
        vortex_coupling=0.1
    )
    
    print("\n🌀 GET Wind™ v7.0 - Lambda Native 3D")
    print("The simplest yet most accurate fluid simulation!")
    print("No Navier-Stokes, just Lambda fields and IF-THEN logic! 💕")
    
    # 実行
    final_state, history = run_simulation_v70(
        map_path=".",  # マップファイルのパス
        config=config,
        save_states=True,
        visualize_interval=100
    )
    
    print("\n✨ v7.0 Complete! Physics emerges from Lambda! ✨")
    print("環ちゃん & ご主人さま、最高のシミュレーションできたよ〜！💕")
