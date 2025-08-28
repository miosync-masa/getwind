#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometric Bernoulli Map 3D Generator for GET Wind™
完全3次元幾何学的構造マップ生成器
～2Dの限界を超えて、真の物理を駆動する～

環ちゃん & ご主人さま Ultimate 3D Edition! 💕
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import json
import time
from concurrent.futures import ThreadPoolExecutor
import gc

# 追加インポート（PCG用）
try:
    from scipy.fft import dctn, idctn
    HAS_DCT = True
except ImportError:
    HAS_DCT = False
    print("Warning: scipy.fft not available, using fallback solver")

# ==============================
# Configuration Classes (3D Extended)
# ==============================

@dataclass
class Obstacle3DConfig:
    """3D障害物の設定"""
    shape_type: str  # 'cylinder', 'square', 'sphere', 'cube'
    center_x: float = 100.0
    center_y: float = 75.0
    center_z: float = 75.0  # Z中心追加
    size: float = 20.0  # 特性長さ
    span: float = 150.0  # スパン長さ（円柱・角柱の場合）
    angle_of_attack: float = 0.0  # 迎角[deg]
    angle_beta: float = 0.0  # Z方向傾き角[deg]（3D迎角）
    
@dataclass
class Flow3DConfig:
    """3D流れの条件"""
    U_inf: float = 0.015    # 一様流速度 [m/s] (Re=200, D=20cm)
    V_inf: float = 0.0       # Y方向速度成分
    W_inf: float = 0.0       # Z方向速度成分
    rho_inf: float = 1.225   # 基準密度 [kg/m³]
    Re: float = 200.0        # Reynolds数
    temperature_inf: float = 293.0  # 基準温度 [K]
    
    # 物理定数
    nu_air: float = 1.5e-5   # 空気の動粘性係数 [m²/s] at 20℃

@dataclass
class Grid3DConfig:
    """3D計算グリッド設定"""
    nx: int = 300
    ny: int = 150
    nz: int = 150  # Z方向追加
    x_min: float = 0.0
    x_max: float = 300.0
    y_min: float = 0.0
    y_max: float = 150.0
    z_min: float = 0.0
    z_max: float = 150.0
    
    # 物理スケーリング
    scale_m_per_unit: float = 0.01   # 1グリッド単位 = 1cm (10倍スケール)
    scale_s_per_step: float = 0.01   # 1ステップ = 0.01秒
    
    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.nx - 1)
    
    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / (self.ny - 1)
    
    @property
    def dz(self) -> float:
        return (self.z_max - self.z_min) / (self.nz - 1)

# ==============================
# 3D Geometric Bernoulli Calculator
# ==============================

class GeometricBernoulli3D:
    """3次元幾何学的ベルヌーイ場の計算器"""
    
    def __init__(self, obstacle: Obstacle3DConfig, 
                 flow: Flow3DConfig, grid: Grid3DConfig):
        self.obstacle = obstacle
        self.flow = flow
        self.grid = grid
        
        print("=" * 70)
        print("GET Wind™ 3D Geometric Bernoulli Map Generator")
        print("環ちゃん & ご主人さま Ultimate 3D Edition! 💕")
        print("=" * 70)
        print(f"\nInitializing 3D grid: {grid.nx}×{grid.ny}×{grid.nz}")
        print(f"Memory estimate: ~{self._estimate_memory():.1f} GB")
        
        # 3Dグリッド生成
        self.x = np.linspace(grid.x_min, grid.x_max, grid.nx)
        self.y = np.linspace(grid.y_min, grid.y_max, grid.ny)
        self.z = np.linspace(grid.z_min, grid.z_max, grid.nz)
        
        # メモリ効率のため、必要時にのみmeshgridを生成
        self._X = None
        self._Y = None
        self._Z = None
        
        # 物理パラメータ
        self.physics_params = self._get_3d_physics_params()
        
    def _estimate_memory(self) -> float:
        """メモリ使用量の推定[GB]"""
        n_points = self.grid.nx * self.grid.ny * self.grid.nz
        # 各フィールド8バイト(float64) × フィールド数
        n_fields = 20  # 概算
        return n_points * 8 * n_fields / (1024**3)
    
    @property
    def X(self):
        if self._X is None:
            self._X, self._Y, self._Z = np.meshgrid(
                self.x, self.y, self.z, indexing='ij'
            )
        return self._X
    
    @property
    def Y(self):
        if self._Y is None:
            self._X, self._Y, self._Z = np.meshgrid(
                self.x, self.y, self.z, indexing='ij'
            )
        return self._Y
    
    @property
    def Z(self):
        if self._Z is None:
            self._X, self._Y, self._Z = np.meshgrid(
                self.x, self.y, self.z, indexing='ij'
            )
        return self._Z
    
    def _get_3d_physics_params(self) -> Dict:
        """3D形状別の物理パラメータ"""
        if self.obstacle.shape_type == 'cylinder':
            return {
                'separation_angle': np.pi/2.2,
                'wake_width_factor': 1.5,
                'vortex_formation_length': 2.0,
                'strouhal_number_2d': 0.195,
                'base_pressure': -0.9,
                'recovery_length': 10.0,
                'spanwise_correlation_length': 3.0,  # スパン方向相関長
                'cell_aspect_ratio': 5.0,  # カルマン渦セルのアスペクト比
                'horseshoe_strength': 0.3,  # 馬蹄渦の強さ
            }
        elif self.obstacle.shape_type == 'square':
            return {
                'separation_angle': np.pi/2,  # 90度固定剥離
                'wake_width_factor': 2.2,
                'vortex_formation_length': 1.5,
                'strouhal_number_2d': 0.14,
                'base_pressure': -1.4,
                'recovery_length': 15.0,
                'spanwise_correlation_length': 2.0,
                'cell_aspect_ratio': 3.0,
                'horseshoe_strength': 0.5,  # 角柱は馬蹄渦が強い
                'oblique_angle': 15.0,  # 斜め渦放出角度[deg]
            }
        elif self.obstacle.shape_type == 'sphere':
            return {
                'separation_angle': np.pi * 0.6,  # 約108度
                'wake_width_factor': 1.2,
                'vortex_formation_length': 1.5,
                'strouhal_number': 0.18,
                'base_pressure': -0.4,
                'recovery_length': 8.0,
            }
        else:
            raise ValueError(f"Unknown shape: {self.obstacle.shape_type}")
    
    def calculate_all_fields(self) -> Dict[str, Dict[str, np.ndarray]]:
        """全ての3D場を計算（メモリ効率版）"""
        
        print("\n🔄 Starting 3D field calculations...")
        start_time = time.time()
        
        # 結果を格納する辞書
        all_maps = {}
        
        # === Map 1: ベクトルポテンシャル＆速度場 ===
        print("\n📊 Map 1: Vector Potential & Velocity Field...")
        t0 = time.time()
        all_maps['map1_velocity'] = self._calculate_velocity_map()
        print(f"  ✅ Completed in {time.time()-t0:.1f}s")
        gc.collect()  # メモリ解放
        
        # === Map 2: 圧力・密度・温度場 ===
        print("\n📊 Map 2: Pressure, Density & Temperature...")
        t0 = time.time()
        all_maps['map2_thermo'] = self._calculate_thermo_map(
            all_maps['map1_velocity']
        )
        print(f"  ✅ Completed in {time.time()-t0:.1f}s")
        gc.collect()
        
        # === Map 3: 渦構造テンソル ===
        print("\n📊 Map 3: Vortex Tensor Fields...")
        t0 = time.time()
        all_maps['map3_vortex'] = self._calculate_vortex_map(
            all_maps['map1_velocity']
        )
        print(f"  ✅ Completed in {time.time()-t0:.1f}s")
        gc.collect()
        
        # === Map 4: 境界層・剥離構造 ===
        print("\n📊 Map 4: Boundary Layer & Separation...")
        t0 = time.time()
        all_maps['map4_boundary'] = self._calculate_boundary_map()
        print(f"  ✅ Completed in {time.time()-t0:.1f}s")
        gc.collect()
        
        # === Map 5: 3D渦形成領域 ===
        print("\n📊 Map 5: 3D Vortex Formation Zones...")
        t0 = time.time()
        all_maps['map5_formation'] = self._calculate_formation_map()
        print(f"  ✅ Completed in {time.time()-t0:.1f}s")
        gc.collect()
        
        # === Map 6: Λ³構造場 ===
        print("\n📊 Map 6: Lambda Structural Fields...")
        t0 = time.time()
        all_maps['map6_lambda'] = self._calculate_lambda_map(
            all_maps['map1_velocity'],
            all_maps['map2_thermo']
        )
        print(f"  ✅ Completed in {time.time()-t0:.1f}s")
        
        total_time = time.time() - start_time
        print(f"\n✨ All fields calculated in {total_time:.1f}s!")
        
        return all_maps
    
    def _calculate_velocity_map(self) -> Dict[str, np.ndarray]:
        """Map 1: 理想流速度場の計算（発散ゼロ保証）"""
        cx, cy, cz = self.obstacle.center_x, self.obstacle.center_y, self.obstacle.center_z
        R = self.obstacle.size
        U = self.flow.U_inf
        
        # === STEP 1: 純粋な理想流（ポテンシャル流）===
        u_ideal = np.zeros((self.grid.nx, self.grid.ny, self.grid.nz))
        v_ideal = np.zeros_like(u_ideal)
        w_ideal = np.zeros_like(u_ideal)
        
        if self.obstacle.shape_type == 'cylinder':
            # 円柱まわりの2Dポテンシャル流（迎角対応）
            print("    Computing ideal potential flow around cylinder...")
            
            # 迎角から速度成分を計算
            alpha_rad = np.deg2rad(self.obstacle.angle_of_attack)
            beta_rad = np.deg2rad(self.obstacle.angle_beta)
            Ux = U * np.cos(alpha_rad) * np.cos(beta_rad)
            Uy = U * np.sin(alpha_rad) * np.cos(beta_rad)
            Uz = U * np.sin(beta_rad)
            
            if abs(alpha_rad) > 1e-3 or abs(beta_rad) > 1e-3:
                print(f"    Flow with angle: α={self.obstacle.angle_of_attack}°, β={self.obstacle.angle_beta}°")
                print(f"    Components: Ux={Ux:.3f}, Uy={Uy:.3f}, Uz={Uz:.3f} m/s")
            
            for i in range(self.grid.nx):
                for j in range(self.grid.ny):
                    dx = self.x[i] - cx
                    dy = self.y[j] - cy
                    r = np.sqrt(dx**2 + dy**2)
                    
                    if r > R * 1.01:  # 表面から少し離す（数値安定性）
                        # 迎角を考慮した座標系での解析解
                        # 流入方向をx'軸とした座標系での計算
                        x_prime = dx * np.cos(alpha_rad) + dy * np.sin(alpha_rad)
                        y_prime = -dx * np.sin(alpha_rad) + dy * np.cos(alpha_rad)
                        theta_prime = np.arctan2(y_prime, x_prime)
                        
                        # x'-y'座標系での速度（円柱理想流）
                        U_mag = np.sqrt(Ux**2 + Uy**2)  # XY平面内の流速
                        u_prime = U_mag * (1 - (R/r)**2 * np.cos(2*theta_prime))
                        v_prime = -U_mag * (R/r)**2 * np.sin(2*theta_prime)
                        
                        # 元の座標系に戻す
                        u_ideal[i,j,:] = u_prime * np.cos(alpha_rad) - v_prime * np.sin(alpha_rad)
                        v_ideal[i,j,:] = u_prime * np.sin(alpha_rad) + v_prime * np.cos(alpha_rad)
                        w_ideal[i,j,:] = Uz  # Z成分は一様
                    else:
                        # 円柱内部は速度ゼロ
                        u_ideal[i,j,:] = 0
                        v_ideal[i,j,:] = 0
                        w_ideal[i,j,:] = 0
                    
        elif self.obstacle.shape_type == 'square':
            # 角柱：数値ラプラシアン解法で厳密なポテンシャル流
            print("    Computing potential flow around square obstacle...")
            print("    Using numerical Laplace solver with ghost cells...")
            
            # レベルセット関数の構築（符号付き距離）
            level_set = self._compute_level_set_square(cx, cy, cz, R)
            
            # φ = U·x + ϕ の変数置換でϕを解く（迎角対応）
            phi_perturbation, Ux, Uy, Uz = self._solve_laplace_square(level_set, U)
            
            # φ = U·x + ϕ の再構成
            print("    Reconstructing potential φ = U·x + ϕ...")
            # 座標グリッド（グリッド単位）
            X_grid, Y_grid, Z_grid = np.meshgrid(
                self.x, self.y, self.z, indexing='ij'
            )
            
            # φ = Ux*x + Uy*y + Uz*z + ϕ
            phi = Ux * X_grid + Uy * Y_grid + Uz * Z_grid + phi_perturbation
            
            # 速度 u = ∇φ
            print("    Computing velocities from potential...")
            u_ideal = np.gradient(phi, self.grid.dx, axis=0)
            v_ideal = np.gradient(phi, self.grid.dy, axis=1)
            w_ideal = np.gradient(phi, self.grid.dz, axis=2)
            
            # 障害物内部は速度ゼロに強制
            mask = level_set < 0  # 内部
            u_ideal[mask] = 0
            v_ideal[mask] = 0
            w_ideal[mask] = 0
        
        # === STEP 2: 3D効果の追加（オプション）===
        # 純粋な理想流では3D効果なし（ポテンシャル流保持）
        
        # === STEP 3: 発散チェック（理想流は自動的に満たす）===
        if self.obstacle.shape_type == 'cylinder':
            # 解析的ポテンシャル流は自動的に∇·u = 0
            print("    Cylinder: Analytically divergence-free")
            u, v, w = u_ideal, v_ideal, w_ideal
        elif self.obstacle.shape_type == 'square':
            # 数値解もポテンシャル流なので∇·u = 0
            print("    Square: Numerically divergence-free (from potential)")
            u, v, w = u_ideal, v_ideal, w_ideal
            
        # === STEP 4: 発散チェック ===
        div_u = self._compute_divergence(u, v, w)
        div_norm = np.sqrt(np.mean(div_u**2))
        div_max = np.max(np.abs(div_u))
        print(f"    Divergence check: ||∇·u||₂ = {div_norm:.2e}, max = {div_max:.2e}")
        
        # ベクトルポテンシャル（将来の実装用）
        psi_x = np.zeros_like(u)
        psi_y = np.zeros_like(u) 
        psi_z = np.zeros_like(u)
        
        return {
            'velocity_u': u,
            'velocity_v': v,
            'velocity_w': w,
            'vector_potential_x': psi_x,
            'vector_potential_y': psi_y,
            'vector_potential_z': psi_z
        }
    
    def _world_to_body(self, X, Y, Z):
        """ワールド座標系から物体座標系への変換（回転対応）"""
        cx, cy, cz = self.obstacle.center_x, self.obstacle.center_y, self.obstacle.center_z
        
        # 回転角（物体の姿勢）
        # 注：angle_of_attackは流れの向きなので、物体回転とは別に必要なら追加
        alpha = np.deg2rad(0.0)  # Z軸回りの回転（yaw）
        beta = np.deg2rad(0.0)   # Y軸回りの回転（pitch）
        # 必要ならrollも追加可能
        
        # 回転行列
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        
        Rz = np.array([[ca, -sa, 0],
                       [sa,  ca, 0],
                       [0,   0,  1]])
        
        Ry = np.array([[cb,  0, sb],
                       [0,   1, 0],
                       [-sb, 0, cb]])
        
        R = Ry @ Rz
        
        # 中心からの相対位置
        dx = X - cx
        dy = Y - cy
        dz = Z - cz
        
        # 回転適用
        xb = R[0,0]*dx + R[0,1]*dy + R[0,2]*dz
        yb = R[1,0]*dx + R[1,1]*dy + R[1,2]*dz
        zb = R[2,0]*dx + R[2,1]*dy + R[2,2]*dz
        
        return xb, yb, zb
    
    def _compute_level_set_square(self, cx: float, cy: float, cz: float, 
                                  R: float) -> np.ndarray:
        """角柱のレベルセット関数（符号付き距離）を計算
        
        内部: < 0, 外部: > 0
        物体回転対応版
        """
        level_set = np.zeros((self.grid.nx, self.grid.ny, self.grid.nz))
        
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                for k in range(self.grid.nz):
                    x, y, z = self.x[i], self.y[j], self.z[k]
                    
                    # 物体座標系に変換
                    xb, yb, zb = self._world_to_body(x, y, z)
                    
                    # 角柱の半寸法
                    hx = R
                    hy = R  
                    hz = self.obstacle.span / 2
                    
                    # 符号付き距離（物体座標系で計算）
                    dx = np.abs(xb) - hx
                    dy = np.abs(yb) - hy
                    dz = np.abs(zb) - hz
                    
                    # 外部距離
                    if dx > 0 and dy > 0 and dz > 0:
                        # コーナー領域（3D）
                        level_set[i,j,k] = np.sqrt(dx**2 + dy**2 + dz**2)
                    elif dx > 0 and dy > 0:
                        # エッジ領域（XY）
                        level_set[i,j,k] = np.sqrt(dx**2 + dy**2)
                    elif dy > 0 and dz > 0:
                        # エッジ領域（YZ）
                        level_set[i,j,k] = np.sqrt(dy**2 + dz**2)
                    elif dx > 0 and dz > 0:
                        # エッジ領域（XZ）
                        level_set[i,j,k] = np.sqrt(dx**2 + dz**2)
                    elif dx > 0:
                        level_set[i,j,k] = dx
                    elif dy > 0:
                        level_set[i,j,k] = dy
                    elif dz > 0:
                        level_set[i,j,k] = dz
                    else:
                        # 内部（最も近い面までの距離）
                        level_set[i,j,k] = max(dx, dy, dz)
        
        return level_set
    
    def _solve_laplace_square(self, level_set: np.ndarray, U_inf: float) -> np.ndarray:
        """角柱まわりのラプラス方程式を解く（DCT前処理付きPCG法、迎角対応）
        
        φ = U·x + ϕ の変数置換で、∇²ϕ = 0 を解く
        外枠: ∂ϕ/∂n = 0 (同次Neumann)
        角柱: ∂ϕ/∂n = -U·n (非同次Neumann)
        """
        print("      Setting up Laplace equation with ghost cells...")
        
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        dx, dy, dz = self.grid.dx, self.grid.dy, self.grid.dz
        cx, cy, cz = self.obstacle.center_x, self.obstacle.center_y, self.obstacle.center_z
        R = self.obstacle.size
        
        # === 0. 迎角から速度成分を計算 ===
        alpha_rad = np.deg2rad(self.obstacle.angle_of_attack)
        beta_rad = np.deg2rad(self.obstacle.angle_beta)
        Ux = U_inf * np.cos(alpha_rad) * np.cos(beta_rad)
        Uy = U_inf * np.sin(alpha_rad) * np.cos(beta_rad)
        Uz = U_inf * np.sin(beta_rad)
        
        print(f"      Flow components: Ux={Ux:.3f}, Uy={Uy:.3f}, Uz={Uz:.3f} m/s")
        print(f"      Angle of attack: α={self.obstacle.angle_of_attack}°, β={self.obstacle.angle_beta}°")
        
        # === 1. マスクの構築 ===
        solid_mask = level_set < 0
        fluid_mask = ~solid_mask
        
        # 流体セルで隣接が固体かをチェック
        def get_solid_neighbors(solid, axis):
            plus = np.roll(solid, -1, axis=axis)
            minus = np.roll(solid, +1, axis=axis)
            # 端はFalse（外枠はNeumann）
            if axis == 0:
                plus[-1,:,:] = False
                minus[0,:,:] = False
            elif axis == 1:
                plus[:,-1,:] = False
                minus[:,0,:] = False
            else:
                plus[:,:,-1] = False
                minus[:,:,0] = False
            return plus & fluid_mask, minus & fluid_mask
        
        solid_plus_x, solid_minus_x = get_solid_neighbors(solid_mask, 0)
        solid_plus_y, solid_minus_y = get_solid_neighbors(solid_mask, 1)
        solid_plus_z, solid_minus_z = get_solid_neighbors(solid_mask, 2)
        
        # === 2. 右辺ベクトルbの構築（迎角対応, 係数: 2g/Δ） ===
        b = np.zeros((nx, ny, nz), dtype=np.float64)
        
        # 境界条件: ∂ϕ/∂n = g = -U·n
        # ゴーストセル法では係数は 2g/Δ になる
        # X面での寄与: g = -Ux
        if abs(Ux) > 1e-10:
            b[solid_plus_x] += (-2.0 * Ux) / dx   # +x面: n=+ex
            b[solid_minus_x] += (+2.0 * Ux) / dx  # -x面: n=-ex
            
        # Y面での寄与: g = -Uy
        if abs(Uy) > 1e-10:
            b[solid_plus_y] += (-2.0 * Uy) / dy   # +y面: n=+ey
            b[solid_minus_y] += (+2.0 * Uy) / dy  # -y面: n=-ey
            
        # Z面での寄与: g = -Uz
        if abs(Uz) > 1e-10:
            b[solid_plus_z] += (-2.0 * Uz) / dz   # +z面: n=+ez
            b[solid_minus_z] += (+2.0 * Uz) / dz  # -z面: n=-ez
        
        # === 2.5 Neumann整合性（右辺の平均ゼロ化） ===
        b_mean = b[fluid_mask].mean()
        if abs(b_mean) > 1e-14:
            print(f"      Enforcing compatibility: b_mean = {b_mean:.2e}")
            b[fluid_mask] -= b_mean
        
        # === 3. DCT前処理関数 ===
        try:
            from scipy.fft import dctn, idctn
            use_dct = True
            print("      Using DCT preconditioner (optimal)")
        except ImportError:
            use_dct = False
            print("      scipy.fft not available, using simple Jacobi")
        
        def neumann_poisson_precond(r):
            if not use_dct:
                # 簡易Jacobi前処理
                return r / (2/dx**2 + 2/dy**2 + 2/dz**2)
            
            # DCTベースの正確な前処理
            R_hat = dctn(r, type=2, norm='ortho')
            
            # 固有値
            kx = np.arange(nx)
            ky = np.arange(ny) 
            kz = np.arange(nz)
            lamx = 2*(1 - np.cos(np.pi * kx / nx)) / dx**2
            lamy = 2*(1 - np.cos(np.pi * ky / ny)) / dy**2
            lamz = 2*(1 - np.cos(np.pi * kz / nz)) / dz**2
            
            Lx, Ly, Lz = np.meshgrid(lamx, lamy, lamz, indexing='ij')
            L = Lx + Ly + Lz
            
            # 0モード（定数）の処理
            L[0,0,0] = 1.0
            Z_hat = R_hat / L
            Z_hat[0,0,0] = 0.0  # 平均ゼロ制約
            
            z = idctn(Z_hat, type=3, norm='ortho')
            return z
        
        # === 4. ラプラシアン演算子 ===
        def apply_laplacian(phi):
            # 7点ステンシルでゴーストセル処理
            Aphi = np.zeros_like(phi)
            
            # X方向の隣接値（ゴーストセル考慮）
            phi_xp = np.roll(phi, -1, axis=0)
            phi_xm = np.roll(phi, 1, axis=0)
            # 外枠Neumann
            phi_xp[-1,:,:] = phi[-1,:,:]
            phi_xm[0,:,:] = phi[0,:,:]
            # 固体隣接はself値に置換
            phi_xp[solid_plus_x] = phi[solid_plus_x]
            phi_xm[solid_minus_x] = phi[solid_minus_x]
            
            # Y方向
            phi_yp = np.roll(phi, -1, axis=1)
            phi_ym = np.roll(phi, 1, axis=1)
            phi_yp[:,-1,:] = phi[:,-1,:]
            phi_ym[:,0,:] = phi[:,0,:]
            phi_yp[solid_plus_y] = phi[solid_plus_y]
            phi_ym[solid_minus_y] = phi[solid_minus_y]
            
            # Z方向
            phi_zp = np.roll(phi, -1, axis=2)
            phi_zm = np.roll(phi, 1, axis=2)
            phi_zp[:,:,-1] = phi[:,:,-1]
            phi_zm[:,:,0] = phi[:,:,0]
            phi_zp[solid_plus_z] = phi[solid_plus_z]
            phi_zm[solid_minus_z] = phi[solid_minus_z]
            
            # ラプラシアン
            Aphi = ((phi_xp - 2*phi + phi_xm) / dx**2 +
                   (phi_yp - 2*phi + phi_ym) / dy**2 +
                   (phi_zp - 2*phi + phi_zm) / dz**2)
            
            # 固体セルはAφ=φ（恒等）として解空間から外す
            Aphi[solid_mask] = phi[solid_mask]
            
            return Aphi
        
        # 流体領域への投影関数
        def project_fluid(arr):
            out = arr.copy()
            out[solid_mask] = 0.0
            return out
        
        # デバッグ出力
        print(f"      b range: {b.min():.3e} .. {b.max():.3e},  ||b||₂={np.linalg.norm(b):.3e}")
        print(f"      Boundary cells: +x:{int(solid_plus_x.sum())} -x:{int(solid_minus_x.sum())} "
              f"+y:{int(solid_plus_y.sum())} -y:{int(solid_minus_y.sum())} "
              f"+z:{int(solid_plus_z.sum())} -z:{int(solid_minus_z.sum())}")
        print(f"      Nonzero(b): {int(np.count_nonzero(np.abs(b) > 0))}")
        print(f"      Fluid ratio: {float(fluid_mask.mean()):.3f}")
        
        # === 5. PCG法 ===
        print("      Solving with PCG...")
        phi = np.zeros((nx, ny, nz), dtype=np.float64)
        r = project_fluid(b - apply_laplacian(phi))
        z = project_fluid(neumann_poisson_precond(r))
        p = z.copy()
        rz_old = np.sum(r * z)
        
        tol = 1e-8
        max_iter = 300
        b_norm = np.linalg.norm(b) + 1e-30
        
        for iteration in range(1, max_iter + 1):
            Ap = project_fluid(apply_laplacian(p))
            den = np.sum(p * Ap) + 1e-30
            alpha = rz_old / den
            phi = phi + alpha * p
            r = project_fluid(r - alpha * Ap)
            
            res_norm = np.linalg.norm(r)
            rel_res = res_norm / b_norm
            
            if rel_res <= tol:
                print(f"        Converged at iteration {iteration}, rel_residual = {rel_res:.2e}")
                break
                
            z = project_fluid(neumann_poisson_precond(r))
            rz_new = np.sum(r * z)
            beta = rz_new / (rz_old + 1e-30)
            p = project_fluid(z + beta * p)
            rz_old = rz_new
            
            if iteration % 50 == 0:
                print(f"        Iteration {iteration}: rel_residual = {rel_res:.2e}")
        
        # === 6. ゲージ固定（流体領域の平均ゼロ） ===
        phi_mean = phi[fluid_mask].mean()
        phi -= phi_mean
        print(f"        Gauge fixing: subtracted mean = {phi_mean:.2e}")
        
        # 発散チェック
        div_check = apply_laplacian(phi) - b
        print(f"        Final: ||Aϕ - b||₂ = {np.linalg.norm(div_check):.2e}")
        
        return phi, Ux, Uy, Uz  # 速度成分も返す
    
    def _hodge_projection(self, u: np.ndarray, v: np.ndarray, 
                         w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Hodge投影による発散ゼロ速度場の生成
        
        u_new = u - ∇ψ where ∇²ψ = ∇·u
        """
        # 発散計算
        div_u = self._compute_divergence(u, v, w)
        
        # ポアソン方程式を解く（簡易Jacobi法）
        psi = self._solve_poisson_3d(div_u, max_iter=100)
        
        # 速度補正
        grad_psi_x = np.gradient(psi, self.grid.dx, axis=0)
        grad_psi_y = np.gradient(psi, self.grid.dy, axis=1)
        grad_psi_z = np.gradient(psi, self.grid.dz, axis=2)
        
        u_new = u - grad_psi_x
        v_new = v - grad_psi_y
        w_new = w - grad_psi_z
        
        return u_new, v_new, w_new
    
    def _compute_divergence(self, u: np.ndarray, v: np.ndarray, 
                          w: np.ndarray) -> np.ndarray:
        """速度場の発散を計算"""
        dudx = np.gradient(u, self.grid.dx, axis=0)
        dvdy = np.gradient(v, self.grid.dy, axis=1)
        dwdz = np.gradient(w, self.grid.dz, axis=2)
        return dudx + dvdy + dwdz
    
    def _solve_poisson_3d(self, rhs: np.ndarray, max_iter: int = 100,
                         tol: float = 1e-6) -> np.ndarray:
        """3Dポアソン方程式の簡易解法（Jacobi法）
        
        ∇²ψ = rhs
        """
        nx, ny, nz = rhs.shape
        psi = np.zeros_like(rhs)
        psi_new = np.zeros_like(rhs)
        
        dx2 = self.grid.dx**2
        dy2 = self.grid.dy**2
        dz2 = self.grid.dz**2
        
        # Jacobi反復
        for iteration in range(max_iter):
            # 内部点の更新
            psi_new[1:-1, 1:-1, 1:-1] = (
                (psi[2:, 1:-1, 1:-1] + psi[:-2, 1:-1, 1:-1]) / dx2 +
                (psi[1:-1, 2:, 1:-1] + psi[1:-1, :-2, 1:-1]) / dy2 +
                (psi[1:-1, 1:-1, 2:] + psi[1:-1, 1:-1, :-2]) / dz2 -
                rhs[1:-1, 1:-1, 1:-1]
            ) / (2/dx2 + 2/dy2 + 2/dz2)
            
            # Neumann境界条件（∂ψ/∂n = 0）
            psi_new[0, :, :] = psi_new[1, :, :]
            psi_new[-1, :, :] = psi_new[-2, :, :]
            psi_new[:, 0, :] = psi_new[:, 1, :]
            psi_new[:, -1, :] = psi_new[:, -2, :]
            psi_new[:, :, 0] = psi_new[:, :, 1]
            psi_new[:, :, -1] = psi_new[:, :, -2]
            
            # 収束判定
            if iteration % 10 == 0:
                residual = np.max(np.abs(psi_new - psi))
                if residual < tol:
                    print(f"      Poisson solver converged at iteration {iteration}")
                    break
            
            psi[:] = psi_new
        
        return psi
    
    def _calculate_thermo_map(self, velocity_map: Dict) -> Dict[str, np.ndarray]:
        """Map 2: 理想流の熱力学場（ベルヌーイの定理に基づく）"""
        u = velocity_map['velocity_u']
        v = velocity_map['velocity_v']
        w = velocity_map['velocity_w']
        
        U_inf = self.flow.U_inf
        p_inf = 101325.0  # 基準圧力 [Pa]（1気圧）
        rho_inf = self.flow.rho_inf  # 基準密度 [kg/m³]
        
        # === 理想流のベルヌーイの定理 ===
        # p + 1/2 ρ |u|² = p∞ + 1/2 ρ U∞²（流線上で一定）
        
        # 速度の大きさの二乗
        V_squared = u**2 + v**2 + w**2
        
        # 圧力係数（無次元）
        Cp = 1.0 - V_squared / U_inf**2
        
        # 静圧（次元あり）[Pa]
        pressure = p_inf + 0.5 * rho_inf * U_inf**2 * Cp
        
        # 理想流では密度・温度は一定（非圧縮性）
        density = np.ones_like(pressure) * rho_inf
        temperature = np.ones_like(pressure) * self.flow.temperature_inf
        
        # 正規化して返す（可視化用）
        pressure_normalized = pressure / p_inf
        density_normalized = density / rho_inf
        
        print(f"    Pressure range [Pa]: {pressure.min():.1f} - {pressure.max():.1f}")
        print(f"    Pressure range [p/p∞]: {pressure.min()/p_inf:.6f} - {pressure.max()/p_inf:.6f}")
        print(f"    Cp range: {Cp.min():.3f} - {Cp.max():.3f}")
        
        # ベルヌーイの定理の検証（流線上で全圧一定か）
        total_pressure = pressure + 0.5 * rho_inf * V_squared
        tp_std = np.std(total_pressure[V_squared > 0.1*U_inf**2])  # 流れがある領域で
        tp_mean = np.mean(total_pressure[V_squared > 0.1*U_inf**2])
        print(f"    Total pressure variation: {tp_std/tp_mean:.2e} (should be ~0 for ideal flow)")
        
        return {
            'pressure': pressure_normalized,  # p/p∞（後方互換性）
            'pressure_Pa': pressure,          # Pascal単位
            'Cp': Cp,                        # 圧力係数
            'density': density_normalized,
            'temperature': temperature
        }
    
    def _calculate_vortex_map(self, velocity_map: Dict) -> Dict[str, np.ndarray]:
        """Map 3: 渦構造テンソルの計算"""
        u = velocity_map['velocity_u']
        v = velocity_map['velocity_v']
        w = velocity_map['velocity_w']
        
        # 渦度ベクトル ω = ∇ × u
        omega_x = np.gradient(w, self.grid.dy, axis=1) - np.gradient(v, self.grid.dz, axis=2)
        omega_y = np.gradient(u, self.grid.dz, axis=2) - np.gradient(w, self.grid.dx, axis=0)
        omega_z = np.gradient(v, self.grid.dx, axis=0) - np.gradient(u, self.grid.dy, axis=1)
        
        # 速度勾配テンソル
        grad_u = np.zeros((self.grid.nx, self.grid.ny, self.grid.nz, 3, 3))
        
        # ∂u_i/∂x_j
        grad_u[:,:,:,0,0] = np.gradient(u, self.grid.dx, axis=0)
        grad_u[:,:,:,0,1] = np.gradient(u, self.grid.dy, axis=1)
        grad_u[:,:,:,0,2] = np.gradient(u, self.grid.dz, axis=2)
        grad_u[:,:,:,1,0] = np.gradient(v, self.grid.dx, axis=0)
        grad_u[:,:,:,1,1] = np.gradient(v, self.grid.dy, axis=1)
        grad_u[:,:,:,1,2] = np.gradient(v, self.grid.dz, axis=2)
        grad_u[:,:,:,2,0] = np.gradient(w, self.grid.dx, axis=0)
        grad_u[:,:,:,2,1] = np.gradient(w, self.grid.dy, axis=1)
        grad_u[:,:,:,2,2] = np.gradient(w, self.grid.dz, axis=2)
        
        # Q判定基準
        Q = np.zeros((self.grid.nx, self.grid.ny, self.grid.nz))
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                for k in range(self.grid.nz):
                    G = grad_u[i,j,k]
                    S = 0.5 * (G + G.T)  # 歪み速度テンソル
                    Omega = 0.5 * (G - G.T)  # 回転テンソル
                    Q[i,j,k] = 0.5 * (np.trace(Omega @ Omega.T) - np.trace(S @ S.T))
        
        # λ2基準（簡易版）
        lambda2 = np.zeros_like(Q)
        
        # ヘリシティ H = u · ω
        helicity = u * omega_x + v * omega_y + w * omega_z
        
        return {
            'vorticity_x': omega_x,
            'vorticity_y': omega_y,
            'vorticity_z': omega_z,
            'Q_criterion': Q,
            'lambda2': lambda2,
            'helicity': helicity
        }
    
    def _calculate_boundary_map(self) -> Dict[str, np.ndarray]:
        """Map 4: 境界層ポテンシャル（理想流では境界層厚さゼロ）"""
        
        # 理想流では境界層なし、剥離なし
        # ここでは「将来の粘性計算用」のマスク場のみ用意
        
        separation_potential = np.zeros((self.grid.nx, self.grid.ny, self.grid.nz))
        boundary_layer_mask = np.zeros_like(separation_potential)
        wall_distance = np.zeros_like(separation_potential)
        
        cx, cy, cz = self.obstacle.center_x, self.obstacle.center_y, self.obstacle.center_z
        R = self.obstacle.size
        
        # 壁面からの距離場を計算（粘性計算で使用）
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                for k in range(self.grid.nz):
                    x, y, z = self.x[i], self.y[j], self.z[k]
                    
                    if self.obstacle.shape_type == 'cylinder':
                        # 円柱表面からの距離
                        r = np.sqrt((x-cx)**2 + (y-cy)**2)
                        wall_distance[i,j,k] = max(0, r - R)
                        
                        # 剥離ポテンシャル位置のマーキング（理想流では使わない）
                        theta = np.arctan2(y-cy, x-cx)
                        if r > R and r < R + 10:
                            # 円柱の場合、約90度で剥離する可能性
                            if np.abs(theta) > np.pi/2:
                                separation_potential[i,j,k] = np.exp(-(r-R)/5)
                                
                    elif self.obstacle.shape_type == 'square':
                        # 角柱表面からの最短距離（簡易版）
                        dx = max(0, np.abs(x-cx) - R)
                        dy = max(0, np.abs(y-cy) - R)
                        dz = max(0, np.abs(z-cz) - self.obstacle.span/2)
                        wall_distance[i,j,k] = np.sqrt(dx**2 + dy**2 + dz**2)
                        
                        # エッジ部の剥離ポテンシャル
                        at_edge = (np.abs(np.abs(x-cx) - R) < 2 or 
                                  np.abs(np.abs(y-cy) - R) < 2)
                        if at_edge:
                            separation_potential[i,j,k] = 1.0
        
        print("    Boundary layer maps prepared (ideal flow: no actual BL)")
        
        return {
            'separation_potential': separation_potential,  # 粘性計算用のポテンシャル
            'boundary_layer_mask': boundary_layer_mask,   # 理想流では全てゼロ
            'wall_distance': wall_distance                # 壁面距離場（参考用）
        }
    
    def _calculate_formation_map(self) -> Dict[str, np.ndarray]:
        """Map 5: 渦形成ポテンシャル（理想流では渦なし）"""
        
        # 理想流（ポテンシャル流）は無渦 (∇×u = 0)
        # ここでは「粘性計算で渦が発生しやすい領域」のマーキングのみ
        
        vortex_formation_potential = np.zeros((self.grid.nx, self.grid.ny, self.grid.nz))
        horseshoe_potential = np.zeros_like(vortex_formation_potential)
        tip_vortex_potential = np.zeros_like(vortex_formation_potential)
        
        cx, cy, cz = self.obstacle.center_x, self.obstacle.center_y, self.obstacle.center_z
        R = self.obstacle.size
        
        # 馬蹄渦が形成されやすい領域（障害物前面の地面付近）
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                x, y = self.x[i], self.y[j]
                
                # 障害物前面
                if x < cx and x > cx - 3*R:
                    for k in range(int(self.grid.nz * 0.2)):  # 下部20%
                        z = self.z[k]
                        dist_to_front = np.sqrt((x-cx+R)**2 + (y-cy)**2)
                        
                        if dist_to_front < 2*R:
                            # ポテンシャルのみマーク（実際の渦は粘性で発生）
                            horseshoe_potential[i,j,k] = np.exp(-dist_to_front/R)
        
        # 端部渦のポテンシャル（有限スパンの影響）
        if self.obstacle.shape_type in ['cylinder', 'square']:
            for k in range(self.grid.nz):
                z_norm = np.abs(self.z[k] - cz) / (self.obstacle.span/2)
                
                # スパン端部近傍
                if z_norm > 0.8:
                    # 端部効果のポテンシャル
                    tip_vortex_potential[:,:,k] = np.exp(-5*(z_norm-1.0)**2)
        
        # カルマン渦列のポテンシャル（後流領域のマーク）
        for i in range(self.grid.nx):
            x_wake = self.x[i] - cx - R
            
            if x_wake > 0 and x_wake < 20*R:
                for j in range(self.grid.ny):
                    for k in range(self.grid.nz):
                        y_wake = np.abs(self.y[j] - cy)
                        
                        # 後流幅内
                        if y_wake < 3*R:
                            # 渦形成領域のポテンシャル（粘性で活性化）
                            vortex_formation_potential[i,j,k] = np.exp(-x_wake/(10*R)) * np.exp(-(y_wake/(2*R))**2)
        
        print("    Vortex formation potentials prepared (ideal flow: irrotational)")
        
        return {
            'horseshoe_potential': horseshoe_potential,
            'tip_vortex_potential': tip_vortex_potential,
            'vortex_formation_potential': vortex_formation_potential
        }
    
    def _calculate_lambda_map(self, velocity_map: Dict, 
                             thermo_map: Dict) -> Dict[str, np.ndarray]:
        """Map 6: Λ³構造場（完全版・物理的に正確な計算）"""
        print("  Computing full 3x3 velocity gradient tensor...")
        
        u = velocity_map['velocity_u']
        v = velocity_map['velocity_v']
        w = velocity_map['velocity_w']
        pressure = thermo_map['pressure']
        
        # === 1. 完全な3x3速度勾配テンソル ∇u の計算 ===
        # Lambda_core = ∂u_i/∂x_j (9成分にflatten)
        print("    - Calculating velocity gradients...")
        Lambda_core = np.zeros((self.grid.nx, self.grid.ny, self.grid.nz, 9))
        
        # 各成分を計算（中心差分）
        # ∂u/∂x, ∂u/∂y, ∂u/∂z
        dudx = np.gradient(u, self.grid.dx, axis=0)
        dudy = np.gradient(u, self.grid.dy, axis=1)
        dudz = np.gradient(u, self.grid.dz, axis=2)
        
        # ∂v/∂x, ∂v/∂y, ∂v/∂z
        dvdx = np.gradient(v, self.grid.dx, axis=0)
        dvdy = np.gradient(v, self.grid.dy, axis=1)
        dvdz = np.gradient(v, self.grid.dz, axis=2)
        
        # ∂w/∂x, ∂w/∂y, ∂w/∂z
        dwdx = np.gradient(w, self.grid.dx, axis=0)
        dwdy = np.gradient(w, self.grid.dy, axis=1)
        dwdz = np.gradient(w, self.grid.dz, axis=2)
        
        # テンソル成分を格納（row-major order）
        Lambda_core[:,:,:,0] = dudx  # (0,0)
        Lambda_core[:,:,:,1] = dudy  # (0,1)
        Lambda_core[:,:,:,2] = dudz  # (0,2)
        Lambda_core[:,:,:,3] = dvdx  # (1,0)
        Lambda_core[:,:,:,4] = dvdy  # (1,1)
        Lambda_core[:,:,:,5] = dvdz  # (1,2)
        Lambda_core[:,:,:,6] = dwdx  # (2,0)
        Lambda_core[:,:,:,7] = dwdy  # (2,1)
        Lambda_core[:,:,:,8] = dwdz  # (2,2)
        
        # === 2. テンション密度 ρT（速度の大きさ）===
        print("    - Computing tension density...")
        rho_T = np.sqrt(u**2 + v**2 + w**2)
        
        # === 3. 構造同期率 σₛ（圧力-速度相関）===
        print("    - Computing structural synchronization...")
        
        # 圧力勾配
        grad_p_x = np.gradient(pressure, self.grid.dx, axis=0)
        grad_p_y = np.gradient(pressure, self.grid.dy, axis=1)
        grad_p_z = np.gradient(pressure, self.grid.dz, axis=2)
        
        # テンション密度の勾配
        grad_rho_T_x = np.gradient(rho_T, self.grid.dx, axis=0)
        grad_rho_T_y = np.gradient(rho_T, self.grid.dy, axis=1)
        grad_rho_T_z = np.gradient(rho_T, self.grid.dz, axis=2)
        
        # 同期率：∇ρT・(u,v,w) / |∇ρT||u|
        numerator = grad_rho_T_x*u + grad_rho_T_y*v + grad_rho_T_z*w
        grad_rho_T_mag = np.sqrt(grad_rho_T_x**2 + grad_rho_T_y**2 + grad_rho_T_z**2)
        denominator = grad_rho_T_mag * rho_T + 1e-8
        sigma_s = numerator / denominator
        
        # === 4. トポロジカルチャージ Q_Λ（3D渦構造の強度）===
        print("    - Computing topological charge...")
        
        # 渦度成分（既に計算済みなら再利用可能）
        omega_x = dwdy - dvdz
        omega_y = dudz - dwdx
        omega_z = dvdx - dudy
        
        # 局所循環強度（ヘリシティ密度の絶対値）
        local_helicity = np.abs(u*omega_x + v*omega_y + w*omega_z)
        
        # エンストロフィー（渦度の二乗）
        enstrophy = omega_x**2 + omega_y**2 + omega_z**2
        
        # トポロジカルチャージ：循環とエンストロフィーの組み合わせ
        Q_Lambda = np.sqrt(local_helicity * enstrophy) / (rho_T + 1e-8)
        
        # === 5. 追加：構造効率場 η（Λ³理論の重要量）===
        print("    - Computing structural efficiency...")
        
        # 歪み速度テンソル S_ij = 0.5(∂u_i/∂x_j + ∂u_j/∂x_i)
        S11 = dudx
        S22 = dvdy
        S33 = dwdz
        S12 = 0.5 * (dudy + dvdx)
        S13 = 0.5 * (dudz + dwdx)
        S23 = 0.5 * (dvdz + dwdy)
        
        # 第2不変量 ||S||² = S_ij S_ij
        S_squared = S11**2 + S22**2 + S33**2 + 2*(S12**2 + S13**2 + S23**2)
        
        # 回転テンソル Ω_ij = 0.5(∂u_i/∂x_j - ∂u_j/∂x_i)
        # 第2不変量 ||Ω||² = Ω_ij Ω_ij = 0.5 * enstrophy
        Omega_squared = 0.5 * enstrophy
        
        # 構造効率：渦が歪みより優勢な領域
        efficiency = Omega_squared / (S_squared + Omega_squared + 1e-8)
        
        # === 6. 創発場 emergence（ΔΛCポテンシャル）===
        print("    - Computing emergence potential...")
        
        # 効率の勾配（構造変化が起きやすい場所）
        grad_eff_x = np.gradient(efficiency, self.grid.dx, axis=0)
        grad_eff_y = np.gradient(efficiency, self.grid.dy, axis=1)
        grad_eff_z = np.gradient(efficiency, self.grid.dz, axis=2)
        grad_eff_mag = np.sqrt(grad_eff_x**2 + grad_eff_y**2 + grad_eff_z**2)
        
        # 創発ポテンシャル：効率勾配が大きく、かつ同期率が高い領域
        emergence = grad_eff_mag * np.abs(sigma_s)
        
        # === 7. スカラー不変量の計算（参考値）===
        print("    - Computing scalar invariants...")
        
        # 第1不変量 P = -tr(∇u) = -(∂u/∂x + ∂v/∂y + ∂w/∂z)
        P = -(dudx + dvdy + dwdz)  # 連続式より≈0になるはず
        
        # 第2不変量 Q = 0.5(||Ω||² - ||S||²)
        Q_invariant = 0.5 * (Omega_squared - S_squared)
        
        # 第3不変量 R = -det(∇u)（計算は複雑なので省略可）
        
        print("    - Lambda map calculation complete!")
        
        return {
            'Lambda_core': Lambda_core,      # (nx,ny,nz,9) 完全な速度勾配テンソル
            'rho_T': rho_T,                 # (nx,ny,nz) テンション密度
            'sigma_s': np.clip(sigma_s, -1, 1),  # (nx,ny,nz) 同期率
            'Q_Lambda': Q_Lambda,            # (nx,ny,nz) トポロジカルチャージ
            'efficiency': efficiency,        # (nx,ny,nz) 構造効率
            'emergence': emergence,          # (nx,ny,nz) 創発ポテンシャル
            'divergence': P,                # (nx,ny,nz) 発散（連続性チェック用）
            'Q_criterion_from_lambda': Q_invariant  # (nx,ny,nz) Q判定基準（検証用）
        }
    
    def _compute_3d_wake_deficit(self) -> np.ndarray:
        """3D後流での圧力欠損"""
        deficit = np.zeros((self.grid.nx, self.grid.ny, self.grid.nz))
        
        cx, cy, cz = self.obstacle.center_x, self.obstacle.center_y, self.obstacle.center_z
        R = self.obstacle.size
        params = self.physics_params
        
        for i in range(self.grid.nx):
            x_wake = self.x[i] - cx - R
            
            if x_wake > 0:
                for j in range(self.grid.ny):
                    y_wake = self.y[j] - cy
                    
                    for k in range(self.grid.nz):
                        z_wake = self.z[k] - cz
                        
                        # 3D後流幅（下流で広がる）
                        wake_width_y = R * params['wake_width_factor'] * (1 + x_wake/(10*R))
                        wake_width_z = wake_width_y  # 対称と仮定
                        
                        # 後流内判定
                        in_wake_y = np.abs(y_wake) < wake_width_y
                        in_wake_z = np.abs(z_wake) < wake_width_z
                        
                        if in_wake_y and in_wake_z:
                            # 圧力回復
                            recovery = 1 - np.exp(-x_wake/(params['recovery_length']*R))
                            
                            # Y-Z断面でのガウス分布
                            distribution = np.exp(-(y_wake/wake_width_y)**2 - (z_wake/wake_width_z)**2)
                            
                            deficit[i,j,k] = params['base_pressure'] * (1 - recovery) * distribution
        
        return deficit
    
    def _compute_3d_vortex_pressure(self) -> np.ndarray:
        """3D渦による圧力変動"""
        pattern = np.zeros((self.grid.nx, self.grid.ny, self.grid.nz))
        
        cx, cy, cz = self.obstacle.center_x, self.obstacle.center_y, self.obstacle.center_z
        R = self.obstacle.size
        params = self.physics_params
        
        # 簡易的な3Dカルマン渦パターン
        for i in range(self.grid.nx):
            x_wake = self.x[i] - cx - R * params['vortex_formation_length']
            
            if x_wake > 0:
                St = params['strouhal_number_2d']
                wavelength = 2*R / St
                
                for j in range(self.grid.ny):
                    y_wake = self.y[j] - cy
                    
                    for k in range(self.grid.nz):
                        z_wake = self.z[k] - cz
                        
                        # スパン方向の変調
                        spanwise_mod = np.cos(2*np.pi * z_wake / (params['spanwise_correlation_length']*R))
                        
                        phase = 2*np.pi * x_wake / wavelength
                        amplitude = 0.5 * R * np.exp(-x_wake/(10*R)) * spanwise_mod
                        
                        # 上下の渦
                        if y_wake > 0:
                            vortex_y = amplitude * np.sin(phase)
                            distance = np.abs(y_wake - vortex_y)
                        else:
                            vortex_y = -amplitude * np.sin(phase)
                            distance = np.abs(y_wake - vortex_y)
                        
                        pattern[i,j,k] = -0.3 * np.exp(-(distance/R)**2)
        
        return pattern
    
    def save_maps(self, maps_dict: Dict, base_name: str = None) -> None:
        """複数のnpzファイルに分割保存"""
        
        if base_name is None:
            base_name = f"{self.obstacle.shape_type}_3d_Re{int(self.flow.Re)}"
        
        print(f"\n💾 Saving 3D maps with base name: {base_name}")
        
        # メタデータ
        metadata = {
            'obstacle': self.obstacle.__dict__,
            'flow': self.flow.__dict__,
            'grid': self.grid.__dict__,
            'physics_params': self.physics_params,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 各マップを個別に保存
        for map_name, map_data in maps_dict.items():
            filename = f"{base_name}_{map_name}.npz"
            
            print(f"  Saving {filename}...", end='')
            np.savez_compressed(
                filename,
                **map_data,
                metadata=json.dumps(metadata)
            )
            
            # ファイルサイズ確認
            import os
            size_mb = os.path.getsize(filename) / (1024**2)
            print(f" ✅ ({size_mb:.1f} MB)")
        
        total_size_gb = sum(
            os.path.getsize(f"{base_name}_{name}.npz") 
            for name in maps_dict.keys()
        ) / (1024**3)
        
        print(f"\n✨ Total size: {total_size_gb:.2f} GB")
        print(f"🎯 Ready for GET Wind™ v7.0 3D simulation!")
    
    def visualize_slice(self, field: np.ndarray, field_name: str, 
                        slice_type: str = 'z', slice_index: int = None) -> None:
        """3D場の2Dスライス可視化"""
        
        if slice_index is None:
            if slice_type == 'z':
                slice_index = self.grid.nz // 2
            elif slice_type == 'y':
                slice_index = self.grid.ny // 2
            else:
                slice_index = self.grid.nx // 2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if slice_type == 'z':
            data = field[:, :, slice_index].T
            extent = [self.grid.x_min, self.grid.x_max, 
                     self.grid.y_min, self.grid.y_max]
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            title_suffix = f"Z = {self.z[slice_index]:.1f}"
        elif slice_type == 'y':
            data = field[:, slice_index, :].T
            extent = [self.grid.x_min, self.grid.x_max,
                     self.grid.z_min, self.grid.z_max]
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            title_suffix = f"Y = {self.y[slice_index]:.1f}"
        else:
            data = field[slice_index, :, :].T
            extent = [self.grid.y_min, self.grid.y_max,
                     self.grid.z_min, self.grid.z_max]
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
            title_suffix = f"X = {self.x[slice_index]:.1f}"
        
        im = ax.imshow(data, origin='lower', extent=extent,
                      aspect='equal', cmap='RdBu_r')
        
        # 障害物を描画
        if slice_type == 'z':
            if self.obstacle.shape_type == 'cylinder':
                circle = plt.Circle((self.obstacle.center_x, self.obstacle.center_y),
                                   self.obstacle.size, fill=False, color='red', linewidth=2)
                ax.add_patch(circle)
            elif self.obstacle.shape_type == 'square':
                rect = plt.Rectangle(
                    (self.obstacle.center_x - self.obstacle.size,
                     self.obstacle.center_y - self.obstacle.size),
                    2*self.obstacle.size, 2*self.obstacle.size,
                    fill=False, color='red', linewidth=2
                )
                ax.add_patch(rect)
        
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"{field_name} - {title_suffix}")
        plt.tight_layout()
        plt.show()

# ==============================
# メイン実行関数
# ==============================

def generate_3d_geometric_maps(shape: str = 'square', Re: float = 200, 
                               save: bool = True, visualize: bool = True):
    """3D幾何学的ベルヌーイマップを生成"""
    
    # 設定
    if shape == 'cylinder':
        obstacle = Obstacle3DConfig(
            shape_type='cylinder',
            span=150.0  # スパン長
        )
    elif shape == 'square':
        obstacle = Obstacle3DConfig(
            shape_type='square',
            span=150.0
        )
    else:
        raise ValueError(f"Unknown shape: {shape}")
    
    flow = Flow3DConfig(Re=Re)
    
    # グリッド設定（メモリに応じて調整可能）
    grid = Grid3DConfig(
        nx=300,
        ny=150,
        nz=150  # Z方向解像度
    )
    
    # 計算実行
    calculator = GeometricBernoulli3D(obstacle, flow, grid)
    
    # 物理パラメータサマリー
    print("\n📊 Physical Parameters:")
    print(f"  Reynolds number: {Re}")
    print(f"  Expected 2D Strouhal: {calculator.physics_params.get('strouhal_number_2d', 'N/A')}")
    print(f"  Spanwise correlation: {calculator.physics_params.get('spanwise_correlation_length', 'N/A')}")
    
    # 全フィールド計算
    all_maps = calculator.calculate_all_fields()
    
    # 保存
    if save:
        calculator.save_maps(all_maps)
    
    # 可視化
    if visualize:
        print("\n📊 Generating visualizations...")
        
        # 代表的なフィールドをスライス表示
        if 'map3_vortex' in all_maps:
            calculator.visualize_slice(
                all_maps['map3_vortex']['Q_criterion'],
                'Q-criterion', 
                slice_type='z'
            )
        
        if 'map2_thermo' in all_maps:
            calculator.visualize_slice(
                all_maps['map2_thermo']['pressure'],
                'Pressure',
                slice_type='y'
            )
    
    return calculator, all_maps

# ==============================
# 使用例
# ==============================

if __name__ == "__main__":
    # 角柱の3Dマップ生成
    print("\n🔷 Generating 3D maps for SQUARE obstacle...")
    calc_square, maps_square = generate_3d_geometric_maps(
        shape='square',
        Re=200,
        save=True,
        visualize=True
    )
    
    # 円柱の3Dマップ生成
    print("\n🔵 Generating 3D maps for CYLINDER obstacle...")
    calc_cylinder, maps_cylinder = generate_3d_geometric_maps(
        shape='cylinder',
        Re=200,
        save=True,
        visualize=True
    )
    
    print("\n" + "="*70)
    print("✨ 3D Geometric Bernoulli Maps Generation Complete!")
    print("🚀 Ready for GET Wind™ v7.0 - Ultimate 3D Edition!")
    print("="*70)
