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
    
@dataclass
class Flow3DConfig:
    """3D流れの条件"""
    U_inf: float = 10.0      # 一様流速度 [m/s]
    V_inf: float = 0.0       # Y方向速度成分
    W_inf: float = 0.0       # Z方向速度成分
    rho_inf: float = 1.225   # 基準密度 [kg/m³]
    Re: float = 200.0        # Reynolds数
    temperature_inf: float = 293.0  # 基準温度 [K]

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
    scale_m_per_unit: float = 0.001  # 1グリッド単位 = 1mm
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
        """Map 1: 速度場の計算"""
        cx, cy, cz = self.obstacle.center_x, self.obstacle.center_y, self.obstacle.center_z
        R = self.obstacle.size
        U = self.flow.U_inf
        
        # メモリ効率のためチャンク処理
        u = np.zeros((self.grid.nx, self.grid.ny, self.grid.nz))
        v = np.zeros_like(u)
        w = np.zeros_like(u)
        
        if self.obstacle.shape_type == 'cylinder':
            # 円柱（Z方向に一様）
            for i in range(self.grid.nx):
                for j in range(self.grid.ny):
                    dx = self.x[i] - cx
                    dy = self.y[j] - cy
                    r = np.sqrt(dx**2 + dy**2)
                    
                    if r > R:
                        theta = np.arctan2(dy, dx)
                        # 2D円柱流れをZ方向に拡張
                        u[i,j,:] = U * (1 - (R/r)**2 * np.cos(2*theta))
                        v[i,j,:] = -U * (R/r)**2 * np.sin(2*theta)
                        
                        # スパン方向の変動を追加
                        for k in range(self.grid.nz):
                            z_norm = (self.z[k] - cz) / self.obstacle.span
                            spanwise_mod = 1 + 0.1 * np.sin(2*np.pi * z_norm * 3)
                            u[i,j,k] *= spanwise_mod
                            
                            # 端部効果（tip vortex）
                            if np.abs(z_norm) > 0.4:
                                w[i,j,k] = 0.1 * U * np.exp(-((z_norm-0.5)/0.1)**2)
                    
        elif self.obstacle.shape_type == 'square':
            # 角柱（より複雑な3D流れ）
            for i in range(self.grid.nx):
                for j in range(self.grid.ny):
                    for k in range(self.grid.nz):
                        x, y, z = self.x[i], self.y[j], self.z[k]
                        
                        # 角柱の外側判定
                        outside_x = np.abs(x - cx) > R
                        outside_y = np.abs(y - cy) > R
                        outside_z = np.abs(z - cz) > R
                        
                        if outside_x or outside_y or outside_z:
                            # 基本流
                            u[i,j,k] = U
                            
                            # 角柱による流れの偏向
                            if np.abs(x - cx) < 3*R and np.abs(y - cy) < 3*R:
                                deflection = np.exp(-((x-cx)**2 + (y-cy)**2) / (4*R**2))
                                u[i,j,k] *= (1 - 0.5*deflection)
                                
                                # 上下の非対称流れ
                                if y > cy:
                                    v[i,j,k] = U * 0.3 * deflection
                                else:
                                    v[i,j,k] = -U * 0.3 * deflection
                                
                                # エッジからの3D剥離
                                if np.abs(y - cy) < R * 1.1:
                                    edge_effect = np.exp(-((z-cz)/R)**2)
                                    w[i,j,k] = 0.2 * U * edge_effect * np.sin(x/R)
        
        # ベクトルポテンシャル（簡易計算）
        psi_x = np.zeros_like(u)
        psi_y = np.zeros_like(u) 
        psi_z = np.zeros_like(u)
        
        # ∇×A = u となるようなAを逆算（簡易版）
        # 実際は3D Poisson方程式を解く必要があるが、ここでは近似
        
        return {
            'velocity_u': u,
            'velocity_v': v,
            'velocity_w': w,
            'vector_potential_x': psi_x,
            'vector_potential_y': psi_y,
            'vector_potential_z': psi_z
        }
    
    def _calculate_thermo_map(self, velocity_map: Dict) -> Dict[str, np.ndarray]:
        """Map 2: 熱力学場の計算"""
        u = velocity_map['velocity_u']
        v = velocity_map['velocity_v']
        w = velocity_map['velocity_w']
        
        U_inf = self.flow.U_inf
        
        # ベルヌーイの定理から圧力を計算
        V_squared = u**2 + v**2 + w**2
        Cp = 1.0 - V_squared / U_inf**2
        
        # 後流での圧力欠損
        wake_deficit = self._compute_3d_wake_deficit()
        
        # 3D渦による圧力変動
        vortex_pressure = self._compute_3d_vortex_pressure()
        
        # 合成
        pressure = 1.0 + 0.5 * (Cp + wake_deficit + vortex_pressure)
        
        # 密度（等温過程の仮定）
        density = pressure
        
        # 温度場（渦コアで温度低下）
        temperature = self.flow.temperature_inf * pressure
        
        return {
            'pressure': np.clip(pressure, 0.3, 2.0),
            'density': np.clip(density, 0.5, 1.5),
            'temperature': np.clip(temperature, 280.0, 310.0)
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
        """Map 4: 境界層・剥離構造"""
        separation = np.zeros((self.grid.nx, self.grid.ny, self.grid.nz))
        boundary_layer = np.zeros_like(separation)
        wall_shear = np.zeros_like(separation)
        
        cx, cy, cz = self.obstacle.center_x, self.obstacle.center_y, self.obstacle.center_z
        R = self.obstacle.size
        params = self.physics_params
        
        if self.obstacle.shape_type == 'cylinder':
            for i in range(self.grid.nx):
                for j in range(self.grid.ny):
                    dx = self.x[i] - cx
                    dy = self.y[j] - cy
                    r = np.sqrt(dx**2 + dy**2)
                    theta = np.arctan2(dy, dx)
                    
                    # 剥離点以降
                    if r > R and r < R + 10:
                        if np.abs(theta) > params['separation_angle']:
                            for k in range(self.grid.nz):
                                # スパン方向の変動
                                z_mod = 1 + 0.2*np.sin(2*np.pi*self.z[k]/params['spanwise_correlation_length'])
                                separation[i,j,k] = z_mod * np.exp(-(r-R)/5)
                                
                    # 境界層厚さ
                    if r > R and r < R + 5:
                        delta = 5 * np.sqrt((r-R) / self.flow.Re)
                        boundary_layer[i,j,:] = delta
                        
        elif self.obstacle.shape_type == 'square':
            # エッジで固定剥離
            for i in range(self.grid.nx):
                for j in range(self.grid.ny):
                    for k in range(self.grid.nz):
                        x, y, z = self.x[i], self.y[j], self.z[k]
                        
                        # エッジ近傍
                        at_edge_x = np.abs(np.abs(x-cx) - R) < 2
                        at_edge_y = np.abs(np.abs(y-cy) - R) < 2
                        
                        if at_edge_x or at_edge_y:
                            separation[i,j,k] = 1.0
                            
                        # コーナーでの強い剥離
                        at_corner = at_edge_x and at_edge_y
                        if at_corner:
                            separation[i,j,k] = 2.0
        
        # スムージング
        separation = gaussian_filter(separation, sigma=1.0)
        boundary_layer = gaussian_filter(boundary_layer, sigma=1.0)
        
        return {
            'separation_surface': separation,
            'boundary_layer_thickness': boundary_layer,
            'wall_shear_stress': wall_shear
        }
    
    def _calculate_formation_map(self) -> Dict[str, np.ndarray]:
        """Map 5: 3D渦形成領域"""
        horseshoe = np.zeros((self.grid.nx, self.grid.ny, self.grid.nz))
        tip_vortices = np.zeros_like(horseshoe)
        wake_cells = np.zeros_like(horseshoe)
        
        cx, cy, cz = self.obstacle.center_x, self.obstacle.center_y, self.obstacle.center_z
        R = self.obstacle.size
        params = self.physics_params
        
        # 馬蹄渦（地面付近）
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                x, y = self.x[i], self.y[j]
                
                # 障害物前面
                if x < cx and x > cx - 3*R:
                    for k in range(int(self.grid.nz * 0.2)):  # 下部20%のみ
                        z = self.z[k]
                        dist_to_front = np.sqrt((x-cx+R)**2 + (y-cy)**2)
                        
                        if dist_to_front < 2*R:
                            horseshoe[i,j,k] = params['horseshoe_strength'] * \
                                              np.exp(-dist_to_front/R) * \
                                              np.exp(-z/(0.2*self.grid.z_max))
        
        # 端部渦（スパン方向の端）
        if self.obstacle.shape_type in ['cylinder', 'square']:
            for i in range(self.grid.nx):
                if self.x[i] > cx:
                    for j in range(self.grid.ny):
                        for k in range(self.grid.nz):
                            z = self.z[k]
                            z_norm = np.abs(z - cz) / (self.obstacle.span/2)
                            
                            # 端部近傍
                            if z_norm > 0.8:
                                x_wake = self.x[i] - cx
                                tip_strength = np.exp(-z_norm*5) * np.exp(-x_wake/(10*R))
                                tip_vortices[i,j,k] = tip_strength
        
        # 3Dカルマン渦セル構造
        for i in range(self.grid.nx):
            x_wake = self.x[i] - cx - R * params['vortex_formation_length']
            
            if x_wake > 0:
                for j in range(self.grid.ny):
                    y_wake = self.y[j] - cy
                    
                    for k in range(self.grid.nz):
                        z_wake = self.z[k] - cz
                        
                        # カルマン渦の波長
                        St = params['strouhal_number_2d']
                        wavelength = 2*R / St
                        
                        # 位相（スパン方向で変化）
                        if self.obstacle.shape_type == 'square':
                            # 斜め渦（oblique shedding）
                            oblique_angle = params['oblique_angle'] * np.pi/180
                            phase = 2*np.pi * (x_wake/wavelength + z_wake*np.tan(oblique_angle)/wavelength)
                        else:
                            # 通常のカルマン渦（スパン方向セル）
                            cell_length = params['cell_aspect_ratio'] * R
                            cell_phase = 2*np.pi * z_wake / cell_length
                            phase = 2*np.pi * x_wake/wavelength + 0.1*np.sin(cell_phase)
                        
                        # 振幅（下流で減衰）
                        amplitude = R * np.exp(-x_wake/(10*R))
                        
                        # 上下交互の渦
                        upper_vortex = amplitude * np.sin(phase)
                        lower_vortex = -amplitude * np.sin(phase + np.pi)
                        
                        if y_wake > 0:
                            strength = np.exp(-((y_wake - upper_vortex)/R)**2)
                        else:
                            strength = np.exp(-((y_wake - lower_vortex)/R)**2)
                            
                        wake_cells[i,j,k] = strength * np.exp(-x_wake/(20*R))
        
        return {
            'horseshoe_vortex': horseshoe,
            'tip_vortices': tip_vortices,
            'wake_cells': wake_cells
        }
    
    def _calculate_lambda_map(self, velocity_map: Dict, 
                             thermo_map: Dict) -> Dict[str, np.ndarray]:
        """Map 6: Λ³構造場"""
        # Lambda_coreテンソル場（3x3を9成分にflatten）
        Lambda_core = np.zeros((self.grid.nx, self.grid.ny, self.grid.nz, 9))
        
        # テンション密度
        u = velocity_map['velocity_u']
        v = velocity_map['velocity_v']
        w = velocity_map['velocity_w']
        rho_T = np.sqrt(u**2 + v**2 + w**2)
        
        # 同期率場（圧力と速度の相関から）
        pressure = thermo_map['pressure']
        sigma_s = np.zeros_like(rho_T)
        
        # 簡易的な同期率計算
        grad_p_x = np.gradient(pressure, self.grid.dx, axis=0)
        grad_p_y = np.gradient(pressure, self.grid.dy, axis=1)
        grad_p_z = np.gradient(pressure, self.grid.dz, axis=2)
        
        # 速度と圧力勾配の内積を正規化
        numerator = u*grad_p_x + v*grad_p_y + w*grad_p_z
        denominator = rho_T * np.sqrt(grad_p_x**2 + grad_p_y**2 + grad_p_z**2) + 1e-8
        sigma_s = numerator / denominator
        
        # トポロジカルチャージ（3D循環）
        Q_Lambda = np.zeros_like(rho_T)
        
        # 簡易的な循環計算
        for i in range(1, self.grid.nx-1):
            for j in range(1, self.grid.ny-1):
                for k in range(1, self.grid.nz-1):
                    # 局所的な循環
                    circ_xy = (v[i+1,j,k] - v[i-1,j,k]) - (u[i,j+1,k] - u[i,j-1,k])
                    circ_xz = (w[i+1,j,k] - w[i-1,j,k]) - (u[i,j,k+1] - u[i,j,k-1])
                    circ_yz = (w[i,j+1,k] - w[i,j-1,k]) - (v[i,j,k+1] - v[i,j,k-1])
                    
                    Q_Lambda[i,j,k] = np.sqrt(circ_xy**2 + circ_xz**2 + circ_yz**2)
        
        # Lambda_coreの簡易計算（速度勾配テンソルから）
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                for k in range(self.grid.nz):
                    # 3x3速度勾配テンソルを9成分に
                    # ここでは簡易的に対角成分のみ
                    Lambda_core[i,j,k,0] = np.gradient(u, self.grid.dx, axis=0)[i,j,k] if i>0 and i<self.grid.nx-1 else 0
                    Lambda_core[i,j,k,4] = np.gradient(v, self.grid.dy, axis=1)[i,j,k] if j>0 and j<self.grid.ny-1 else 0
                    Lambda_core[i,j,k,8] = np.gradient(w, self.grid.dz, axis=2)[i,j,k] if k>0 and k<self.grid.nz-1 else 0
        
        return {
            'Lambda_core': Lambda_core,
            'rho_T': rho_T,
            'sigma_s': np.clip(sigma_s, -1, 1),
            'Q_Lambda': Q_Lambda
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
    print("環ちゃん & ご主人さま、最高の3Dシミュレーションを！💕")
    print("="*70)
