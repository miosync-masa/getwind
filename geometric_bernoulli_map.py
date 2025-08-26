#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometric Bernoulli Map Generator for GET Wind™
理想流体の幾何学的構造を持つ圧力・密度場生成器

物理的に正しい流線形状と圧力分布を生成し、
粘性粒子法との美しい責務分担を実現！
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import json

# ==============================
# Configuration Classes
# ==============================

@dataclass
class ObstacleConfig:
    """障害物の設定"""
    shape_type: str  # 'cylinder', 'square', 'airfoil'
    center_x: float = 100.0
    center_y: float = 75.0
    size: float = 20.0  # 特性長さ（円柱なら半径、角柱なら半幅）
    angle: float = 0.0  # 迎角[deg]

@dataclass
class FlowConfig:
    """流れの条件"""
    U_inf: float = 10.0      # 一様流速度 [m/s]
    rho_inf: float = 1.225   # 基準密度 [kg/m³]
    Re: float = 200.0        # Reynolds数（形状特性の決定用）

@dataclass
class GridConfig:
    """計算グリッド設定（物理単位対応版）"""
    nx: int = 300
    ny: int = 150
    x_min: float = 0.0
    x_max: float = 300.0
    y_min: float = 0.0
    y_max: float = 150.0
    
    # ✨ NEW: 物理スケーリング
    scale_m_per_unit: float = 0.001  # 1グリッド単位 = 1mm
    scale_s_per_step: float = 0.01   # 1ステップ = 0.01秒
    
    @property
    def physical_width(self) -> float:
        """物理的な幅 [m]"""
        return (self.x_max - self.x_min) * self.scale_m_per_unit
    
    @property
    def physical_height(self) -> float:
        """物理的な高さ [m]"""
        return (self.y_max - self.y_min) * self.scale_m_per_unit

# ==============================
# Geometric Bernoulli Field Calculator
# ==============================

class GeometricBernoulliCalculator:
    """幾何学的ベルヌーイ場の計算"""
    
    def __init__(self, obstacle: ObstacleConfig, flow: FlowConfig, grid: GridConfig):
        self.obstacle = obstacle
        self.flow = flow
        self.grid = grid
        
        # グリッド生成
        self.x = np.linspace(grid.x_min, grid.x_max, grid.nx)
        self.y = np.linspace(grid.y_min, grid.y_max, grid.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # 形状別の物理パラメータ（Re=200での実験値）
        self.physics_params = self._get_physics_params()
        
    def _get_physics_params(self) -> Dict:
        """形状別の物理パラメータ（実験値ベース）"""
        if self.obstacle.shape_type == 'cylinder':
            return {
                'separation_angle': np.pi/2.2,  # 約82度
                'wake_width_factor': 1.5,
                'vortex_formation_length': 2.0,  # 直径の2倍後方
                'strouhal_number': 0.195,
                'base_pressure': -0.9,  # 背圧係数
                'recovery_length': 10.0,  # 圧力回復長さ（直径単位）
            }
        elif self.obstacle.shape_type == 'square':
            return {
                'separation_angle': np.pi/2,  # 90度（エッジ）
                'wake_width_factor': 2.2,
                'vortex_formation_length': 1.5,
                'strouhal_number': 0.14,
                'base_pressure': -1.4,
                'recovery_length': 15.0,
            }
        else:
            raise ValueError(f"Unknown shape: {self.obstacle.shape_type}")
    
    def calculate_all_fields(self) -> Dict[str, np.ndarray]:
        """全ての場を計算"""
        print(f"Calculating geometric fields for {self.obstacle.shape_type}...")
        
        # 1. 流線関数と速度場
        psi, u, v = self.calculate_stream_function()
        
        # 2. 幾何学的圧力場
        pressure = self.calculate_geometric_pressure(psi, u, v)
        
        # 3. 密度場（圧力から導出）
        density = self.calculate_density_from_pressure(pressure)
        
        # 4. 剥離領域
        separation = self.calculate_separation_zones()
        
        # 5. 渦形成領域
        vortex_formation = self.calculate_vortex_formation_zones()
        
        # 6. 後流構造
        wake_structure = self.calculate_wake_structure()
        
        # 7. せん断層
        shear_layer = self.calculate_shear_layers()
        
        return {
            'stream_function': psi,
            'velocity_u': u,
            'velocity_v': v,
            'pressure': pressure,
            'density': density,
            'separation': separation,
            'vortex_formation': vortex_formation,
            'wake_structure': wake_structure,
            'shear_layer': shear_layer
        }
    
    def calculate_stream_function(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """流線関数と速度場の計算"""
        cx, cy = self.obstacle.center_x, self.obstacle.center_y
        R = self.obstacle.size
        U = self.flow.U_inf
        
        if self.obstacle.shape_type == 'cylinder':
            # 円柱周りの流線関数
            dx = self.X - cx
            dy = self.Y - cy
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            
            # ψ = U∞r sin(θ)(1 - R²/r²)
            psi = np.where(r > R,
                          U * r * np.sin(theta) * (1 - (R/r)**2),
                          0)
            
            # 速度成分（流線関数から導出）
            u = np.where(r > R,
                        U * (1 - (R/r)**2 * np.cos(2*theta)),
                        0)
            v = np.where(r > R,
                        -U * (R/r)**2 * np.sin(2*theta),
                        0)
            
        elif self.obstacle.shape_type == 'square':
            # 角柱（近似的な取り扱い）
            dx = self.X - cx
            dy = self.Y - cy
            
            # 角柱の外側判定
            outside = (np.abs(dx) > R) | (np.abs(dy) > R)
            
            # Schwarz-Christoffel変換の近似
            psi = np.zeros_like(self.X)
            u = np.ones_like(self.X) * U
            v = np.zeros_like(self.Y)
            
            # 角柱周りの流れ（簡易モデル）
            for i in range(self.grid.nx):
                for j in range(self.grid.ny):
                    if outside[i, j]:
                        x, y = self.X[i, j], self.Y[i, j]
                        # 角の影響を考慮
                        if np.abs(x - cx) < 2*R and np.abs(y - cy) < 2*R:
                            # 流れの偏向
                            deflection = np.exp(-((x-cx)**2 + (y-cy)**2) / (4*R**2))
                            u[i, j] = U * (1 - 0.5*deflection)
                            if y > cy:
                                v[i, j] = U * 0.3 * deflection
                            else:
                                v[i, j] = -U * 0.3 * deflection
            
            # 流線関数の計算（速度場から逆算）
            psi = self._velocity_to_stream_function(u, v)
        
        return psi, u, v
    
    def _velocity_to_stream_function(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """速度場から流線関数を計算（積分）"""
        psi = np.zeros_like(u)
        # 簡易的な積分（下端から）
        for i in range(self.grid.nx):
            for j in range(1, self.grid.ny):
                psi[i, j] = psi[i, j-1] + u[i, j-1] * self.grid.dy
        return psi
    
    def calculate_geometric_pressure(self, psi: np.ndarray, 
                                   u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """幾何学的圧力場の計算"""
        U_inf = self.flow.U_inf
        params = self.physics_params
        
        # 1. 基本ベルヌーイ圧力
        V_squared = u**2 + v**2
        Cp_bernoulli = 1.0 - V_squared / U_inf**2  # 圧力係数
        
        # 2. 流線の曲率による補正
        curvature = self._compute_streamline_curvature(psi)
        Cp_curvature = -0.1 * curvature * np.sqrt(V_squared) / U_inf
        
        # 3. 後流での圧力欠損（実験値ベース）
        wake_pressure_deficit = self._compute_wake_pressure_deficit()
        
        # 4. 渦形成領域での圧力変動
        vortex_pressure = self._compute_vortex_pressure_pattern()
        
        # 5. 合成
        Cp = Cp_bernoulli + Cp_curvature + wake_pressure_deficit + vortex_pressure
        
        # 圧力比として返す（無次元）
        pressure_ratio = 1.0 + 0.5 * Cp
        
        return np.clip(pressure_ratio, 0.3, 2.0)
    
    def _compute_streamline_curvature(self, psi: np.ndarray) -> np.ndarray:
        """流線の曲率を計算"""
        # 流線関数の勾配
        dpsi_dx = np.gradient(psi, self.grid.dx, axis=0)
        dpsi_dy = np.gradient(psi, self.grid.dy, axis=1)
        
        # 二階微分
        d2psi_dx2 = np.gradient(dpsi_dx, self.grid.dx, axis=0)
        d2psi_dy2 = np.gradient(dpsi_dy, self.grid.dy, axis=1)
        
        # 曲率の近似
        grad_norm = np.sqrt(dpsi_dx**2 + dpsi_dy**2) + 1e-8
        curvature = np.abs(d2psi_dx2 + d2psi_dy2) / grad_norm
        
        # スムージング
        curvature = gaussian_filter(curvature, sigma=2.0)
        
        return np.clip(curvature, 0, 10.0)
    
    def _compute_wake_pressure_deficit(self) -> np.ndarray:
        """後流での圧力欠損"""
        cx, cy = self.obstacle.center_x, self.obstacle.center_y
        R = self.obstacle.size
        params = self.physics_params
        
        deficit = np.zeros_like(self.X)
        
        # 後流領域
        x_wake = self.X - cx - R
        y_wake = self.Y - cy
        
        # 基底圧力（実験値）
        base_pressure = params['base_pressure']
        
        # 後流幅（下流に向かって広がる）
        wake_width = R * params['wake_width_factor'] * (1 + x_wake / (10*R))
        
        # 圧力回復
        recovery = 1 - np.exp(-x_wake / (params['recovery_length'] * R))
        
        # 後流内での圧力分布
        in_wake = (x_wake > 0) & (np.abs(y_wake) < wake_width)
        
        # Y方向のガウス分布
        y_distribution = np.exp(-(y_wake / wake_width)**2)
        
        deficit = np.where(in_wake,
                          base_pressure * (1 - recovery) * y_distribution,
                          0)
        
        return deficit
    
    def _compute_vortex_pressure_pattern(self) -> np.ndarray:
        """渦形成パターンによる圧力変動"""
        cx, cy = self.obstacle.center_x, self.obstacle.center_y
        R = self.obstacle.size
        params = self.physics_params
        
        pattern = np.zeros_like(self.X)
        
        # 渦形成領域
        x_wake = self.X - cx - R * params['vortex_formation_length']
        y_wake = self.Y - cy
        
        if self.obstacle.shape_type == 'cylinder':
            # カルマン渦列の幾何学的パターン
            St = params['strouhal_number']
            wavelength = 1.0 / St * 2 * R  # 渦の間隔
            
            # 上下交互の渦列
            phase = 2 * np.pi * x_wake / wavelength
            amplitude = 0.5 * R * np.exp(-x_wake / (10*R))
            
            # 上側渦列
            upper_vortex_y = amplitude * np.sin(phase)
            upper_distance = np.abs(y_wake - upper_vortex_y)
            upper_pressure = -0.3 * np.exp(-(upper_distance / R)**2)
            
            # 下側渦列（位相が逆）
            lower_vortex_y = -amplitude * np.sin(phase)
            lower_distance = np.abs(y_wake - lower_vortex_y)
            lower_pressure = -0.3 * np.exp(-(lower_distance / R)**2)
            
            pattern = np.where(x_wake > 0, upper_pressure + lower_pressure, 0)
            
        elif self.obstacle.shape_type == 'square':
            # より強い渦、より規則的
            St = params['strouhal_number']
            wavelength = 1.0 / St * 2 * R
            
            phase = 2 * np.pi * x_wake / wavelength
            amplitude = 0.8 * R * np.exp(-x_wake / (15*R))
            
            # エッジから放出される強い渦
            upper_vortex_y = R + amplitude * np.sin(phase)
            lower_vortex_y = -R - amplitude * np.sin(phase)
            
            upper_distance = np.abs(y_wake - upper_vortex_y)
            lower_distance = np.abs(y_wake - lower_vortex_y)
            
            # 角柱は渦が強い
            upper_pressure = -0.5 * np.exp(-(upper_distance / (0.8*R))**2)
            lower_pressure = -0.5 * np.exp(-(lower_distance / (0.8*R))**2)
            
            pattern = np.where(x_wake > 0, upper_pressure + lower_pressure, 0)
        
        return pattern
    
    def calculate_density_from_pressure(self, pressure: np.ndarray) -> np.ndarray:
        """圧力から密度を計算（等温過程）"""
        # ρ/ρ∞ = P/P∞（理想気体、等温）
        density_ratio = pressure
        
        # 渦コアでの密度低下を強調
        vortex_regions = pressure < 0.8
        density_ratio = np.where(vortex_regions,
                                 density_ratio * 0.9,
                                 density_ratio)
        
        return np.clip(density_ratio, 0.5, 1.5)
    
    def calculate_separation_zones(self) -> np.ndarray:
        """剥離領域の計算"""
        cx, cy = self.obstacle.center_x, self.obstacle.center_y
        R = self.obstacle.size
        params = self.physics_params
        
        separation = np.zeros_like(self.X)
        
        if self.obstacle.shape_type == 'cylinder':
            # 剥離点から下流
            dx = self.X - cx
            dy = self.Y - cy
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            
            # 剥離角度以降
            sep_angle = params['separation_angle']
            is_separated = ((np.abs(theta) > sep_angle) & (np.abs(theta) < np.pi) &
                          (r > R) & (r < R + 10))
            
            separation = np.where(is_separated, 1.0, 0.0)
            
        elif self.obstacle.shape_type == 'square':
            # エッジから即座に剥離
            dx = self.X - cx
            dy = self.Y - cy
            
            # 前縁エッジ
            at_edges = ((np.abs(dx - R) < 5) & (np.abs(dy) <= R))
            # コーナー
            at_corners = ((np.abs(np.abs(dx) - R) < 5) & 
                         (np.abs(np.abs(dy) - R) < 5))
            
            separation = np.where(at_edges | at_corners, 1.0, 0.0)
        
        # スムージング
        separation = gaussian_filter(separation, sigma=1.0)
        
        return separation
    
    def calculate_vortex_formation_zones(self) -> np.ndarray:
        """渦形成領域"""
        cx, cy = self.obstacle.center_x, self.obstacle.center_y
        R = self.obstacle.size
        params = self.physics_params
        
        formation = np.zeros_like(self.X)
        
        # 渦形成長さの位置から開始
        x_formation = self.X - cx - R * params['vortex_formation_length']
        y_wake = self.Y - cy
        
        # 渦形成領域（後流の初期部分）
        in_formation = ((x_formation > 0) & 
                       (x_formation < 5*R) &
                       (np.abs(y_wake) < 2*R))
        
        # 上下で交互に強くなる
        upper_strength = np.where(y_wake > 0, 1.0, 0.3)
        lower_strength = np.where(y_wake <= 0, 1.0, 0.3)
        
        alternating = np.sin(2*np.pi * x_formation / (5*R))
        
        formation = np.where(in_formation,
                           np.where(alternating > 0,
                                   upper_strength,
                                   lower_strength),
                           0)
        
        return gaussian_filter(formation, sigma=2.0)
    
    def calculate_wake_structure(self) -> np.ndarray:
        """後流構造"""
        cx, cy = self.obstacle.center_x, self.obstacle.center_y
        R = self.obstacle.size
        params = self.physics_params
        
        wake = np.zeros_like(self.X)
        
        x_wake = self.X - cx
        y_wake = self.Y - cy
        
        # 後流幅
        wake_width = R * params['wake_width_factor'] * np.sqrt(1 + x_wake / (10*R))
        
        in_wake = (x_wake > R) & (np.abs(y_wake) < wake_width)
        
        # 中心ほど強い
        intensity = np.exp(-(y_wake / wake_width)**2)
        
        # 下流に向かって減衰
        decay = np.exp(-x_wake / (20*R))
        
        wake = np.where(in_wake, intensity * decay, 0)
        
        return wake
    
    def calculate_shear_layers(self) -> np.ndarray:
        """せん断層（剥離せん断層）"""
        cx, cy = self.obstacle.center_x, self.obstacle.center_y
        R = self.obstacle.size
        
        shear = np.zeros_like(self.X)
        
        if self.obstacle.shape_type == 'cylinder':
            # 剥離点から放出されるせん断層
            dx = self.X - cx
            dy = self.Y - cy
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            
            # 上下の剥離せん断層
            upper_shear_angle = self.physics_params['separation_angle']
            lower_shear_angle = -upper_shear_angle
            
            # せん断層の厚さ
            shear_thickness = 0.1 * R * np.sqrt(1 + (r - R) / R)
            
            upper_shear = np.abs(theta - upper_shear_angle) < shear_thickness / r
            lower_shear = np.abs(theta - lower_shear_angle) < shear_thickness / r
            
            shear = np.where((r > R) & (upper_shear | lower_shear), 1.0, 0.0)
            
        elif self.obstacle.shape_type == 'square':
            # エッジから放出される強いせん断層
            dx = self.X - cx
            dy = self.Y - cy
            
            # 上下エッジから
            from_upper_edge = ((dx > R) & (np.abs(dy - R) < 5))
            from_lower_edge = ((dx > R) & (np.abs(dy + R) < 5))
            
            shear = np.where(from_upper_edge | from_lower_edge, 1.0, 0.0)
        
        return gaussian_filter(shear, sigma=1.0)
    
    def visualize_fields(self, fields: Dict[str, np.ndarray]) -> None:
        """場の可視化"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        extent = [self.grid.x_min, self.grid.x_max, 
                 self.grid.y_min, self.grid.y_max]
        
        # 各場をプロット
        field_names = ['stream_function', 'pressure', 'density',
                      'separation', 'vortex_formation', 'wake_structure',
                      'shear_layer', 'velocity_u', 'velocity_v']
        
        for ax, name in zip(axes.flat, field_names):
            if name in fields:
                im = ax.imshow(fields[name].T, origin='lower',
                             extent=extent, aspect='equal',
                             cmap='RdBu_r' if 'pressure' in name or 'density' in name else 'viridis')
                ax.set_title(name.replace('_', ' ').title())
                plt.colorbar(im, ax=ax, fraction=0.046)
                
                # 障害物を描画
                if self.obstacle.shape_type == 'cylinder':
                    circle = plt.Circle((self.obstacle.center_x, self.obstacle.center_y),
                                       self.obstacle.size, fill=False, color='red', linewidth=2)
                    ax.add_patch(circle)
                elif self.obstacle.shape_type == 'square':
                    rect = plt.Rectangle(
                        (self.obstacle.center_x - self.obstacle.size,
                         self.obstacle.center_y - self.obstacle.size),
                        2 * self.obstacle.size, 2 * self.obstacle.size,
                        fill=False, color='red', linewidth=2
                    )
                    ax.add_patch(rect)
        
        plt.tight_layout()
        plt.show()

# ==============================
# Main Function
# ==============================

def generate_geometric_map(shape='cylinder', Re=200, save=True):
    """幾何学的ベルヌーイマップを生成"""
    
    print(f"\n{'='*60}")
    print(f"Generating Geometric Bernoulli Map")
    print(f"Shape: {shape}, Re: {Re}")
    print(f"{'='*60}")
    
    # 設定
    obstacle = ObstacleConfig(shape_type=shape)
    flow = FlowConfig(Re=Re)
    grid = GridConfig()
    
    # 計算
    calculator = GeometricBernoulliCalculator(obstacle, flow, grid)
    fields = calculator.calculate_all_fields()
    
    # 保存
    if save:
        filename = f"{shape}_Re{int(Re)}_geometric.npz"
        
        # メタデータ
        metadata = {
            'obstacle': obstacle.__dict__,
            'flow': flow.__dict__,
            'grid': grid.__dict__,
            'physics_params': calculator.physics_params
        }
        
        # 保存
        save_dict = {**fields, 'metadata': json.dumps(metadata)}
        np.savez_compressed(filename, **save_dict)
        print(f"\nSaved to {filename}")
        
        # 統計表示
        print("\nField statistics:")
        for name, field in fields.items():
            print(f"  {name:20s}: min={field.min():.3f}, "
                  f"max={field.max():.3f}, mean={field.mean():.3f}")
    
    # 可視化
    calculator.visualize_fields(fields)
    
    return fields

# ==============================
# 使用例
# ==============================

if __name__ == "__main__":
    # 円柱のマップ生成
    cylinder_fields = generate_geometric_map('cylinder', Re=200)
    
    # 角柱のマップ生成
    square_fields = generate_geometric_map('square', Re=200)
    
    print("\n✨ Geometric Bernoulli Maps generated!")
    print("Ready for GET Wind™ particle simulation!")
