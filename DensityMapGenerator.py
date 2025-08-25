#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HIGH PRESSURE!!!™
Density Map Generator for GET Wind™
ベルヌーイの定理を使った密度場マップ生成器

シンプルに、分かりやすく、実用的に！
風洞実験の条件から各種場を事前計算します。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from typing import Tuple, Optional
import json

# ==============================
# Configuration
# ==============================

@dataclass
class ObstacleConfig:
    """障害物の設定"""
    shape_type: str  # 'cylinder', 'square', 'airfoil', 'none'
    center_x: float = 100.0
    center_y: float = 75.0
    size: float = 20.0  # 特性長さ（円柱なら半径）
    angle: float = 0.0  # 迎角[deg]

@dataclass
class FlowConfig:
    """流れの条件（Re=200用）"""
    U_inf: float = 10.0      # 一様流速度 [m/s]
    rho_inf: float = 1.225   # 基準密度 [kg/m³]
    P_inf: float = 101325.0  # 基準圧力 [Pa]
    T_inf: float = 293.0     # 温度 [K]
    nu: float = 2.0          # 動粘性係数 [m²/s] ← Re=200になる！

@dataclass
class GridConfig:
    """計算グリッド設定"""
    nx: int = 300
    ny: int = 150
    x_min: float = 0.0
    x_max: float = 300.0
    y_min: float = 0.0
    y_max: float = 150.0
    
    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.nx
    
    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / self.ny

# ==============================
# Field Calculator (Bernoulli-based)
# ==============================

class DensityFieldCalculator:
    """ベルヌーイの定理による場の計算"""
    
    def __init__(self, obstacle: ObstacleConfig, flow: FlowConfig, grid: GridConfig):
        self.obstacle = obstacle
        self.flow = flow
        self.grid = grid
        
        # グリッド生成
        self.x = np.linspace(grid.x_min, grid.x_max, grid.nx)
        self.y = np.linspace(grid.y_min, grid.y_max, grid.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # 気体定数
        self.R = 287.0  # J/(kg·K)
        
    def calculate_all_fields(self) -> dict:
        """全ての場を計算"""
        print("Calculating velocity field...")
        u, v = self.calculate_velocity_field()
        
        print("Calculating pressure field...")
        pressure = self.calculate_pressure_field(u, v)
        
        print("Calculating density field...")
        density = self.calculate_density_field(pressure)
        
        print("Calculating separation tendency...")
        separation = self.calculate_separation_field(u, v)
        
        print("Calculating vorticity potential...")
        vorticity = self.calculate_vorticity_potential(u, v)
        
        print("Calculating wake region...")
        wake = self.calculate_wake_region()
        
        print("Calculating boundary layer...")
        boundary = self.calculate_boundary_layer()
        
        return {
            'velocity_u': u,
            'velocity_v': v,
            'pressure': pressure,
            'density': density,
            'separation': separation,
            'vorticity_potential': vorticity,
            'wake_region': wake,
            'boundary_layer': boundary
        }
    
    def calculate_velocity_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """速度場の計算（ポテンシャル流）"""
        u = np.ones_like(self.X) * self.flow.U_inf
        v = np.zeros_like(self.Y)
        
        if self.obstacle.shape_type == 'none':
            return u, v
        
        # 円柱の場合のポテンシャル流
        if self.obstacle.shape_type == 'cylinder':
            # 相対位置
            dx = self.X - self.obstacle.center_x
            dy = self.Y - self.obstacle.center_y
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            
            # 円柱の外側のみ計算
            R = self.obstacle.size
            outside = r > R
            
            # ポテンシャル流の速度場
            # u = U∞(1 - R²/r² cos(2θ))
            # v = -U∞ R²/r² sin(2θ)
            u_r = self.flow.U_inf * (1 - (R/r)**2) * np.cos(theta)
            u_theta = -self.flow.U_inf * (1 + (R/r)**2) * np.sin(theta)
            
            # 直交座標に変換
            u[outside] = u_r[outside] * np.cos(theta[outside]) - u_theta[outside] * np.sin(theta[outside])
            v[outside] = u_r[outside] * np.sin(theta[outside]) + u_theta[outside] * np.cos(theta[outside])
            
            # 円柱内部は速度ゼロ
            inside = r <= R
            u[inside] = 0
            v[inside] = 0
            
        elif self.obstacle.shape_type == 'square':
            # 相対位置
            dx = self.X - self.obstacle.center_x
            dy = self.Y - self.obstacle.center_y
            
            # 角柱内部
            inside = (np.abs(dx) <= self.obstacle.size) & (np.abs(dy) <= self.obstacle.size)
            
            # 角柱外部のポテンシャル流（近似解）
            # 角からの距離
            corner_dist = np.sqrt((np.abs(dx) - self.obstacle.size)**2 + 
                                 (np.abs(dy) - self.obstacle.size)**2)
            
            # 流れの偏向
            deflection_x = np.where(np.abs(dx) > self.obstacle.size,
                                    1.0 - (self.obstacle.size**2 / (dx**2 + 1e-8)),
                                    0.0)
            deflection_y = np.where(np.abs(dy) > self.obstacle.size,
                                    1.0 - (self.obstacle.size**2 / (dy**2 + 1e-8)),
                                    0.0)
            
            # 速度場の修正
            u = u * deflection_x
            v = v * deflection_y
            
            # エッジでの剥離点（角の位置で固定）
            edge_mask = (corner_dist < 5.0) & ~inside
            u[edge_mask] *= 0.5  # エッジでの速度低下
            v[edge_mask] *= 0.5
            
            # 内部は速度ゼロ
            u[inside] = 0
            v[inside] = 0
            
        return u, v
    
    def calculate_pressure_field(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """圧力場の計算（ベルヌーイの定理）"""
        # P + 0.5*ρ*V² = P∞ + 0.5*ρ*U∞²
        V_squared = u**2 + v**2
        U_inf_squared = self.flow.U_inf**2
        
        # 無次元化
        Cp = 1.0 - V_squared / U_inf_squared  # 圧力係数
        
        # 圧力比 (P/P_inf)
        pressure_ratio = 1.0 + 0.5 * Cp
        
        # 物理的な範囲に制限
        pressure_ratio = np.clip(pressure_ratio, 0.5, 2.0)
        
        return pressure_ratio
    
    def calculate_density_field(self, pressure_ratio: np.ndarray) -> np.ndarray:
        """密度場の計算（理想気体）"""
        # P = ρRT より ρ/ρ_inf = P/P_inf （温度一定）
        density_ratio = pressure_ratio
        
        # 後流での密度低下を追加（実験的補正）
        if self.obstacle.shape_type != 'none':
            wake_mask = self.calculate_wake_region()
            # 後流では密度が10-30%低下
            density_ratio = density_ratio * (1 - 0.2 * wake_mask)
        
        # 物理的な範囲に制限
        density_ratio = np.clip(density_ratio, 0.5, 2.0)
        
        return density_ratio
    
    def calculate_separation_field(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """剥離傾向場の計算"""
        separation = np.zeros_like(self.X)
        
        if self.obstacle.shape_type == 'none':
            return separation
        
        # 速度勾配から剥離傾向を推定
        du_dx = np.gradient(u, axis=0)
        du_dy = np.gradient(u, axis=1)
        dv_dx = np.gradient(v, axis=0)
        dv_dy = np.gradient(v, axis=1)
        
        # 逆圧力勾配の指標
        adverse_pressure = du_dx < 0
        
        if self.obstacle.shape_type == 'cylinder':
            # 円柱の場合
            dx = self.X - self.obstacle.center_x
            dy = self.Y - self.obstacle.center_y
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            
            # 境界層内
            in_boundary = (r > self.obstacle.size) & (r < self.obstacle.size + 10)
            
            # 60度以降で剥離しやすい
            separation_angle = np.abs(theta) > np.pi/3
            
            separation[in_boundary & separation_angle & adverse_pressure] = 0.8
            
            # 90度以降はほぼ確実に剥離
            strong_separation = np.abs(theta) > np.pi/2
            separation[in_boundary & strong_separation] = 1.0
            
        elif self.obstacle.shape_type == 'square':
            dx = self.X - self.obstacle.center_x
            dy = self.Y - self.obstacle.center_y
            
            # 前縁エッジ（左側！）← ここが重要！
            front_edge = (np.abs(dx + self.obstacle.size) < 5) & \
                         (np.abs(dy) <= self.obstacle.size)
            separation[front_edge] = 1.0
            
            # 4つの角での剥離も追加
            corners = ((np.abs(np.abs(dx) - self.obstacle.size) < 3) & \
                       (np.abs(np.abs(dy) - self.obstacle.size) < 3))
            separation[corners] = 1.0
            
        return separation
    
    def calculate_vorticity_potential(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """渦度生成ポテンシャル場"""
        # 渦度 ω = ∂v/∂x - ∂u/∂y
        dv_dx = np.gradient(v, axis=0)
        du_dy = np.gradient(u, axis=1)
        vorticity = dv_dx - du_dy
        
        # 渦度の絶対値（渦の強さ）
        vorticity_potential = np.abs(vorticity)
        
        # スムージング
        vorticity_potential = gaussian_filter(vorticity_potential, sigma=1.0)
        
        return vorticity_potential
    
    def calculate_wake_region(self) -> np.ndarray:
        """後流領域のマスク（0-1）"""
        wake = np.zeros_like(self.X)
        
        if self.obstacle.shape_type == 'none':
            return wake
        
        # 障害物の後方
        behind = self.X > self.obstacle.center_x
        
        # Y方向の広がり（距離とともに拡大）
        distance_from_obstacle = self.X - self.obstacle.center_x
        wake_width = self.obstacle.size + 0.3 * np.maximum(distance_from_obstacle, 0)
        
        within_wake_width = np.abs(self.Y - self.obstacle.center_y) < wake_width
        
        # 後流の強度（距離とともに減衰）
        wake_strength = np.exp(-distance_from_obstacle / (5 * self.obstacle.size))
        wake_strength = np.clip(wake_strength, 0, 1)
        
        wake[behind & within_wake_width] = wake_strength[behind & within_wake_width]
        
        return wake
    
    def calculate_boundary_layer(self) -> np.ndarray:
        """境界層厚さ分布"""
        boundary = np.zeros_like(self.X)
        
        if self.obstacle.shape_type == 'none':
            return boundary
        
        # Blasius境界層の理論
        # δ ≈ 5.0 * x / sqrt(Re_x)
        
        if self.obstacle.shape_type in ['cylinder', 'square']:
            dx = self.X - self.obstacle.center_x
            dy = self.Y - self.obstacle.center_y
            r = np.sqrt(dx**2 + dy**2)
            
            # 表面からの距離
            if self.obstacle.shape_type == 'cylinder':
                dist_from_surface = r - self.obstacle.size
            else:  # square
                dist_from_surface = np.minimum(
                    np.abs(dx) - self.obstacle.size,
                    np.abs(dy) - self.obstacle.size
                )
            
            # 境界層内（表面から10単位以内）
            in_boundary = (dist_from_surface > 0) & (dist_from_surface < 10)
            
            # 境界層厚さ（簡易モデル）
            Re_local = self.flow.U_inf * dist_from_surface / self.flow.nu
            delta = 5.0 * dist_from_surface / np.sqrt(np.maximum(Re_local, 1))
            
            boundary[in_boundary] = np.clip(delta[in_boundary], 0, 10)
        
        return boundary

# ==============================
# Data Export/Import
# ==============================

class FieldData:
    """計算済み場のデータ管理"""
    
    def __init__(self, fields: dict, obstacle: ObstacleConfig, 
                 flow: FlowConfig, grid: GridConfig):
        self.fields = fields
        self.obstacle = obstacle
        self.flow = flow
        self.grid = grid
        
    def save(self, filename: str):
        """NPZファイルとして保存"""
        # メタデータをJSON文字列に
        metadata = {
            'obstacle': self.obstacle.__dict__,
            'flow': self.flow.__dict__,
            'grid': self.grid.__dict__
        }
        
        # 全データを保存
        save_dict = {**self.fields}
        save_dict['metadata'] = json.dumps(metadata)
        
        np.savez_compressed(filename, **save_dict)
        print(f"Saved to {filename}")
    
    @classmethod
    def load(cls, filename: str):
        """NPZファイルから読み込み"""
        data = np.load(filename, allow_pickle=True)
        
        # メタデータ復元
        metadata = json.loads(str(data['metadata']))
        obstacle = ObstacleConfig(**metadata['obstacle'])
        flow = FlowConfig(**metadata['flow'])
        grid = GridConfig(**metadata['grid'])
        
        # フィールドデータ復元
        fields = {key: data[key] for key in data.files if key != 'metadata'}
        
        return cls(fields, obstacle, flow, grid)
    
    def visualize(self, figsize=(15, 10)):
        """全フィールドの可視化"""
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.flatten()
        
        # カラーマップの設定
        field_configs = [
            ('density', 'RdBu_r', 0.5, 1.5, 'Density Ratio'),
            ('pressure', 'RdBu_r', 0.5, 1.5, 'Pressure Ratio'),
            ('separation', 'hot', 0, 1, 'Separation Tendency'),
            ('vorticity_potential', 'RdBu_r', None, None, 'Vorticity Potential'),
            ('wake_region', 'Blues', 0, 1, 'Wake Region'),
            ('boundary_layer', 'viridis', 0, 10, 'Boundary Layer'),
        ]
        
        # 速度場はベクトル表示
        ax = axes[0]
        skip = 5  # ベクトルの間引き
        u = self.fields['velocity_u'][::skip, ::skip]
        v = self.fields['velocity_v'][::skip, ::skip]
        x = np.linspace(self.grid.x_min, self.grid.x_max, self.grid.nx)[::skip]
        y = np.linspace(self.grid.y_min, self.grid.y_max, self.grid.ny)[::skip]
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        speed = np.sqrt(u**2 + v**2)
        ax.quiver(X, Y, u, v, speed, cmap='viridis', scale=50)
        ax.set_title('Velocity Field')
        ax.set_aspect('equal')
        self.add_obstacle_outline(ax)
        
        # その他のフィールド
        for i, (field_name, cmap, vmin, vmax, title) in enumerate(field_configs, 1):
            if i >= len(axes):
                break
                
            ax = axes[i]
            field = self.fields[field_name]
            
            im = ax.imshow(field.T, origin='lower', cmap=cmap,
                          vmin=vmin, vmax=vmax,
                          extent=[self.grid.x_min, self.grid.x_max,
                                 self.grid.y_min, self.grid.y_max],
                          aspect='equal')
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046)
            self.add_obstacle_outline(ax)
        
        # 最後の軸は流線
        ax = axes[-1]
        x = np.linspace(self.grid.x_min, self.grid.x_max, self.grid.nx)
        y = np.linspace(self.grid.y_min, self.grid.y_max, self.grid.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        ax.streamplot(X.T, Y.T, 
                     self.fields['velocity_u'].T,
                     self.fields['velocity_v'].T,
                     density=1.5, color='k', linewidth=0.5)
        ax.set_title('Streamlines')
        ax.set_aspect('equal')
        self.add_obstacle_outline(ax)
        
        plt.tight_layout()
        return fig
    
    def add_obstacle_outline(self, ax):
        """障害物の輪郭を追加"""
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

# ==============================
# Main Function
# ==============================

def main():
    """使用例"""
    
    # 設定
    obstacle = ObstacleConfig(
        shape_type='cylinder',  # 'cylinder', 'square', 'airfoil', 'none'
        center_x=100,
        center_y=75,
        size=20
    )
    
    flow = FlowConfig(
        U_inf=10.0,  # 風速 10 m/s
        rho_inf=1.225,
        P_inf=101325.0,
        T_inf=293.0,
        nu=2.0  # Re=200になるように調整
    )
    
    grid = GridConfig(
        nx=300,
        ny=150,
        x_max=300,
        y_max=150
    )
    
    # 計算
    print(f"Calculating fields for {obstacle.shape_type}...")
    Re = flow.U_inf * obstacle.size * 2 / flow.nu
    print(f"Reynolds number: Re = {Re:.1f}")
    
    calculator = DensityFieldCalculator(obstacle, flow, grid)
    fields = calculator.calculate_all_fields()
    
    # データ管理
    field_data = FieldData(fields, obstacle, flow, grid)
    
    # 保存
    filename = f"{obstacle.shape_type}_Re{int(Re)}_fields.npz"
    field_data.save(filename)
    
    # 可視化
    fig = field_data.visualize()
    plt.savefig(f"{obstacle.shape_type}_fields.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nField statistics:")
    for name, field in fields.items():
        if field.ndim == 2:  # スカラー場のみ
            print(f"  {name:20s}: min={field.min():.3f}, max={field.max():.3f}, mean={field.mean():.3f}")
    
    return field_data

if __name__ == "__main__":
    field_data = main()
    print("\n✨ Density map generation complete! Ready for GET Wind™!")
