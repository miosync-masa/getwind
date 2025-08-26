#　実装例：任意形状対応版

def compute_physics_params_from_geometry(geometry_data):
    """形状データから物理パラメータを推定"""
    
    # === 曲率ベースの剥離点推定 ===
    curvature = compute_surface_curvature(geometry_data)
    max_curvature_idx = np.argmax(np.abs(curvature))
    separation_angle = geometry_data.angles[max_curvature_idx]
    
    # === 形状係数から推定 ===
    # ブロッケージ比
    blockage = geometry_data.frontal_area / domain_area
    
    # 形状の鋭さ（エッジ検出）
    sharpness = compute_edge_sharpness(geometry_data)
    
    # === 経験式による推定 ===
    if sharpness > 0.8:  # 鋭いエッジ
        # 角柱的な特性
        wake_width_factor = 1.5 + 0.7 * sharpness
        vortex_formation_length = 1.0 + 0.5 * (1 - sharpness)
        base_pressure = -0.8 - 0.6 * sharpness
        St = 0.12 + 0.03 * blockage  # ブロッケージ補正
        
    else:  # 滑らかな形状
        # 円柱的な特性
        wake_width_factor = 1.2 + 0.3 * blockage
        vortex_formation_length = 2.0 - 0.5 * sharpness
        base_pressure = -0.7 - 0.2 * blockage
        St = 0.195 * (1 + 0.1 * blockage)
    
    return {
        'separation_angle': separation_angle,
        'wake_width_factor': wake_width_factor,
        'vortex_formation_length': vortex_formation_length,
        'strouhal_number': St,
        'base_pressure': base_pressure,
        'recovery_length': 10.0 / (1 - sharpness)
    }



import trimesh

def extract_params_from_stl(stl_file):
    """STLファイルから物理パラメータを推定"""
    
    # STL読み込み
    mesh = trimesh.load(stl_file)
    
    # === 形状特徴の計算 ===
    # 1. 投影面積（前面）
    frontal_area = compute_projected_area(mesh, direction=[1, 0, 0])
    
    # 2. 曲率分布
    vertex_curvatures = mesh.vertex_defects  # ガウス曲率
    
    # 3. エッジの鋭さ
    edge_angles = mesh.face_angles
    sharp_edges = np.sum(edge_angles > np.pi * 0.8) / len(edge_angles)
    
    # 4. 形状の対称性
    symmetry = compute_symmetry(mesh)
    
    # === 形状分類 ===
    if sharp_edges > 0.3:
        shape_type = 'bluff_body_sharp'  # 角柱系
        params = square_like_params(sharp_edges, frontal_area)
    elif symmetry > 0.9:
        shape_type = 'axisymmetric'  # 円柱系
        params = cylinder_like_params(vertex_curvatures, frontal_area)
    else:
        shape_type = 'complex'
        params = hybrid_params(mesh)
    
    return params


def estimate_from_cfd_mesh(mesh_data):
    """CFDメッシュの境界層情報から推定"""
    
    # 境界層の成長率から剥離点を推定
    boundary_layer = compute_boundary_layer_thickness(mesh_data)
    
    # 圧力勾配から剥離を検出
    dp_dx = np.gradient(mesh_data.surface_pressure)
    separation_points = np.where(dp_dx > threshold)[0]
    
    # 剥離角度を計算
    separation_angles = mesh_data.surface_angles[separation_points]
    
    return {
        'separation_angle': np.mean(separation_angles),
        'wake_width_factor': estimate_wake_width(boundary_layer),
        # ...
    }

class ShapeToPhysicsNN:
    """形状→物理パラメータのニューラルネット"""
    
    def __init__(self):
        self.model = self.build_model()
        
    def build_model(self):
        # 形状特徴 → 物理パラメータ
        model = Sequential([
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(6)  # 6つの物理パラメータ
        ])
        return model
    
    def predict_params(self, shape_features):
        """形状特徴から物理パラメータを予測"""
        # shape_features: [曲率, エッジ数, アスペクト比, ...]
        params_vector = self.model.predict(shape_features)
        
        return {
            'separation_angle': params_vector[0] * np.pi,
            'wake_width_factor': 1.0 + params_vector[1],
            'vortex_formation_length': params_vector[2] * 5.0,
            'strouhal_number': 0.1 + params_vector[3] * 0.15,
            'base_pressure': -2.0 * params_vector[4],
            'recovery_length': 5.0 + params_vector[5] * 20.0
        }

@dataclass
class ObstacleConfig:
    """障害物の設定（CAD対応版）"""
    shape_type: str  # 'cylinder', 'square', 'cad', 'custom'
    center_x: float = 100.0
    center_y: float = 75.0
    size: float = 20.0
    angle: float = 0.0
    
    # CAD/カスタム形状用
    geometry_file: Optional[str] = None  # STL, OBJ, etc
    shape_features: Optional[Dict] = None  # 事前計算済み特徴

class GeometricBernoulliCalculator:
    def _get_physics_params(self) -> Dict:
        """形状別の物理パラメータ（CAD対応版）"""
        
        if self.obstacle.shape_type == 'cylinder':
            # 既存の円柱パラメータ
            return {...}
            
        elif self.obstacle.shape_type == 'square':
            # 既存の角柱パラメータ
            return {...}
            
        elif self.obstacle.shape_type == 'cad':
            # CADファイルから推定
            if self.obstacle.geometry_file:
                return extract_params_from_stl(self.obstacle.geometry_file)
            else:
                raise ValueError("CAD file not specified")
                
        elif self.obstacle.shape_type == 'custom':
            # 事前計算済みの特徴から推定
            if self.obstacle.shape_features:
                return compute_physics_params_from_features(
                    self.obstacle.shape_features
                )
            else:
                # デフォルト値
                return {
                    'separation_angle': np.pi/2,
                    'wake_width_factor': 1.8,
                    'vortex_formation_length': 1.8,
                    'strouhal_number': 0.18,
                    'base_pressure': -1.0,
                    'recovery_length': 12.0
                }
