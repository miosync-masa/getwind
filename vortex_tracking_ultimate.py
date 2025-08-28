#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GET Wind™ Vortex Analysis — Refactored Ultimate (DBSCAN + Strouhal)
環ちゃん & ご主人さま 完全版 💕

主な改善:
- 構成を関数/クラス単位に整理して可読性を向上
- DBSCAN 検出の auto-eps を堅牢化（第2近傍×係数 + クリップ）
- Q基準のしきい・最小粒子数を引数化
- 揚力係数 CL の等角ビン積分をユニットテストしやすく分離
- rFFT / Welch を選択可能、ピーク精緻化（放物線補間）
- U_eff 推定を堅牢化（逆流除外・中央値・ドメイン境界チェック）
- 物理スケール（m/unit, s/step）を config から安全に読取
- 例外とフォールバックを簡潔化

実行例:
  python vortex_analysis_refactored.py --file simulation_results_v63_cylinder.npz --method rfft
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch, find_peaks

# ==============================
# データ構造
# ==============================

@dataclass
class Vortex:
    center: np.ndarray      # (x, y)
    n_particles: int
    circulation: float
    cluster_id: int

# （必要なら使う）
@dataclass
class VortexSnapshot:
    step: int
    vortices: List[Vortex]
    total_particles: int


# ==============================
# 低レベル・ユーティリティ
# ==============================

def _safe_get_scales(config) -> Tuple[float, float]:
    """config から [m/unit], [s/step] を安全に取得（フォールバック付）"""
    L = getattr(config, "scale_m_per_unit", 0.01)  # 1 unit = 1 cm 既定
    T = getattr(config, "scale_s_per_step", 1.0)   # dt 自体が秒なら 1
    return float(L), float(T)


def _parabolic_peak_index(power: np.ndarray) -> float:
    """離散ピーク周り 3点の放物線補間でサブビン推定。戻り値はインデックス。"""
    k = int(np.argmax(power))
    if 0 < k < len(power) - 1:
        y1, y2, y3 = power[k - 1], power[k], power[k + 1]
        denom = (y1 - 2 * y2 + y3)
        if denom != 0:
            delta = 0.5 * (y1 - y3) / denom
            return float(k + np.clip(delta, -0.5, 0.5))
    return float(k)


# ==============================
# 渦の循環推定
# ==============================

def compute_circulation(Lambda_F: np.ndarray, positions: np.ndarray, center: np.ndarray) -> float:
    """接線方向速度の重み付き平均で循環を簡便推定。"""
    rel = positions - center
    r = np.linalg.norm(rel, axis=1) + 1e-8
    tangent = np.stack([-rel[:, 1], rel[:, 0]], axis=1) / r[:, None]
    v_t = np.sum(Lambda_F * tangent, axis=1)
    w = np.exp(-r / 10.0)
    return float(np.sum(v_t * w) / np.sum(w))


# ==============================
# DBSCAN 渦検出
# ==============================

def detect_vortices_dbscan(
    positions: np.ndarray,
    Lambda_F: np.ndarray,
    Q_criterion: np.ndarray,
    active_mask: np.ndarray,
    *,
    eps: Optional[float] = None,
    min_samples: int = 8,
    Q_threshold: float = 0.2,
    auto_eps: bool = True,
    auto_eps_k: int = 5,
    auto_eps_coeff: float = 3.0,
    auto_eps_clip: Tuple[float, float] = (8.0, 60.0),
) -> List[Vortex]:
    """DBSCAN で渦クラスタを検出（auto-eps 強化版）。

    positions: (N,2 or 3) を想定（z を無視して 2D に投影）
    """
    mask = (active_mask.astype(bool)) & (Q_criterion > Q_threshold)
    pts = positions[mask][:, :2]
    vel = Lambda_F[mask][:, :2]

    if len(pts) < min_samples:
        return []

    if auto_eps and eps is None:
        k = min(auto_eps_k, max(2, len(pts) - 1))
        nbrs = NearestNeighbors(n_neighbors=k).fit(pts)
        dists, _ = nbrs.kneighbors(pts)
        base = np.median(dists[:, 1]) if dists.shape[1] > 1 else 25.0
        eps = float(np.clip(base * auto_eps_coeff, *auto_eps_clip))
    elif eps is None:
        eps = 25.0

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)

    vortices: List[Vortex] = []
    for cid in set(int(l) for l in labels if l != -1):
        m = labels == cid
        if m.sum() < min_samples:
            continue
        cpos = pts[m]
        cvel = vel[m]
        center = cpos.mean(axis=0)
        circ = compute_circulation(cvel, cpos, center)
        vortices.append(Vortex(center=center, n_particles=int(m.sum()), circulation=float(circ), cluster_id=cid))
    return vortices


# ==============================
# シンプル・トラッカー
# ==============================

class SimpleVortexTracker:
    def __init__(self, matching_threshold: float = 40.0):
        self.matching_threshold = float(matching_threshold)
        self.next_id = 0
        self.tracks: Dict[int, List[Tuple[int, np.ndarray, float]]] = {}

    def update(self, vortices: Sequence[Vortex], step: int) -> None:
        if not vortices:
            # 古いトラックの掃除
            self.tracks = {tid: tr for tid, tr in self.tracks.items() if (step - tr[-1][0]) < 100}
            return

        cand = np.array([v.center for v in vortices])
        used: set[int] = set()
        new_tracks: Dict[int, List[Tuple[int, np.ndarray, float]]] = {}

        # 既存トラック延長
        for tid, tr in self.tracks.items():
            if not tr:
                continue
            last_step, last_pos, _ = tr[-1]
            # 簡易予測（下流へ）
            pred = last_pos + np.array([10.0 * (step - last_step) * 0.02, 0.0])
            best_i, best_d = None, np.inf
            for i, p in enumerate(cand):
                if i in used:
                    continue
                if p[0] < last_pos[0] - 20:  # 強い逆流は無視
                    continue
                d = np.linalg.norm(p - pred)
                if d < self.matching_threshold and d < best_d:
                    best_i, best_d = i, d
            if best_i is not None:
                v = vortices[best_i]
                new_tracks[tid] = tr + [(step, v.center, v.circulation)]
                used.add(best_i)

        # 新規トラック（障害物背後のみ）
        for i, v in enumerate(vortices):
            if i in used:
                continue
            cx = float(getattr(self, "obstacle_cx", 100.0))
            if cx - 20 < v.center[0] < cx + 60 and abs(v.circulation) > 1.0 and v.n_particles > 8:
                tid = self.next_id
                self.next_id += 1
                new_tracks[tid] = [(step, v.center, v.circulation)]

        # 古い/空トラック除去
        self.tracks = {tid: tr for tid, tr in new_tracks.items() if tr and (step - tr[-1][0]) < 100}


# ==============================
# 揚力係数（CL）
# ==============================

def lift_coefficient_ring_binned(state, config, n_bins: int = 64) -> float:
    pos = state['position'] if isinstance(state, dict) else state.position
    vel = state['Lambda_F'] if isinstance(state, dict) else state.Lambda_F
    act = state['is_active'] if isinstance(state, dict) else state.is_active

    dx = pos[:, 0] - float(getattr(config, 'obstacle_center_x', 100.0))
    dy = pos[:, 1] - float(getattr(config, 'obstacle_center_y', 75.0))
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)

    R = float(getattr(config, 'obstacle_size', 20.0))
    ring = (r > R * 1.00) & (r < R * 1.30) & act
    if np.sum(ring) < 32:
        return 0.0

    U_in = float(getattr(config, 'Lambda_F_inlet', 1.0))
    vel_mag = np.linalg.norm(vel[ring], axis=1)
    Cp = 1.0 - (vel_mag / U_in) ** 2

    th = theta[ring]
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.digitize(th, bins) - 1

    up, lo = [], []
    for k in range(n_bins):
        m = idx == k
        if not np.any(m):
            continue
        thm = np.median(th[m])
        rep = np.median(Cp[m])
        term = rep * np.sin(thm)
        (up if thm > 0 else lo).append(term)

    up_m = np.mean(up) if up else 0.0
    lo_m = np.mean(lo) if lo else 0.0
    return float((up_m - lo_m) * 2.0)


def compute_lift_coefficient(state, config, method: str = 'binned') -> float:
    if method == 'binned':
        return lift_coefficient_ring_binned(state, config)
    # 追加メソッドは必要に応じて
    return lift_coefficient_ring_binned(state, config)


# ==============================
# U_eff / Re_eff 推定
# ==============================

def estimate_Ueff(states: Sequence, config, x_probe: Optional[float] = None) -> float:
    cx = float(getattr(config, 'obstacle_center_x', 100.0))
    R = float(getattr(config, 'obstacle_size', 20.0))
    if x_probe is None:
        x_probe = cx - 3.0 * R
    x_probe = float(max(20.0, x_probe))

    vals = []
    s0, s1 = 1000, min(3000, len(states))
    for st in states[s0:s1]:
        pos = st['position'] if isinstance(st, dict) else st.position
        vel = st['Lambda_F'] if isinstance(st, dict) else st.Lambda_F
        act = st['is_active'] if isinstance(st, dict) else st.is_active
        m = act & (np.abs(pos[:, 0] - x_probe) < 3.0)
        if np.sum(m) > 5:
            u = vel[m, 0]
            u_pos = u[u > 0]
            if len(u_pos) > 0:
                vals.append(np.mean(u_pos))

    U_in = float(getattr(config, 'Lambda_F_inlet', 1.0))
    if vals:
        med = float(np.median(vals))
        return float(np.clip(med, 0.7 * U_in, 1.0 * U_in))
    return 0.9 * U_in


def compute_effective_reynolds(states: Sequence, config, n_samples: int = 100) -> Tuple[float, float]:
    # 簡易モデル：近傍から一定の有効粘性を仮定
    nu_eff = float(getattr(config, 'viscosity_factor', 1.0) * 0.05)
    D = 2.0 * float(getattr(config, 'obstacle_size', 20.0))
    U = float(getattr(config, 'Lambda_F_inlet', 1.0))
    Re = U * D / max(nu_eff, 1e-9)
    return float(Re), float(nu_eff)


# ==============================
# Strouhal 解析（物理単位）
# ==============================

def compute_strouhal(
    states: Sequence,
    config,
    *,
    debug: bool = True,
    method: str = 'rfft',
) -> float:
    print("\n📊 Computing lift coefficient time series...")

    L_scale, T_scale = _safe_get_scales(config)
    D_phys = 2.0 * float(getattr(config, 'obstacle_size', 20.0)) * L_scale
    dt_phys = float(getattr(config, 'dt', 0.01)) * T_scale

    # CL 時系列
    CL = [compute_lift_coefficient(st, config, 'binned') for st in states]
    CL = np.asarray(CL, float)
    CL = CL[1000:]  # 初期過渡を除去
    if len(CL) < 500:
        print("Warning: Not enough samples after transient removal.")
        return 0.0

    t_phys = np.arange(len(CL)) * dt_phys
    CL -= CL.mean()

    # U_eff
    print("\n  Estimating effective velocity U_eff...")
    U_eff_grid = estimate_Ueff(states, config)
    U_eff_phys = U_eff_grid * L_scale / T_scale
    U_inlet_phys = float(getattr(config, 'Lambda_F_inlet', 1.0)) * L_scale / T_scale
    print(f"  U_eff (grid): {U_eff_grid:.3f} unit/step")
    print(f"  U_eff (physical): {U_eff_phys:.3f} m/s")
    print(f"  U_inlet (physical): {U_inlet_phys:.3f} m/s")

    # 初期周波数推定（自己相関）
    sig = CL.copy()
    ac = np.correlate(sig - sig.mean(), sig - sig.mean(), mode='full')[len(sig) - 1:]
    ac /= ac.max() if ac.max() > 0 else 1.0
    peaks, _ = find_peaks(ac, distance=max(1, int(0.1 / max(dt_phys, 1e-6))))
    if len(peaks) > 1:
        f0 = 1.0 / (peaks[1] * dt_phys)
    else:
        f0 = 0.2 * U_eff_phys / max(D_phys, 1e-9)
    print(f"  Initial frequency estimate: {f0:.4f} Hz")

    # スペクトル
    if method == 'rfft':
        win = np.hanning(len(CL))
        xw = CL * win
        n_pad = 1 << int(np.ceil(np.log2(len(xw) * 4)))
        spec = np.fft.rfft(xw, n=n_pad)
        freqs = np.fft.rfftfreq(n_pad, d=dt_phys)
        power = np.abs(spec) ** 2
    elif method == 'welch':
        nperseg = min(4096, len(CL))
        freqs, power = welch(CL, fs=1.0 / dt_phys, window='hann', nperseg=nperseg, noverlap=nperseg // 2)
    else:
        raise ValueError(f"Unknown method: {method}")

    # ピーク探索（狭窓）
    fmin, fmax = 0.7 * f0, 1.3 * f0
    vr = (freqs > fmin) & (freqs < fmax)
    if not np.any(vr):
        print(f"Warning: No valid peak in [{fmin:.3f},{fmax:.3f}] Hz")
        return 0.0

    vf, vp = freqs[vr], power[vr]
    kref = _parabolic_peak_index(vp)
    peak_freq = float(np.interp(kref, np.arange(len(vf)), vf))

    # Strouhal
    St = peak_freq * D_phys / max(U_eff_phys, 1e-9)

    # 物理パラメータも出力
    nu_air = 1.5e-5
    Re_phys = U_eff_phys * D_phys / nu_air

    if debug:
        print("\n✨ Ultimate Physical Analysis:")
        print("  === Frequency Analysis ===")
        print(f"  Peak frequency: {peak_freq:.4f} Hz")
        print(f"  Expected freq (St=0.195): {0.195 * U_eff_phys / D_phys:.4f} Hz")
        print("  === Physical Parameters ===")
        print(f"  D (physical): {D_phys:.4f} m")
        print(f"  U_eff (physical): {U_eff_phys:.3f} m/s")
        print(f"  Reynolds (physical): {Re_phys:.0f}")
        print("  === Strouhal Number ===")
        print(f"  Computed St: {St:.4f}")
        print("  Target St (Re=200): 0.195")
        print(f"  Error: {abs(St - 0.195) / 0.195 * 100:.1f}%")

        # プロット
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        # 1. 元系列
        t_full = np.arange(len(states)) * dt_phys
        axes[0, 0].plot(t_full, [compute_lift_coefficient(s, config, 'binned') for s in states], lw=0.5)
        axes[0, 0].set_title('Raw CL')
        axes[0, 0].set_xlabel('Time [s]')
        axes[0, 0].set_ylabel('CL')
        axes[0, 0].grid(alpha=0.3)
        # 2. 処理後
        axes[0, 1].plot(t_phys, CL, lw=0.6)
        axes[0, 1].set_title('CL (detrended, transient removed)')
        axes[0, 1].set_xlabel('Time [s]')
        axes[0, 1].set_ylabel('CL')
        axes[0, 1].grid(alpha=0.3)
        # 3. スペクトル（線形）
        m = freqs < 2.0
        axes[0, 2].plot(freqs[m], power[m])
        axes[0, 2].axvline(peak_freq, ls='--', c='r', label=f'Peak {peak_freq:.3f} Hz')
        axes[0, 2].axvspan(fmin, fmax, alpha=0.2, color='yellow')
        axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)
        axes[0, 2].set_title(f'Power Spectrum ({method.upper()})')
        # 4. スペクトル（対数）
        axes[1, 0].semilogy(freqs[m], power[m])
        axes[1, 0].axvline(peak_freq, ls='--', c='r')
        axes[1, 0].axvline(0.195 * U_eff_phys / D_phys, ls=':', c='g')
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].set_title('Power (log)')
        # 5. St vs Re（簡易）
        Re_range = np.array([100, 150, 200, 250, 300])
        St_emp = 0.195 * np.ones_like(Re_range)
        axes[1, 1].plot(Re_range, St_emp, 'g-', label='Empirical (cyl)')
        axes[1, 1].scatter([Re_phys], [St], c='r', s=80, zorder=5, label=f'Sim (Re={Re_phys:.0f})')
        axes[1, 1].set_xlim(50, 350); axes[1, 1].set_ylim(0.1, 0.3)
        axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)
        axes[1, 1].set_title('St vs Re')
        # 6. サマリー
        domain_h_phys = float(getattr(config, 'domain_height', 150.0)) * L_scale
        blockage = D_phys / max(domain_h_phys, 1e-9)
        txt = (
            f"=== PHYSICAL PARAMETERS ===\n"
            f"Length scale: {L_scale*1000:.3f} mm/unit\n"
            f"Time scale: {T_scale*1000:.1f} ms/step\n"
            f"─────────────────\n"
            f"Cylinder diameter: {D_phys*1000:.1f} mm\n"
            f"Domain height: {domain_h_phys*1000:.0f} mm\n"
            f"Blockage: {blockage:.1%}\n"
            f"─────────────────\n"
            f"U_eff: {U_eff_phys:.3f} m/s\n"
            f"U_inlet: {U_inlet_phys:.3f} m/s\n"
            f"Reynolds: {Re_phys:.0f}\n"
            f"─────────────────\n"
            f"Shedding freq: {peak_freq:.3f} Hz\n"
            f"Shedding period: {1/peak_freq:.3f} s\n"
            f"Strouhal: {St:.4f}\n"
            f"Error from 0.195: {(St-0.195)/0.195*100:+.1f}%"
        )
        axes[1, 2].text(0.1, 0.5, txt, transform=axes[1, 2].transAxes,
                        fontsize=10, va='center', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1, 2].axis('off'); axes[1, 2].set_title('Summary')
        plt.tight_layout()
        plt.savefig('strouhal_analysis_physical.png', dpi=150, bbox_inches='tight')
        print("  📊 Plot saved to 'strouhal_analysis_physical.png'")

    return float(St)


# ==============================
# 可視化: きれいな軌跡
# ==============================

def plot_clean_vortex_trajectories(tracker: SimpleVortexTracker, figsize=(14, 7)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    valid = []
    for tid, tr in tracker.tracks.items():
        if len(tr) < 10:
            continue
        P = np.array([p for (_, p, _) in tr])
        C = np.array([c for (*_, c) in tr])
        # 逆流カット
        x = P[:, 0]
        for i in range(1, len(x)):
            if x[i] < x[i - 1] - 15:
                P = P[:i]
                C = C[:i]
                break
        if len(P) < 20:
            continue
        dist = np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))
        if dist < 20 or dist > 800:
            continue
        if np.mean(np.abs(C)) < 0.3:
            continue
        valid.append((tid, P, np.mean(np.abs(C))))

    for idx, (tid, P, mc) in enumerate(valid):
        if len(P) > 5:
            P[:, 0] = gaussian_filter1d(P[:, 0], sigma=1.5)
            P[:, 1] = gaussian_filter1d(P[:, 1], sigma=1.5)
        color = 'red' if P[0, 1] > 75 else 'blue'
        label = 'Upper vortex' if (color == 'red' and idx == 0) else ('Lower vortex' if (color == 'blue' and idx == 0) else None)
        ax.plot(P[:, 0], P[:, 1], color=color, alpha=0.6, lw=2, label=label)
        ax.scatter(P[0, 0], P[0, 1], color=color, s=50, marker='o', zorder=5)
        ax.scatter(P[-1, 0], P[-1, 1], color=color, s=50, marker='s', zorder=5)

    # 参考の理想軌跡
    t = np.linspace(0, 150, 100)
    x0 = 100 + t
    y_up = 75 + 25 * np.sin(2 * np.pi * t / 50) * np.exp(-t / 200)
    y_lo = 75 - 25 * np.sin(2 * np.pi * t / 50 + np.pi) * np.exp(-t / 200)
    ax.plot(x0, y_up, 'r--', alpha=0.2, lw=1, label='Ideal upper')
    ax.plot(x0, y_lo, 'b--', alpha=0.2, lw=1, label='Ideal lower')

    # 障害物（円）
    circle = plt.Circle((100, 75), 20, fill=True, color='gray', alpha=0.25, ec='k')
    ax.add_patch(circle)

    ax.set_xlim(0, 300); ax.set_ylim(0, 150); ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X position'); ax.set_ylabel('Y position')
    ax.set_title('Clean Vortex Trajectories (Karman Vortex Street)')
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig


# ==============================
# メイン処理
# ==============================

def process_simulation_results(
    simulation_file: str = 'simulation_results_v63_cylinder.npz',
    *,
    debug: bool = True,
    fft_method: str = 'rfft',
) -> float:
    print("=" * 70)
    print("GET Wind™ Vortex Analysis — Refactored Ultimate")
    print("環ちゃん & ご主人さま Complete Fix! 💕")
    print("=" * 70)

    # ロード
    print("\n📁 Loading simulation data...")
    print(f"  File: {simulation_file}")

    data = None
    try:
        data = np.load(simulation_file, allow_pickle=True)
    except FileNotFoundError:
        for fb in ['simulation_results_v63.npz', 'simulation_results_v62.npz']:
            try:
                print(f"  Trying fallback: {fb}")
                data = np.load(fb, allow_pickle=True)
                simulation_file = fb
                break
            except FileNotFoundError:
                pass
        if data is None:
            raise FileNotFoundError('No simulation result file found!')

    states = data['states'].tolist() if hasattr(data['states'], 'tolist') else data['states']
    cfg_raw = data['config'].item() if hasattr(data['config'], 'item') else data['config']

    class SimpleConfig:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    config = SimpleConfig(**cfg_raw) if isinstance(cfg_raw, dict) else cfg_raw

    print(f"  Loaded {len(states)} timesteps")
    print(f"  dt = {getattr(config, 'dt', None)}")
    print(f"  Obstacle: center=({getattr(config,'obstacle_center_x',None)}, {getattr(config,'obstacle_center_y',None)}), radius={getattr(config,'obstacle_size',None)}")
    print(f"  Inlet velocity: {getattr(config,'Lambda_F_inlet', None)}")

    # 名目Re（簡易）
    D = 2.0 * float(getattr(config, 'obstacle_size', 20.0))
    Re_nom = float(getattr(config, 'Lambda_F_inlet', 1.0)) * D / max(float(getattr(config, 'viscosity_factor', 1.0)) * 0.05, 1e-9)
    print(f"  Nominal Reynolds number: {Re_nom:.1f}")

    # Strouhal
    St = compute_strouhal(states, config, debug=debug, method=fft_method)

    # 可視化（任意）
    if debug:
        print("\n🔍 Vortex tracking for visualization (subset)...")
        tracker = SimpleVortexTracker(matching_threshold=40.0)
        tracker.obstacle_cx = float(getattr(config, 'obstacle_center_x', 100.0))
        for i, st in enumerate(states[::10]):  # 軽量化のため 1/10 サンプリング
            pos = st['position'] if isinstance(st, dict) else st.position
            vel = st['Lambda_F'] if isinstance(st, dict) else st.Lambda_F
            Qc = st.get('Q_criterion', None) if isinstance(st, dict) else getattr(st, 'Q_criterion', None)
            act = st['is_active'] if isinstance(st, dict) else st.is_active
            if Qc is None:
                # Q 基準が無ければ速度勾配情報が必要。なければスキップ。
                continue
            vort = detect_vortices_dbscan(pos, vel, Qc, act, eps=None, min_samples=8, Q_threshold=0.2, auto_eps=True)
            strong = [v for v in vort if abs(v.circulation) > 1.0 and v.n_particles > 10]
            tracker.update(strong, i)
        fig = plot_clean_vortex_trajectories(tracker)
        plt.savefig('clean_vortex_trajectories_ultimate.png', dpi=150)
        print("  Saved to 'clean_vortex_trajectories_ultimate.png'")

    print("\n" + "=" * 70)
    print("✨ FINAL RESULTS (Ultimate Analysis):")
    print(f"  Strouhal number: {St:.4f}")
    print("  Target (Re=200): 0.195")
    print(f"  Error: {abs(St - 0.195) / 0.195 * 100:.1f}%")
    print(f"  FFT Method: {fft_method.upper()}")
    if 0.18 < St < 0.21:
        print("  🎉 SUCCESS! Strouhal number is within 10% of target!")
    elif 0.15 < St < 0.25:
        print("  ✅ Good! Strouhal number is physically reasonable.")
    else:
        print("  ⚠️  Strouhal number needs further tuning.")
    print("=" * 70)

    return float(St)


# ==============================
# CLI
# ==============================

def main() -> None:
    p = argparse.ArgumentParser(description='GET Wind™ Refactored Ultimate Vortex Analysis')
    p.add_argument('--file', type=str, default='simulation_results_v63_cylinder.npz', help='Simulation result file (.npz)')
    p.add_argument('--method', type=str, choices=['rfft', 'welch'], default='rfft', help='Spectrum estimation method')
    p.add_argument('--no-debug', action='store_true', help='Disable debug plots')
    args = p.parse_args()

    St = process_simulation_results(simulation_file=args.file, debug=not args.no_debug, fft_method=args.method)
    print(f"\n🌀 Final Strouhal number: {St:.4f}")
    print("✨ Ultimate Analysis complete! 💕")


if __name__ == '__main__':
    main()
