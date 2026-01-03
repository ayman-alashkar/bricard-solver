"""
Bricard 6R Linkage — Tensor-Based Kinematic Analysis

Interactive Streamlit application for analyzing the line-symmetric Bricard 6R
overconstrained mechanism using tensor-based closure equations.

Tensor Framework:
    Corotational bases: e^n_i = Q^n · E_i  [Equation (1)]
    Inter-link rotation: Q^{(n+1),n} = Q^{n+1} · (Q^n)^T  [Equation (2)]

Closure Equations:
    Rotational [Eq. 4]:    Q^{2,1}·Q^{3,2}·Q^{4,3}·Q^{5,4}·Q^{6,5}·Q^{1,6} = I
    Translational [Eq. 5]: Σ(R_n·e^n_3 + a_n·e^{n+1}_1) = 0

Line-symmetric constraint [Eq. 8]: θ₁=θ₄, θ₂=θ₅, θ₃=θ₆

Author: Ayman Alashkar
Affiliation: Okinawa Institute of Science and Technology (OIST)
Date: December 2025
"""

import math
import numpy as np
import pandas as pd
import streamlit as st

try:
    import altair as alt
    _HAS_ALTAIR = True
except ImportError:
    _HAS_ALTAIR = False


# ============================================================
# TENSOR-BASED KINEMATICS
# ============================================================

def tensor_Q(theta: float, alpha: float) -> np.ndarray:
    """
    Inter-link rotation tensor Q^{n+1,n}.
    
    Parameters
    ----------
    theta : float
        Joint angle (rotation about z-axis)
    alpha : float
        Twist angle (rotation about x-axis)
    
    Returns
    -------
    Q : ndarray (3×3)
        Rotation tensor components
    """
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    
    return np.array([
        [ct, -ca*st,  sa*st],
        [st,  ca*ct, -sa*ct],
        [0.0,    sa,     ca]
    ], dtype=float)


def build_tensor_bases(theta6: np.ndarray, alpha6: np.ndarray):
    """
    Build cumulative rotation tensors Q^{n,0}.
    
    Parameters
    ----------
    theta6 : ndarray
        Joint angles [θ₁, θ₂, θ₃, θ₄, θ₅, θ₆]
    alpha6 : ndarray
        Twist angles [α₁, α₂, α₃, α₄, α₅, α₆]
    
    Returns
    -------
    Q_list : list
        Individual rotation tensors Q^{n+1,n}
    Q_cum : list
        Cumulative rotations Q^{n,0}
    """
    Q_list = [tensor_Q(theta6[n], alpha6[n]) for n in range(6)]
    
    Q_cum = [np.eye(3, dtype=float)]
    Qc = np.eye(3, dtype=float)
    for n in range(6):
        Qc = Qc @ Q_list[n]
        Q_cum.append(Qc.copy())
    
    return Q_list, Q_cum


def tensor_closure_residual(theta6: np.ndarray, alpha6: np.ndarray,
                            a6: np.ndarray, R6: np.ndarray) -> np.ndarray:
    """
    Compute closure residual using tensor formulation.
    
    Rotational closure [Eq. 4]: Q^{2,1}Q^{3,2}Q^{4,3}Q^{5,4}Q^{6,5}Q^{1,6} = I
    Translational closure [Eq. 5]: Σ(R_n·e^n_3 + a_n·e^{n+1}_1) = 0
    
    Parameters
    ----------
    theta6, alpha6, a6, R6 : ndarray
        Linkage parameters
    
    Returns
    -------
    residual : ndarray (6,)
        [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
    """
    _, Q_cum = build_tensor_bases(theta6, alpha6)
    
    # Rotational closure error [Equation (4)]
    Q6 = Q_cum[6]
    rot = np.array([
        0.5 * (Q6[2, 1] - Q6[1, 2]),
        0.5 * (Q6[0, 2] - Q6[2, 0]),
        0.5 * (Q6[1, 0] - Q6[0, 1])
    ], dtype=float)
    
    # Translational closure [Equation (5)]: Σ(R_n·e^n_3 + a_n·e^{n+1}_1) = 0
    trans = np.zeros(3, dtype=float)
    for n in range(6):
        # e^n_3 = 3rd column of Q^{n,0} (joint axis direction)
        e3_n = Q_cum[n][:, 2]
        # e^{n+1}_1 = 1st column of Q^{n+1,0} (common normal direction)
        e1_np1 = Q_cum[n + 1][:, 0]
        # Equation (5): R_n·e^n_3 + a_n·e^{n+1}_1
        trans += R6[n] * e3_n + a6[n] * e1_np1
    
    return np.concatenate([rot, trans])


# ============================================================
# PARAMETER HANDLING
# ============================================================

def line_sym_params(alpha_deg, a3, R3):
    """Convert user parameters to full 6-link arrays with line symmetry."""
    a6 = np.array([a3[0], a3[1], a3[2], a3[0], a3[1], a3[2]], dtype=float)
    R6 = np.array([R3[0], R3[1], R3[2], R3[0], R3[1], R3[2]], dtype=float)
    alpha6 = np.deg2rad(np.array([
        alpha_deg[0], alpha_deg[1], alpha_deg[2],
        alpha_deg[0], alpha_deg[1], alpha_deg[2]
    ], dtype=float))
    return alpha6, a6, R6


def closure_residual(theta6, alpha_deg, a3, R3):
    """Wrapper for tensor closure residual."""
    alpha6, a6, R6 = line_sym_params(alpha_deg, a3, R3)
    return tensor_closure_residual(np.asarray(theta6, dtype=float), alpha6, a6, R6)


def jacobian_fd(theta6, alpha_deg, a3, R3, h=2e-6):
    """Compute 6×6 Jacobian using finite differences."""
    theta6 = np.asarray(theta6, dtype=float)
    J = np.zeros((6, 6), dtype=float)
    for j in range(6):
        d = np.zeros(6, dtype=float)
        d[j] = h
        J[:, j] = (closure_residual(theta6 + d, alpha_deg, a3, R3) -
                   closure_residual(theta6 - d, alpha_deg, a3, R3)) / (2 * h)
    return J


# ============================================================
# SOLVER
# ============================================================

def wrap_pi(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi


def ang_dist(a: float, b: float) -> float:
    return abs(wrap_pi(a - b))


def pair_dist(p, q) -> float:
    return math.hypot(ang_dist(p[0], q[0]), ang_dist(p[1], q[1]))


def solve_theta2_theta3(theta1, x0, alpha_deg, a3, R3,
                        max_iter=80, tol=1e-10, fd_step=2e-6, lm0=1e-3):
    """
    Solve for θ₂, θ₃ given θ₁ using Levenberg-Marquardt.
    
    Parameters
    ----------
    theta1 : float
        Input angle (radians)
    x0 : tuple
        Initial guess (θ₂, θ₃) in radians
    alpha_deg, a3, R3 : array-like
        Linkage parameters
    
    Returns
    -------
    theta2, theta3 : float
        Solution angles (radians)
    err : float
        Final residual norm
    ok : bool
        Convergence flag
    iters : int
        Number of iterations
    """
    x = np.array([wrap_pi(x0[0]), wrap_pi(x0[1])], dtype=float)
    lam = float(lm0)

    def F_of_x(xx):
        theta6 = np.array([theta1, xx[0], xx[1], theta1, xx[0], xx[1]], dtype=float)
        return closure_residual(theta6, alpha_deg, a3, R3)

    Fx = F_of_x(x)
    err = float(np.linalg.norm(Fx))

    for k in range(max_iter):
        # Finite difference Jacobian (6×2)
        J = np.zeros((6, 2), dtype=float)
        for j in range(2):
            dx = np.zeros(2)
            dx[j] = fd_step
            J[:, j] = (F_of_x(x + dx) - F_of_x(x - dx)) / (2 * fd_step)

        A = J.T @ J + lam * np.eye(2)
        b = -J.T @ Fx
        
        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(A, b, rcond=None)[0]

        x_trial = np.array([wrap_pi(x[0] + delta[0]), wrap_pi(x[1] + delta[1])], dtype=float)
        F_trial = F_of_x(x_trial)
        err_trial = float(np.linalg.norm(F_trial))

        if err_trial < err:
            x, Fx, err = x_trial, F_trial, err_trial
            lam = max(lam / 3.0, 1e-12)
        else:
            lam = min(lam * 10.0, 1e12)

        if err <= tol:
            return float(x[0]), float(x[1]), err, True, k + 1

    return float(x[0]), float(x[1]), err, (err <= tol), max_iter


# ============================================================
# TWO-FORM INITIALIZATION
# ============================================================

def seed_grid(deg_min, deg_max, n):
    vals = np.deg2rad(np.linspace(deg_min, deg_max, n))
    return [(t2, t3) for t2 in vals for t3 in vals]


def find_two_initial(theta1, alpha_deg, a3, R3, seeds, tol, unique_eps):
    sols = []
    for s in seeds:
        th2, th3, err, ok, _ = solve_theta2_theta3(theta1, s, alpha_deg, a3, R3, tol=tol)
        if not ok:
            continue
        cand = (th2, th3)
        if all(pair_dist(cand, (ex["theta2"], ex["theta3"])) >= unique_eps for ex in sols):
            sols.append({"theta2": th2, "theta3": th3, "err": err})
    
    sols.sort(key=lambda d: (d["theta2"], d["theta3"]))
    
    if len(sols) == 0:
        return None, None, 0
    if len(sols) == 1:
        return sols[0], None, 1
    
    A = sols[0]
    dists = [pair_dist((s["theta2"], s["theta3"]), (A["theta2"], A["theta3"])) for s in sols[1:]]
    B = sols[1 + int(np.argmax(dists))]
    return A, B, len(sols)


def local_seeds_around(x, radius_deg=25.0, n=10, seed=0):
    rad = math.radians(radius_deg)
    rng = np.random.default_rng(seed)
    return [(wrap_pi(x[0] + d[0]), wrap_pi(x[1] + d[1]))
            for d in rng.normal(0.0, rad / 3.0, size=(n, 2))]


# ============================================================
# COLLISION DETECTION
# ============================================================

def seg_seg_distance(P0, P1, Q0, Q1, eps=1e-12) -> float:
    """Segment-segment distance."""
    P0, P1 = np.asarray(P0, float), np.asarray(P1, float)
    Q0, Q1 = np.asarray(Q0, float), np.asarray(Q1, float)

    u, v, w = P1 - P0, Q1 - Q0, P0 - Q0
    a, b, c = float(u @ u), float(u @ v), float(v @ v)
    d, e = float(u @ w), float(v @ w)
    D = a * c - b * b

    sN, sD, tN, tD = 0.0, D, 0.0, D

    if D < eps:
        sN, sD, tN, tD = 0.0, 1.0, e, c
    else:
        sN, tN = b*e - c*d, a*e - b*d
        if sN < 0:
            sN, tN, tD = 0.0, e, c
        elif sN > sD:
            sN, tN, tD = sD, e + b, c

    if tN < 0:
        tN = 0.0
        sN = 0.0 if -d < 0 else (sD if -d > a else -d)
        sD = a if -d > 0 and -d <= a else sD
    elif tN > tD:
        tN = tD
        val = -d + b
        sN = 0.0 if val < 0 else (sD if val > a else val)
        sD = a if val > 0 and val <= a else sD

    sc = 0.0 if abs(sD) < eps else sN / sD
    tc = 0.0 if abs(tD) < eps else tN / tD
    return float(np.linalg.norm(w + sc * u - tc * v))


def point_seg_distance(P, A, B, eps=1e-12) -> float:
    """Point-segment distance."""
    P, A, B = np.asarray(P, float), np.asarray(A, float), np.asarray(B, float)
    AB = B - A
    denom = float(AB @ AB)
    if denom < eps:
        return float(np.linalg.norm(P - A))
    t = min(1.0, max(0.0, float((P - A) @ AB / denom)))
    return float(np.linalg.norm(P - A - t * AB))


def polyline_vertices_tensor(theta6, alpha_deg, a3, R3):
    """
    Build polyline vertices using tensor formulation [Equation (5)].
    
    Position increments: R_n along e^n_3, then a_n along e^{n+1}_1
    """
    alpha6, a6, R6 = line_sym_params(alpha_deg, a3, R3)
    _, Q_cum = build_tensor_bases(np.asarray(theta6, dtype=float), alpha6)

    V = np.zeros((13, 3), dtype=float)
    p = np.zeros(3, dtype=float)
    V[0] = p.copy()

    for n in range(6):
        # e^n_3 = 3rd column of Q^{n,0} (joint axis)
        e3_n = Q_cum[n][:, 2]
        # After offset along joint axis
        p_after_R = p + R6[n] * e3_n
        V[2 * n + 1] = p_after_R
        
        # e^{n+1}_1 = 1st column of Q^{n+1,0} (common normal)
        e1_np1 = Q_cum[n + 1][:, 0]
        # After link along common normal
        p = p_after_R + a6[n] * e1_np1
        V[2 * n + 2] = p

    return V - V.mean(axis=0, keepdims=True)


def min_collision_clearance(theta6, alpha_deg, a3, R3, geom):
    """Compute minimum collision clearance."""
    V = polyline_vertices_tensor(theta6, alpha_deg, a3, R3)

    linkR = 0.5 * math.sqrt(geom["link_w"]**2 + geom["link_d"]**2)
    R = {"joint": geom["r_joint"], "offset": geom["r_offset"], "link": linkR}
    segR = [R["offset"] if i % 2 == 0 else R["link"] for i in range(12)]
    spheres = [{"vid": 2*n + 1, "c": V[2*n + 1], "r": R["joint"]} for n in range(6)]

    minClr = float("inf")

    # Segment-segment
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(i - j) <= 1 or (i == 0 and j == 11):
                continue
            d = seg_seg_distance(V[i], V[i+1], V[j], V[j+1])
            minClr = min(minClr, d - segR[i] - segR[j])

    # Segment-sphere
    for i in range(12):
        for s in spheres:
            if s["vid"] in (i, i + 1):
                continue
            d = point_seg_distance(s["c"], V[i], V[i+1])
            minClr = min(minClr, d - s["r"] - segR[i])

    # Sphere-sphere
    for i in range(len(spheres)):
        for j in range(i + 1, len(spheres)):
            d = float(np.linalg.norm(spheres[i]["c"] - spheres[j]["c"]))
            minClr = min(minClr, d - spheres[i]["r"] - spheres[j]["r"])

    return float(minClr) if np.isfinite(minClr) else float("nan")


def contiguous_ranges(theta1_deg, mask, min_span_deg=5.0):
    """Convert boolean mask to contiguous ranges."""
    ranges = []
    in_run, start, prev = False, None, None
    
    for t, m in zip(theta1_deg, mask):
        if m and not in_run:
            in_run, start = True, float(t)
        if not m and in_run:
            in_run = False
            if float(prev) - start > min_span_deg:
                ranges.append((start, float(prev)))
        prev = float(t)
    
    if in_run and prev is not None and float(prev) - start > min_span_deg:
        ranges.append((start, float(prev)))
    
    return ranges


# ============================================================
# MAIN COMPUTATION
# ============================================================

@st.cache_data(show_spinner=False)
def compute_branches_svd_and_ranges(
    alpha_deg, a3, R3,
    th1_min_deg, th1_max_deg, n_points,
    grid_n, unique_eps, tol,
    guessA_deg, guessB_deg,
    recover, recover_radius_deg, recover_trials,
    sv_fd_step, geom, min_span_deg
):
    th1_vals_deg = np.linspace(th1_min_deg, th1_max_deg, n_points)
    th1_vals = np.deg2rad(th1_vals_deg)

    seeds = seed_grid(-180.0, 180.0, grid_n)
    seeds.append((math.radians(guessA_deg[0]), math.radians(guessA_deg[1])))
    seeds.append((math.radians(guessB_deg[0]), math.radians(guessB_deg[1])))

    A0, B0, _ = find_two_initial(th1_vals[0], alpha_deg, a3, R3, seeds, tol, unique_eps)
    if A0 is None:
        return None, None, None, None, "No solution found. Try: (1) Open 'Advanced Settings' and increase 'Grid search resolution', (2) Increase 'Solver tolerance' to 1e-6, or (3) Adjust initial guesses."

    xA = (A0["theta2"], A0["theta3"])
    xB = None if B0 is None else (B0["theta2"], B0["theta3"])

    rowsA, rowsB = [], []

    for t1_deg, t1 in zip(th1_vals_deg, th1_vals):
        # Branch A
        A_ok, A = False, (np.nan, np.nan, np.nan)
        if xA is not None:
            th2, th3, err, ok, _ = solve_theta2_theta3(t1, xA, alpha_deg, a3, R3, tol=tol)
            if ok:
                xA, A_ok, A = (th2, th3), True, (th2, th3, err)
            elif recover:
                best = None
                for s in local_seeds_around(xA, recover_radius_deg, recover_trials, 0):
                    th2r, th3r, errr, okr, _ = solve_theta2_theta3(t1, s, alpha_deg, a3, R3, tol=tol)
                    if okr and (best is None or errr < best[2]):
                        best = (th2r, th3r, errr)
                if best:
                    xA, A_ok, A = (best[0], best[1]), True, best

        # Branch B
        B_ok, B = False, (np.nan, np.nan, np.nan)
        if xB is not None:
            th2, th3, err, ok, _ = solve_theta2_theta3(t1, xB, alpha_deg, a3, R3, tol=tol)
            if ok:
                xB, B_ok, B = (th2, th3), True, (th2, th3, err)
            elif recover:
                best = None
                for s in local_seeds_around(xB, recover_radius_deg, recover_trials, 1):
                    th2r, th3r, errr, okr, _ = solve_theta2_theta3(t1, s, alpha_deg, a3, R3, tol=tol)
                    if okr and (best is None or errr < best[2]):
                        best = (th2r, th3r, errr)
                if best:
                    xB, B_ok, B = (best[0], best[1]), True, best

        # SVD and collision
        svA, clrA, physA = [np.nan]*6, np.nan, False
        svB, clrB, physB = [np.nan]*6, np.nan, False

        if A_ok:
            theta6 = np.array([t1, A[0], A[1], t1, A[0], A[1]], dtype=float)
            svA = list(np.linalg.svd(jacobian_fd(theta6, alpha_deg, a3, R3, sv_fd_step), compute_uv=False))
            clrA = min_collision_clearance(theta6, alpha_deg, a3, R3, geom)
            physA = np.isfinite(clrA) and clrA >= 0

        if B_ok:
            theta6 = np.array([t1, B[0], B[1], t1, B[0], B[1]], dtype=float)
            svB = list(np.linalg.svd(jacobian_fd(theta6, alpha_deg, a3, R3, sv_fd_step), compute_uv=False))
            clrB = min_collision_clearance(theta6, alpha_deg, a3, R3, geom)
            physB = np.isfinite(clrB) and clrB >= 0

        rowsA.append({
            "theta1_deg": float(t1_deg),
            "theta2_deg": float(np.rad2deg(A[0])) if A_ok else np.nan,
            "theta3_deg": float(np.rad2deg(A[1])) if A_ok else np.nan,
            "closure_err": float(A[2]) if A_ok else np.nan,
            "ok": bool(A_ok),
            "min_clearance": float(clrA) if np.isfinite(clrA) else np.nan,
            "phys_ok": bool(physA),
            **{f"sv{k}": svA[k] for k in range(6)},
        })
        rowsB.append({
            "theta1_deg": float(t1_deg),
            "theta2_deg": float(np.rad2deg(B[0])) if B_ok else np.nan,
            "theta3_deg": float(np.rad2deg(B[1])) if B_ok else np.nan,
            "closure_err": float(B[2]) if B_ok else np.nan,
            "ok": bool(B_ok),
            "min_clearance": float(clrB) if np.isfinite(clrB) else np.nan,
            "phys_ok": bool(physB),
            **{f"sv{k}": svB[k] for k in range(6)},
        })

    dfA, dfB = pd.DataFrame(rowsA), pd.DataFrame(rowsB)
    
    th = dfA["theta1_deg"].to_numpy(float)
    ranges_A = contiguous_ranges(th, dfA["ok"].to_numpy(bool) & dfA["phys_ok"].to_numpy(bool), min_span_deg)
    ranges_B = contiguous_ranges(th, dfB["ok"].to_numpy(bool) & dfB["phys_ok"].to_numpy(bool), min_span_deg)

    return dfA, dfB, ranges_A, ranges_B, None


def _altair_valid_curve(df: pd.DataFrame, ycol: str, title: str):
    base = alt.Chart(df).encode(
        x=alt.X("theta1_deg:Q", title="θ₁ (deg)"),
        y=alt.Y(f"{ycol}:Q", title=title),
    )
    invalid = base.transform_filter((alt.datum.ok == True) & (alt.datum.phys_ok == False)) \
        .mark_line().encode(color=alt.value("#888888"))
    valid = base.transform_filter((alt.datum.ok == True) & (alt.datum.phys_ok == True)) \
        .mark_line().encode(color=alt.value("#2ca02c"))
    return (invalid + valid).properties(height=250)


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Bricard 6R — Tensor Method", layout="wide")
st.title("Bricard 6R Linkage — Tensor-Based Kinematic Analysis")

st.markdown("""
This application analyzes the **line-symmetric Bricard 6R** overconstrained mechanism
using tensor-based closure equations. Features include:

- Two solution forms (Form I and Form II)
- Singular value decomposition of the Jacobian
- Collision detection for identifying valid motion ranges
""")

with st.sidebar:
    st.header("Linkage Parameters")

    st.subheader("Link Lengths (a)")
    a1 = st.number_input("a₁₂ = a₄₅", value=2.4, step=0.1, format="%.4f")
    a2 = st.number_input("a₂₃ = a₅₆", value=2.9, step=0.1, format="%.4f")
    a3_val = st.number_input("a₃₄ = a₆₁", value=1.5, step=0.1, format="%.4f")

    st.subheader("Twist Angles α (degrees)")
    alpha1 = st.number_input("α₁₂ = α₄₅", value=40.0, step=1.0, format="%.2f")
    alpha2 = st.number_input("α₂₃ = α₅₆", value=80.0, step=1.0, format="%.2f")
    alpha3 = st.number_input("α₃₄ = α₆₁", value=130.0, step=1.0, format="%.2f")

    st.subheader("Joint Offsets (R)")
    R1 = st.number_input("R₁ = R₄", value=0.5, step=0.05, format="%.4f")
    R2 = st.number_input("R₂ = R₅", value=0.55, step=0.05, format="%.4f")
    R3_val = st.number_input("R₃ = R₆", value=0.42, step=0.05, format="%.4f")

    st.subheader("θ₁ Sweep Range")
    th1_min = st.slider("θ₁ min (deg)", -180.0, 180.0, -180.0, 1.0)
    th1_max = st.slider("θ₁ max (deg)", -180.0, 180.0, 180.0, 1.0)
    n_points = st.slider("Number of samples", 61, 721, 361, 20)

    st.subheader("Collision Geometry")
    r_joint = st.number_input("Joint radius", value=0.10, step=0.01, format="%.4f")
    r_offset = st.number_input("Offset radius", value=0.06, step=0.01, format="%.4f")
    link_w = st.number_input("Link width", value=0.08, step=0.01, format="%.4f")
    link_d = st.number_input("Link depth", value=0.04, step=0.01, format="%.4f")

    # Advanced settings (collapsed by default)
    with st.expander("⚙️ Advanced Settings", expanded=False):
        st.caption("Adjust these if solver fails to find solutions")
        
        grid_n = st.slider("Grid search resolution", 5, 25, 13, 2,
                          help="Higher = better chance of finding both forms, but slower")
        tol = st.number_input("Solver tolerance", value=1e-9, format="%.2e",
                             help="Increase (e.g. 1e-6) if no roots found")
        
        st.markdown("**Initial guesses (degrees)**")
        col1, col2 = st.columns(2)
        with col1:
            guessA_t2 = st.number_input("Form I: θ₂", value=0.0, step=10.0)
            guessA_t3 = st.number_input("Form I: θ₃", value=0.0, step=10.0)
        with col2:
            guessB_t2 = st.number_input("Form II: θ₂", value=60.0, step=10.0)
            guessB_t3 = st.number_input("Form II: θ₃", value=-60.0, step=10.0)

    run = st.button("Compute", type="primary")

# Default values for parameters not exposed in simple mode
UNIQUE_EPS = 5e-3
RECOVER = True
RECOVER_RADIUS = 25.0
RECOVER_TRIALS = 10
SV_FD_STEP = 2e-6
MIN_SPAN_DEG = 5.0

# Use advanced settings if provided, otherwise defaults
if 'grid_n' not in dir():
    grid_n = 13
if 'tol' not in dir():
    tol = 1e-9
if 'guessA_t2' not in dir():
    guessA_t2, guessA_t3 = 0.0, 0.0
if 'guessB_t2' not in dir():
    guessB_t2, guessB_t3 = 60.0, -60.0

if "ran" not in st.session_state:
    st.session_state["ran"] = True
    run = True

alpha_deg = [alpha1, alpha2, alpha3]
a3v = [a1, a2, a3_val]
R3v = [R1, R2, R3_val]
geom = {"r_joint": r_joint, "r_offset": r_offset, "link_w": link_w, "link_d": link_d}

if run:
    with st.spinner("Computing..."):
        dfA, dfB, ranges_A, ranges_B, err = compute_branches_svd_and_ranges(
            alpha_deg, a3v, R3v, th1_min, th1_max, n_points,
            grid_n, UNIQUE_EPS, tol, (guessA_t2, guessA_t3), (guessB_t2, guessB_t3),
            RECOVER, RECOVER_RADIUS, RECOVER_TRIALS, SV_FD_STEP, geom, MIN_SPAN_DEG
        )

    if err:
        st.error(err)
        st.stop()

    # Metrics
    A_math = float(dfA["ok"].mean())
    B_math = float(dfB["ok"].mean())
    A_av = float((dfA["ok"] & dfA["phys_ok"]).mean())
    B_av = float((dfB["ok"] & dfB["phys_ok"]).mean())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Form I solved", f"{100*A_math:.1f}%")
    m2.metric("Form I valid", f"{100*A_av:.1f}%")
    m3.metric("Form II solved", f"{100*B_math:.1f}%")
    m4.metric("Form II valid", f"{100*B_av:.1f}%")

    st.subheader("Valid Motion Ranges")
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Form I**")
        if ranges_A:
            st.dataframe(pd.DataFrame(ranges_A, columns=["θ₁ start", "θ₁ end"]), use_container_width=True)
        else:
            st.caption("No valid regions.")
    with cB:
        st.markdown("**Form II**")
        if ranges_B:
            st.dataframe(pd.DataFrame(ranges_B, columns=["θ₁ start", "θ₁ end"]), use_container_width=True)
        else:
            st.caption("No valid regions.")

    # SVD
    sv_cols = [f"sv{k}" for k in range(6)]
    st.subheader("Singular Values — Form I")
    st.line_chart(dfA.set_index("theta1_deg")[sv_cols], height=280)
    st.subheader("Singular Values — Form II")
    st.line_chart(dfB.set_index("theta1_deg")[sv_cols], height=280)

    # Angle curves
    st.subheader("Joint Angles vs θ₁")
    if _HAS_ALTAIR:
        left, right = st.columns(2)
        with left:
            st.altair_chart(_altair_valid_curve(dfA, "theta2_deg", "Form I: θ₂"), use_container_width=True)
            st.altair_chart(_altair_valid_curve(dfA, "theta3_deg", "Form I: θ₃"), use_container_width=True)
        with right:
            st.altair_chart(_altair_valid_curve(dfB, "theta2_deg", "Form II: θ₂"), use_container_width=True)
            st.altair_chart(_altair_valid_curve(dfB, "theta3_deg", "Form II: θ₃"), use_container_width=True)
    else:
        st.line_chart(dfA.set_index("theta1_deg")[["theta2_deg", "theta3_deg"]], height=220)
        st.line_chart(dfB.set_index("theta1_deg")[["theta2_deg", "theta3_deg"]], height=220)

    with st.expander("Raw Data"):
        st.dataframe(dfA, use_container_width=True)
        st.dataframe(dfB, use_container_width=True)

    st.download_button("Download Form I CSV", dfA.to_csv(index=False).encode(), "form_I.csv", "text/csv")
    st.download_button("Download Form II CSV", dfB.to_csv(index=False).encode(), "form_II.csv", "text/csv")