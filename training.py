import os, sys, random
from pathlib import Path
sys.path.insert(0, os.fspath(Path(__file__).parents[1]))

import numpy as np
import torch
import math
import csv
from pathlib import Path

from models.vtol_dynamics import VtolDynamics
import parameters.simulation_parameters as SIM
from models.vtol_dynamics_adapter import VtolDynamicsAdapter
from tools.policy import COCPICNNPolicy, PolicyConfig
from message_types.msg_delta import MsgDelta
from message_types.msg_controls import MsgControls
from controllers.tiltrotor_control import TiltRotorSwitcher, TiltConfig
from scipy.spatial import cKDTree

# ---- Load dataset ----
DATA_DIR = Path(__file__).resolve().parent / "data_rollouts"

state_files = sorted(DATA_DIR.glob("state_db_*.npy"))
index_files = sorted(DATA_DIR.glob("tilt_index_*.npz"))

STATE_DB_PATH = state_files[-1]
INDEX_PATH    = index_files[-1]

X_db = np.load(STATE_DB_PATH)
idx  = np.load(INDEX_PATH)

# --- Build KD-tree for 2D features: [height, tilt] ---
feats_n = idx["feats_n"].astype(np.float32)  # (N,2)
mu = idx["mu"].astype(np.float32)            # (2,)
sig = idx["sig"].astype(np.float32)          # (2,)

tree = cKDTree(feats_n)

print(f"[DATA] Loaded DB {X_db.shape} and KD features {feats_n.shape} from:")
print(f"       {STATE_DB_PATH.name}")
print(f"       {INDEX_PATH.name}")

# ---- Parameters ----
T_steps: int = 200
r_u: float = 1e-2           # r * ||u||^2
q_pos: tuple = (1.0, 1.0, 1.0)  # diag(Q) for position error
gamma: float = 0.99         # discount factor

u_min = np.array([0.0, 0.0, 0.0, -0.785398, -0.785398], dtype=np.float32)
u_max = np.array([1.0, 1.0, 1.0,  0.785398,  0.785398], dtype=np.float32)
idx_fast = [0,1,2,3,4]     # thr_rear, thr_left, thr_right, elevon_left, elevon_right

# ===============================================================
#  State Sampling
# ===============================================================
def sample_initial_and_reference_states(K, state_dim=15, rng=None):
    """
    Returns: list of (init_state, ref_state) tuples, length K.
    State layout (NED): [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r, tilt_r, tilt_l]
    """
    rng = np.random.default_rng() if rng is None else rng

    POS  = slice(0, 3)    # [pn, pe, pd]
    VEL  = slice(3, 6)    # [u, v, w]
    QUAT = slice(6, 10)   # [e0, e1, e2, e3]
    RATE = slice(10, 13)  # [p, q, r]
    TILT = slice(13, 15)  # [right, left]

    # Base level hover template
    base = np.zeros(state_dim)
    base[VEL]  = 0.0
    base[QUAT] = np.array([1., 0., 0., 0.])
    base[RATE] = 0.0
    base[TILT] = np.array([1.4544, 1.4544]) # based on convergence parameters

    pairs = []
    
    tilt_cfg = TiltConfig(trigger_height_m=1.2,
                              hysteresis_m=0.2,
                              transition_time_s=2.0,
                              vertical_angle=math.pi/2,
                              forward_angle=0.0,
                              angle_min=0.0,
                              angle_max=math.radians(115.0))
    
    tilt_switcher = TiltRotorSwitcher(tilt_cfg)
    
    for _ in range(K):
        # Initial state: random north/east in [0,1], down in [-1,0] 
        init = base.copy()
        init[POS] = np.array([rng.uniform(0, 1),
                              rng.uniform(0, 1),
                              rng.uniform(-1, 0)])
        
        # get corresponding tilt angles
        #tilt_angles = tilt_switcher.step_ned(init[2], SIM.ts_simulation)
        #init[TILT] = np.array(tilt_angles)

        # sample other state components using 2D KDTree from dataset
        h_cmd = float(-init[2])              # height = -pd (NED)
        tilt_cmd = float(init[13])           # symm tilt (right==left)

        #template = _pick_template_from_tree(rng, h_cmd, tilt_cmd, k=50)

        # keep sampled position
        #template[POS] = init[POS]

        # enforce tilt from slow controller (symm)
        #template[TILT] = np.array([tilt_cmd, tilt_cmd], dtype=np.float32)

        # small domain randomization
        #template[VEL]  += rng.normal(0.0, 0.2, size=3).astype(np.float32)
        #template[RATE] += rng.normal(0.0, 0.05, size=3).astype(np.float32)

        # normalize quaternion (safety)
        #_normalize_quat_inplace(template)

        #init = template
        
        #if not np.isfinite(init).all():
            #raise RuntimeError(f"Non-finite init state sampled! init={init}")

        #q = init[6:10]
        #qn = np.linalg.norm(q)
        #if not np.isfinite(qn) or qn < 1e-6:
            #raise RuntimeError(f"Bad quaternion sampled! norm={qn}, q={q}")

        # kep velocities/rates in reasonable range (start conservative)
        #init[3:6] = np.clip(init[3:6], -5.0, 5.0)
        #init[10:13] = np.clip(init[10:13], -5.0, 5.0)

        # tilt sanity
        #init[13:15] = np.clip(init[13:15], 0.0, np.deg2rad(115.0))
        
        # Reference state: random north/east in [0,2], down in [-2,0]
        ref = base.copy()
        ref[POS] = np.array([rng.uniform(1, 2),
                             rng.uniform(1, 2),
                             rng.uniform(-2, -1)])

        pairs.append((init, ref))

    return pairs


# ===============================================================
#  Utility
# ===============================================================
def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def euler_discretize(Ac, Bc, bc, dt):
    n = Ac.shape[0]
    Ad = np.eye(n) + dt*Ac
    Bd = dt*Bc
    cd = dt*bc
    return Ad, Bd, cd


def linearize_fd(f, x0, u0, eps=1e-5):
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    u0 = np.asarray(u0, dtype=float).reshape(-1)

    n = x0.size
    m = u0.size
    f0 = f(x0, u0)
    A = np.zeros((n,n))
    B = np.zeros((n,m))
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps
        A[:, i] = (f(x0+dx, u0) - f(x0-dx, u0)) / (2*eps)
    for j in range(m):
        du = np.zeros(m); du[j] = eps
        B[:, j] = (f(x0, u0+du) - f(x0, u0-du)) / (2*eps)
    b = f0 - A @ x0 - B @ u0
    return A, B, b


def pack_full_controls(u_fast_np, tilt_right, tilt_left):
    out = MsgControls()
    out.throttle_rear  = float(u_fast_np[0])
    out.throttle_left  = float(u_fast_np[1])
    out.throttle_right = float(u_fast_np[2])
    out.elevon_left    = float(u_fast_np[3])
    out.elevon_right   = float(u_fast_np[4])
    out.servo_right    = float(tilt_right)  # from slow controller
    out.servo_left     = float(tilt_left)   # from slow controller
    return out

def lin_disc_fast(adapter, xk_np, u_fast0_np, theta_np, dt, eps=1e-5):
    tilt_r, tilt_l = float(theta_np[0]), float(theta_np[1])

    # Wrapper: takes u_fast (5D), packs full (7D), calls f_cont_np
    def f_fast(x, u_fast):
        u_full_msg = pack_full_controls(u_fast, tilt_r, tilt_l)  # MsgControls
        return adapter.f_cont_np(x, u_full_msg)                  # returns dx/dt as (nx,)

    Ac, Bc_fast, bc = linearize_fd(f_fast, xk_np, u_fast0_np, eps=eps)  # B is (nx,5)
    Ad, Bd, cd = euler_discretize(Ac, Bc_fast, bc, dt)

    # make cd (nx,1) for cvxpylayer convenience
    return Ad.astype(np.float32), Bd.astype(np.float32), cd.astype(np.float32).reshape(-1, 1)

def _normalize_quat_inplace(x):
    q = x[6:10]
    n = np.linalg.norm(q)
    if n < 1e-8:
        x[6:10] = np.array([1., 0., 0., 0.], dtype=np.float32)
    else:
        x[6:10] = (q / n).astype(np.float32)


def _pick_template_from_tree(rng, h_cmd, tilt_cmd, k=50):
    """
    Returns a copy of a template state from X_db whose (height, tilt) is close to (h_cmd, tilt_cmd).
    height: h = -pd (NED)
    tilt_cmd: scalar (symm tilt)
    """
    query = np.array([h_cmd, tilt_cmd], dtype=np.float32)
    query_n = (query - mu) / sig

    N = X_db.shape[0]
    k_eff = min(k, N)

    dists, idxs = tree.query(query_n, k=k_eff)
    idxs = np.atleast_1d(idxs)
    dists = np.atleast_1d(dists).astype(np.float32)

    # distance-weighted pick
    w = 1.0 / (dists + 1e-6)
    w = w / w.sum()
    j = int(rng.choice(idxs, p=w))

    return X_db[j].copy().astype(np.float32)


def Q_hat_trajectory(policy, x0_np, xref_np, u0_override_np,
                     Qp, R, T=50, gamma=0.99, dt=SIM.ts_simulation):
    """
    Simulate trajectory under policy starting from x0_np to approximate
    our Action-Value function Q_hat(x0, u0_override_np)=∑ γ^t c(x_t, u_t).
    """
    # 1) Initialize simulation
    vtol = VtolDynamics(dt)
    vtol.external_set_state(x0_np.reshape(-1, 1))

    tilt_cfg = TiltConfig(trigger_height_m=1.2,
                              hysteresis_m=0.2,
                              transition_time_s=2.0,
                              vertical_angle=math.pi/2,
                              forward_angle=0.0,
                              angle_min=0.0,
                              angle_max=math.radians(115.0))
    
    tilt_switcher = TiltRotorSwitcher(tilt_cfg)
    wind = np.zeros((6, 1), dtype=np.float32)

    total_return = 0.0
    discount = 1.0

    # 2) First action: SPSA-perturbed u0
    u_np = u0_override_np.copy()
    
    for t in range(T):
        # ---- Current state ----
        xk = vtol._state.reshape(-1)
        
        # ---- Servo angles (tilt rotors) ----
        d_down = xk[2]
        tilt_right, tilt_left = tilt_switcher.step_ned(d_down, dt)
        
        # ---- Build reference state for this step ----
        xref_step = xref_np.copy()
        xref_step[13] = tilt_right
        xref_step[14] = tilt_left

        # ---- Stage cost ----
        epos = xk[:3] - xref_step[:3]
        stage_cost = float(epos.T @ Qp @ epos + u_np @ (R @ u_np))
        total_return += discount * stage_cost
        discount *= gamma
        
        # ---- Build control message ----
        ctrl = pack_full_controls(u_np, tilt_right, tilt_left)
        delta = MsgDelta(ctrl)
        wind = np.zeros((6, 1))

        # ---- Simulate real nonlinear dynamics ----
        vtol.update(delta, wind)

        # ---- Next action via optimal controller ----
        if t < T - 1:           # if not last time step

            # Compute optimal next control
            u_np = policy.forward_eval(
                torch.tensor(xk, dtype=torch.float32),
                torch.tensor(xref_step, dtype=torch.float32)
            ).cpu().numpy()

    return float(total_return)


# ---- Training Functions ----
def compute_grad_u_spsa(policy, x0_np, xref_np, u_star_np, 
                        Qp, R, T_steps, gamma, c_k):
    """
    Computes SPSA approximation of ∇_u Q_hat(x0, u_star).
    """

    # 1. Sample Rademacher ±1 vector
    Delta_u = np.random.choice([-1.0, 1.0], size=u_star_np.shape)

    # 2) Perturbed actions
    u_plus  = np.clip(u_star_np + c_k * Delta_u, u_min, u_max)
    u_minus = np.clip(u_star_np - c_k * Delta_u, u_min, u_max)

    # 3) Evaluate Q_hat(x, u) via Rollouts
    R_plus = Q_hat_trajectory(policy, x0_np, xref_np,
                              u_plus, Qp, R, T_steps, gamma)

    R_minus = Q_hat_trajectory(policy, x0_np, xref_np,
                               u_minus, Qp, R, T_steps, gamma)

    # 4) SPSA gradient estimate
    grad_u_np = ((R_plus - R_minus) / (2 * c_k)) * Delta_u

    # convert to torch
    return grad_u_np

def spsa_train(policy, Qp, R, T_steps, iters=20, K = 8, seed=1,
               c=0.01, a=1e-5, alpha=0.501, A=100, gamma=0.99, device="cpu"):
    """
    Train the ICNN policy:
    - exact gradient w.r.t. ψ via CVXPYLayer (autograd)
    - SPSA only in u-space via Q_hat_trajectory
    """
    seed_all(seed)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_path = log_dir / "train_log.csv"

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iter",
            "loss_mean",
            "Q_hat_mean",
            "grad_u_norm",
            "grad_u_max",
            "a_k",
            "c_k",
            "u_mean",
            "param_norm",
        ])

    for k in range(iters):
        losses = []
        Q_hats = []
        grad_u_norms = []
        grad_u_maxs = []
        u_means = []
        
        # --- Step sizes ---
        c_k = c / ((k+1)**0.11)
        a_k = a / ((k+1 + A)**alpha)
        
        # --- reset gradients ---
        for p in policy.parameters():
            if p.grad is not None:
                p.grad.zero_()
        
        losses = []

        # --- Monte Carlo loop ---
        for x0_np, xref_np in sample_initial_and_reference_states(K):

            # Torch-Inputs
            x0_t = torch.tensor(x0_np, dtype=torch.float32, device=device)
            xref_t = torch.tensor(xref_np, dtype=torch.float32, device=device)
            
            # policy output
            u_star_t = policy.forward_train(x0_t, xref_t)
            u_means.append(u_star_t.detach().abs().mean().item())
            
            # Estimate ∇_u Q̂(x0, u*) via SPSA using non-differentiable rollouts
            # detach u* from autograd and treat the resulting gradient as constant.
            u_star_np = u_star_t.detach().cpu().numpy()
            grad_u_np = compute_grad_u_spsa(
                policy,
                x0_np,
                xref_np,
                u_star_np,
                Qp,
                R,
                T_steps,
                gamma,
                c_k
            )
            # ---- SPSA diagnostics ----
            grad_u_norms.append(np.linalg.norm(grad_u_np))
            grad_u_maxs.append(np.max(np.abs(grad_u_np)))
            
            Q_hat = Q_hat_trajectory(
                policy,
                x0_np,
                xref_np,
                u_star_np,
                Qp,
                R,
                T_steps,
                gamma
            )
            Q_hats.append(Q_hat)
            
            grad_u_t = torch.tensor(
                grad_u_np,
                dtype=u_star_t.dtype,
                device=u_star_t.device
            )
            
            # Chain rule: loss = u* dot grad_u
            loss = torch.dot(u_star_t, grad_u_t)
            
            # backward
            loss.backward()
            losses.append(loss.item())
        
        
        param_norm = 0.0
        for p in policy.parameters():
            param_norm += p.data.norm().item() ** 2
        param_norm = param_norm ** 0.5


        # --- Parameter update ---
        with torch.no_grad():
            for p in policy.parameters():
                if p.grad is not None:
                    p -= a_k * p.grad
        policy.project_icnn_parameters()
        
        # --- Logging ---
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                k,
                float(np.mean(losses)),
                float(np.mean(Q_hats)),
                float(np.mean(grad_u_norms)),
                float(np.mean(grad_u_maxs)),
                float(a_k),
                float(c_k),
                float(np.mean(u_means)),
                float(param_norm),
            ])
            
        print(
            f"[iter {k:03d}] "
            f"mean loss = {sum(losses) / len(losses):.6f}"
        )


def main():
    # -----------------------------
    # General settings
    # -----------------------------
    device = "cpu"
    torch.set_default_dtype(torch.float32)

    # -----------------------------
    # Dimensions
    # -----------------------------
    nx = 15          # VTOL state dimension
    nu = 5           # control input dimension

    # -----------------------------
    # Simulation / discretization
    # -----------------------------
    dt = 0.01
    T_steps = 30
    gamma = 0.99
    # -----------------------------
    # Cost matrices
    # -----------------------------
    Qp = np.eye(3, dtype=np.float32) * 10.0
    r_u = 0.1
    R = r_u * np.eye(nu, dtype=np.float32)

    # -----------------------------
    # Control constraints
    # -----------------------------
    u_min = np.array([0.0, 0.0, 0.0, -0.785398, -0.785398], dtype=np.float32)
    u_max = np.array([1.0, 1.0, 1.0,  0.785398,  0.785398], dtype=np.float32)

    # -----------------------------
    # VTOL dynamics (NumPy + Torch)
    # -----------------------------
    vtol = VtolDynamics(ts=dt)
    adapter = VtolDynamicsAdapter(
        vtol,
        dt=dt,
        device=device
    )

    # -----------------------------
    # Policy configuration
    # -----------------------------
    config = PolicyConfig(
        nx=nx,
        nu=nu,
        ntheta=2,
        gamma=gamma,
        u_min=u_min,
        u_max=u_max,
        R=R,
        device=device,
        dt=dt,
    )

    def dyn_lin_disc(x_torch, u0_torch, theta_torch):
        x_np = x_torch.detach().cpu().numpy()
        u0_np = u0_torch.detach().cpu().numpy()
        theta_np = theta_torch.detach().cpu().numpy()
        return lin_disc_fast(adapter, x_np, u0_np, theta_np, dt)
    
    
    # -----------------------------
    # Policy
    # -----------------------------
    policy = COCPICNNPolicy(
        config=config,
        dynamics_linearize_discretize_fn=dyn_lin_disc
    ).to(device)
    

    # -----------------------------
    # Sanity check (policy forward)
    # -----------------------------
    x0_np, xref_np = sample_initial_and_reference_states(1)[0]

    x0_t = torch.tensor(x0_np, dtype=torch.float32)
    xref_t = torch.tensor(xref_np, dtype=torch.float32)

    u_test = policy.forward_eval(x0_t, xref_t)

    print("Sanity check u*:", u_test)
    print("Shape:", u_test.shape)

    # -----------------------------
    # SPSA training (manual update)
    # -----------------------------
    spsa_train(
        policy=policy,
        Qp=Qp,
        R=R,
        T_steps=T_steps,
        iters=500,
        K=8,
        seed=1,
        c=0.1,
        a=1e-3,
        alpha=0.501,
        A=100,
        gamma=0.99,
        device=device,
    )

    # -----------------------------
    # Save trained value function
    # -----------------------------
    torch.save(policy.state_dict(), "cocp_icnn_policy.pt")
    print("Training finished. ICNN value saved.")


if __name__ == "__main__":
    main()
    