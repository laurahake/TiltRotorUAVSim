import os, sys, random
from pathlib import Path

from tools import policy
sys.path.insert(0, os.fspath(Path(__file__).parents[1]))

import numpy as np
import torch
import math
import time
import csv
from pathlib import Path

from models.vtol_dynamics import VtolDynamics
import parameters.simulation_parameters as SIM
from models.vtol_dynamics_adapter import VtolDynamicsAdapter
from tools.policy import COCPICNNPolicy, PolicyConfig
from message_types.msg_delta import MsgDelta
from message_types.msg_controls import MsgControls
from controllers.tiltrotor_control import TiltRotorSwitcher, TiltConfig
from tools.replay_memory import ReplayBuffer
from dataclasses import dataclass

# ==============================
# Control Limits
# ==============================
u_min = np.array([0.0, 0.0, 0.0, -0.785398, -0.785398], dtype=np.float32)
u_max = np.array([1.0, 1.0, 1.0,  0.785398,  0.785398], dtype=np.float32)
idx_fast = [0,1,2,3,4]     # thr_rear, thr_left, thr_right, elevon_left, elevon_right


# ===============================================================
#  Utility
# ===============================================================
def _ckpt_payload(policy, k, extra=None, rng=None):
    payload = {
        "iter": int(k),
        "policy_state_dict": policy.state_dict(),
        "rng": {
            "python": random.getstate(),
            "torch": torch.random.get_rng_state(),
        },
        "timestamp": time.time(),
    }
    if rng is not None:
        payload["rng"]["numpy_generator_state"] = rng.bit_generator.state
    if torch.cuda.is_available():
        payload["rng"]["torch_cuda"] = torch.cuda.get_rng_state_all()
    if extra:
        payload.update(extra)
    return payload

def save_checkpoint(path: Path, policy, k: int, extra=None, rng=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_ckpt_payload(policy, k, extra, rng=rng), path)

def load_checkpoint(path: Path, policy, rng, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    policy.load_state_dict(ckpt["policy_state_dict"])

    st = ckpt.get("rng", {})

    if "python" in st:
        random.setstate(st["python"])
    if "torch" in st:
        torch.random.set_rng_state(st["torch"])
    if torch.cuda.is_available() and ("torch_cuda" in st):
        torch.cuda.set_rng_state_all(st["torch_cuda"])

    if "numpy_generator_state" in st:
        rng.bit_generator.state = st["numpy_generator_state"]

    start_iter = int(ckpt.get("iter", -1)) + 1
    return start_iter, ckpt

@dataclass
class StateBounds:
    p_max: float = 5.0          # meters (pos error, each axis)
    v_max: float = 5.0          # m/s
    angle_max: float = np.deg2rad(110.0)  # rad
    omega_max: float = 15.0      # rad/s
    q_norm_eps: float = 0.05    # quaternion norm tolerance
    tilt_min: float = 0.0
    tilt_max: float = np.deg2rad(115.0)

def quat_angle_err(q, qref):
    # angle between quaternions
    dot = abs(float(np.dot(q, qref)))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)

def is_valid_state(x_np, xref_np, b: StateBounds):
    p_err = x_np[0:3] - xref_np[0:3]
    if np.any(np.abs(p_err) > b.p_max):
        return False, "pos"

    v_err = x_np[3:6] - xref_np[3:6]
    if np.any(np.abs(v_err) > b.v_max):
        return False, "vel"

    q = x_np[6:10]; qref = xref_np[6:10]
    qn = np.linalg.norm(q)
    if abs(qn - 1.0) > b.q_norm_eps:
        return False, "quat_norm"

    ang = quat_angle_err(q, qref)
    if ang > b.angle_max:
        return False, "angle"

    omega = x_np[10:13]
    if np.any(np.abs(omega) > b.omega_max):
        return False, "omega"

    tilt = x_np[13:15]
    if np.any(tilt < b.tilt_min) or np.any(tilt > b.tilt_max):
        return False, "tilt"

    return True, None

def is_valid_state_with_margin(x, xref, b, margin=0.2):
    b2 = StateBounds(
        p_max=b.p_max - margin,
        v_max=b.v_max - margin,
        angle_max=b.angle_max - np.deg2rad(5),
        omega_max=b.omega_max - margin,
        q_norm_eps=b.q_norm_eps * 0.5,
        tilt_min=b.tilt_min + np.deg2rad(2),
        tilt_max=b.tilt_max - np.deg2rad(2),
    )
    return is_valid_state(x, xref, b2)

def sample_initial_and_reference_states(K, state_dim=15, rng=None, bounds: StateBounds = None):
    rng = np.random.default_rng() if rng is None else rng
    b = bounds if bounds is not None else StateBounds()

    POS, VEL, QUAT, RATE, TILT = slice(0,3), slice(3,6), slice(6,10), slice(10,13), slice(13,15)

    pairs = []
    for _ in range(K):
        # --- reference (nominal hover) ---
        xref = np.zeros(state_dim, dtype=np.float32)
        xref[POS]  = np.array([0.0, 0.0, -1.5], dtype=np.float32)
        xref[VEL]  = 0.0
        xref[QUAT] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        xref[RATE] = 0.0
        xref[TILT] = np.array([1.4544, 1.4544], dtype=np.float32)

        # --- sample within 2*2 area ---
        x0 = np.zeros(state_dim, dtype=np.float32)
        x0[0:2] = rng.uniform(-2, 2, size=2).astype(np.float32)
        x0[2] = 0.0
        x0[VEL] = 0.0
        x0[QUAT] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        x0[RATE] = 0.0
        x0[TILT] = np.array([1.4544, 1.4544], dtype=np.float32)

        pairs.append((x0, xref))
    return pairs

def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    return rng

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


def Q_hat_trajectory(policy, x0_np, xref_np, u0_override_np,
                     Qp, R, T=50, gamma=0.99, dt=SIM.ts_simulation,
                     bounds: StateBounds = None):
    """
    Simulate trajectory under policy starting from x0_np to approximate
    our Action-Value function Q_hat(x0, u0_override_np)=∑ γ^t c(x_t, u_t).
    """
    
    b = bounds if bounds is not None else StateBounds()
    
    # 1) Initialize simulation
    vtol = VtolDynamics(dt)
    vtol.external_set_state(np.array(x0_np, dtype=np.float32, copy=True).reshape(-1, 1))
    
    wind = np.zeros((6, 1), dtype=np.float32)

    total_return = 0.0
    discount = 1.0

    # 2) First action: SPSA-perturbed u0
    u_np = u0_override_np.copy()
    
    for t in range(T):
        # ---- Current state ----
        xk = vtol._state.reshape(-1)
        
        # ---- Servo angles (tilt rotors) ----
        tilt_right = float(xref_np[13])
        tilt_left  = float(xref_np[14])
        
        # ---- Build reference state for this step ----
        xref_step = xref_np.copy()
        xref_step[13] = tilt_right
        xref_step[14] = tilt_left
        
        # early termination check
        valid, _reason = is_valid_state(xk, xref_step, b)
        if not valid:
            break

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
            theta_t = torch.tensor([[tilt_right], [tilt_left]], dtype=torch.float32)  # (2,1)
            # Compute optimal next control
            u_next = policy.forward_eval(
                torch.tensor(xk, dtype=torch.float32),
                torch.tensor(xref_step, dtype=torch.float32),
                theta = theta_t,
                u_nominal=torch.tensor(u_np, dtype=torch.float32)
            ).cpu().numpy()
            
            u_np = u_next

    return float(total_return)


# ---- Training Functions ----
def compute_grad_u_spsa(policy, x0_np, xref_np, u_star_np, 
                        Qp, R, T_SPSA, gamma, c_k, rng=None, KSPSA=1,
                        bounds: StateBounds = None):
    """
    Computes SPSA approximation of ∇_u Q_hat(x0, u_star).
    """

    rng = np.random.default_rng() if rng is None else rng

    grad_u_acc = np.zeros_like(u_star_np, dtype=np.float32)

    for _ in range(KSPSA):
        # 1) Sample Rademacher ±1 vector
        Delta_u = rng.choice([-1.0, 1.0], size=u_star_np.shape).astype(np.float32)

        # 2) Perturbed actions
        u_plus  = np.clip(u_star_np + c_k * Delta_u, u_min, u_max)
        u_minus = np.clip(u_star_np - c_k * Delta_u, u_min, u_max)

        # 3) Evaluate Q_hat(x, u) via Rollouts
        R_plus = Q_hat_trajectory(
            policy, x0_np, xref_np, u_plus,
            Qp, R, T_SPSA, gamma,
            bounds=bounds
        )

        R_minus = Q_hat_trajectory(
            policy, x0_np, xref_np, u_minus,
            Qp, R, T_SPSA, gamma,
            bounds=bounds
        )

        # 4) SPSA gradient estimate
        grad_u_j = ((R_plus - R_minus) / (2.0 * c_k)) * Delta_u
        grad_u_acc += grad_u_j.astype(np.float32)

    grad_u_np = grad_u_acc / float(KSPSA)
    return grad_u_np


def rollout_and_fill_replay(policy, x0_np, xref_np, replay, bounds, device,
                            T_max=400, dt=SIM.ts_simulation, explore_noise=0.0):
    vtol_ep = VtolDynamics(dt)
    vtol_ep.external_set_state(np.array(x0_np, dtype=np.float32, copy=True).reshape(-1, 1))

    tilt_cfg = TiltConfig(
        trigger_height_m=3.0, hysteresis_m=0.2, transition_time_s=2.0,
        vertical_angle=math.pi/2, forward_angle=0.0,
        angle_min=0.0, angle_max=math.radians(115.0)
    )
    tilt_switcher = TiltRotorSwitcher(tilt_cfg)
    
    u_prev = policy.u_nominal_default.detach().cpu().numpy().reshape(-1).astype(np.float32)

    for t in range(T_max):
        xk = vtol_ep._state.reshape(-1).astype(np.float32)

        d_down = xk[2]
        tilt_right, tilt_left = tilt_switcher.step_ned(d_down, dt)

        xref_step = xref_np.copy()
        xref_step[13] = tilt_right
        xref_step[14] = tilt_left

        valid, reason = is_valid_state(xk, xref_step, bounds)
        done = not valid

        theta_np = np.array([tilt_right, tilt_left], dtype=np.float32)
        replay.store(xk, xref_step, theta_np, u_prev, done=done, reason=reason)

        if done:
            break

        # policy action (no grad)
        with torch.no_grad():
            xk_t = torch.tensor(xk, dtype=torch.float32, device=device)
            xref_t = torch.tensor(xref_step, dtype=torch.float32, device=device)
            theta_t = torch.tensor([[tilt_right], [tilt_left]], dtype=torch.float32, device=device)
            u_prev_t = torch.tensor(u_prev, dtype=torch.float32, device=device)
            u_star = policy.forward_eval(xk_t, xref_t, theta=theta_t, u_nominal = u_prev_t).cpu().numpy()
            u_prev = u_star.astype(np.float32)

        if explore_noise > 0.0:
            u_star = np.clip(u_star + explore_noise * np.random.randn(*u_star.shape).astype(np.float32),
                             u_min, u_max)

        ctrl = pack_full_controls(u_star, tilt_right, tilt_left)
        vtol_ep.update(MsgDelta(ctrl), np.zeros((6,1), dtype=np.float32))


def spsa_train(policy, Qp, R, T_SPSA, T_max = 400, iters=20, K = 5, KSPSA= 2, seed=1,
               c=0.01, a=1e-5, alpha=0.501, A=100, gamma=0.99, device="cpu"):
    """
    Train the ICNN policy:
    - exact gradient w.r.t. ψ via CVXPYLayer (autograd)
    - SPSA only in u-space via Q_hat_trajectory
    """
    rng = seed_all(seed)
    
    bounds = StateBounds(p_max=5.0, v_max=7.0, angle_max=np.deg2rad(110), omega_max=15.0)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_path = log_dir / "train_log.csv"

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iter",
            "loss_mean",
            "episode_cost_mean",
            "episode_cost_std",
            "episode_length_mean",
            "fail_count",
            "fail_pos",
            "fail_vel",
            "fail_angle",
            "fail_omega",
            "fail_tilt",
            "fail_quat_norm",
            "a_k",
            "c_k",
            "replay_size",        
            "train_used_mean",
            "param_norm",
            "grad_param_norm",
            "delta_param_norm_approx",
            "delta_param_norm_actual",
            "rel_update_approx",
            "rel_update_actual",
            "param_norm_before",
            "param_norm_after",
        ])

    ckpt_dir = Path("logs") / "checkpoints"
    latest_ckpt = ckpt_dir / "latest.pt"
    best_ckpt = ckpt_dir / "best.pt"
    save_every = 5

    best_loss = float("inf")
    start_k = 0

    if latest_ckpt.exists():
        start_k, ckpt = load_checkpoint(latest_ckpt, policy, rng, device=device)
        best_loss = float(ckpt.get("best_loss", best_loss))
        print(f"[RESUME] loaded {latest_ckpt} -> continuing at iter {start_k}")
        
    fail_count = 0
    fail_reason_counter = {
        "pos": 0,
        "vel": 0,
        "angle": 0,
        "omega": 0,
        "tilt": 0,
        "quat_norm": 0,
    }
    
    replay = ReplayBuffer(max_size=90000)    # 90k samples
    min_replay = 5000                        # min size to start training
    warmup_episodes = 75                     # warmup via while len(replay)<min_replay

    # ---- Phase 1: Warmup (fill replay) ----
    filled = 0
    for (x0_np, xref_np) in sample_initial_and_reference_states(warmup_episodes, rng=rng, bounds=bounds):
        rollout_and_fill_replay(policy, x0_np, xref_np, replay, bounds=bounds, device=device, T_max=T_max)
        filled = len(replay)
        if filled >= min_replay:
            break

    print(f"[WARMUP] replay size = {len(replay)}")
    
    # ---- Phase 2: Main training loop ----
    batch_size = 32
    try:
        for k in range(start_k, iters):
            
            # --- Step sizes ---
            c_k = c / ((k+1)**0.11)
            a_k = a / ((k+1 + A)**alpha)
            
            # --- reset gradients ---
            for p in policy.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            
            losses = []
            episode_costs = []
            episode_lengths = []
            train_used = []

            # ==========================================================
            # Phase 2a) Collect fresh on-policy data into replay (K episodes)
            # ==========================================================
            for x0_np, xref_np in sample_initial_and_reference_states(K, rng=rng, bounds=bounds):
                rollout_and_fill_replay(
                    policy, x0_np, xref_np, replay,
                    bounds=bounds, device=device, T_max=T_max,
                    explore_noise=0.0
                )

            # ==========================================================
            # Phase 2b) Train from replay (state-replay batches)
            # ==========================================================
            if len(replay) < min_replay:
                print(f"[WARN] replay not ready: size={len(replay)} < min_replay={min_replay}")
            else:
                num_replay_batches = 1

                for _ in range(num_replay_batches):
                    xB, xrefB, thetaB, uPrevB, doneB, reasonB = replay.sample(batch_size, rng=rng)

                    episode_loss = 0.0
                    T_k = 0

                    for i in range(batch_size):
                        if bool(doneB[i]):
                            continue

                        xk = xB[i].astype(np.float32)
                        xref_step = xrefB[i].astype(np.float32)
                        theta_np = thetaB[i].astype(np.float32)  # shape (2,)

                        # --- Policy forward (TRAIN graph) ---
                        xk_t = torch.tensor(xk, dtype=torch.float32, device=device)
                        xref_t = torch.tensor(xref_step, dtype=torch.float32, device=device)
                        theta_t = torch.tensor(theta_np.reshape(2, 1), dtype=torch.float32, device=device)  # (2,1)
                        
                        u_prev_np = uPrevB[i].astype(np.float32)
                        u_prev_t = torch.tensor(u_prev_np, dtype=torch.float32, device=device)
                        u_star_t = policy.forward_train(xk_t, xref_t, theta=theta_t, u_nominal=u_prev_t)
                        u_star_np = u_star_t.detach().cpu().numpy()
                            
                        valid, _ = is_valid_state_with_margin(xk, xref_step, bounds, margin=0.2)
                        if not valid:
                            continue

                        # --- SPSA grad_u at replay state ---
                        grad_u_np = compute_grad_u_spsa(
                            policy,
                            xk,
                            xref_step,
                            u_star_np,
                            Qp,
                            R,
                            T_SPSA,
                            gamma,
                            c_k,
                            KSPSA=KSPSA,
                            rng=rng,
                            bounds=bounds,
                        )
                        grad_u_t = torch.tensor(grad_u_np, dtype=u_star_t.dtype, device=u_star_t.device)

                        # --- Chain rule surrogate loss ---
                        episode_loss += torch.dot(u_star_t, grad_u_t)
                        T_k += 1

                    # normalize
                    if T_k > 0:
                        episode_loss /= float(T_k)

                    # backward on this replay batch
                    episode_loss.backward()

                    losses.append(float(episode_loss.item()))
                    train_used.append(int(T_k))
                    
                    # ==========================================================
                    # Evaluation rollouts for episode_cost / episode_length logs
                    # ==========================================================
                    episode_costs_eval = []
                    episode_lengths_eval = []

                K_eval = K     

                for x0_np, xref_np in sample_initial_and_reference_states(K_eval, rng=rng, bounds=bounds):
                    # Rollout without SPSA/backward, just forward policy in eval mode to get episode cost and length statistics for logging.
                    vtol_ep = VtolDynamics(SIM.ts_simulation)
                    vtol_ep.external_set_state(np.array(x0_np, dtype=np.float32, copy=True).reshape(-1, 1))

                    tilt_cfg = TiltConfig(
                        trigger_height_m=3.0, hysteresis_m=0.2, transition_time_s=2.0,
                        vertical_angle=math.pi/2, forward_angle=0.0,
                        angle_min=0.0, angle_max=math.radians(115.0)
                    )
                    tilt_switcher = TiltRotorSwitcher(tilt_cfg)

                    episode_cost = 0.0
                    discount = 1.0
                    T_eval = 0
                    u_prev = policy.u_nominal_default.detach().cpu().numpy().reshape(-1).astype(np.float32)
                    for t in range(T_max):
                        if t == 80 :
                            print("Debug breakpoint: t=85 in eval rollout")
                        xk = vtol_ep._state.reshape(-1).astype(np.float32)

                        d_down = xk[2]
                        tilt_right, tilt_left = tilt_switcher.step_ned(d_down, SIM.ts_simulation)

                        xref_step = xref_np.copy()
                        xref_step[13] = tilt_right
                        xref_step[14] = tilt_left

                        valid, reason = is_valid_state(xk, xref_step, bounds)
                        if not valid:
                            fail_count += 1
                            if reason is not None:
                                fail_reason_counter[reason] += 1
                            break

                        # policy action (NO grad)
                        with torch.no_grad():
                            xk_t = torch.tensor(xk, dtype=torch.float32, device=device)
                            xref_t = torch.tensor(xref_step, dtype=torch.float32, device=device)
                            theta_t = torch.tensor([[tilt_right], [tilt_left]], dtype=torch.float32, device=device)
                            u_prev_t = torch.tensor(u_prev, dtype=torch.float32, device=device)
                            u_star_np = policy.forward_eval(xk_t, xref_t, theta=theta_t, u_nominal=u_prev_t).cpu().numpy()
                            u_prev = u_star_np.reshape(-1).astype(np.float32)

                        # cost
                        epos = xk[:3] - xref_step[:3]
                        stage_cost = float(epos.T @ Qp @ epos + u_star_np @ (R @ u_star_np))
                        episode_cost += discount * stage_cost
                        discount *= gamma

                        # step dynamics
                        ctrl = pack_full_controls(u_star_np, tilt_right, tilt_left)
                        vtol_ep.update(MsgDelta(ctrl), np.zeros((6,1), dtype=np.float32))

                        T_eval += 1

                    episode_costs_eval.append(float(episode_cost))
                    episode_lengths_eval.append(int(T_eval))

                episode_costs = episode_costs_eval
                episode_lengths = episode_lengths_eval    
                        
            # --- Param + Grad diagnostics (before update) ---
            with torch.no_grad():
                params_before = []
                param_norm_before_sq = 0.0
                grad_param_norm_sq = 0.0

                for p in policy.parameters():
                    if not p.requires_grad:
                        continue
                    params_before.append(p.detach().clone())
                    param_norm_before_sq += float(p.data.norm().item() ** 2)
                    if p.grad is not None:
                        grad_param_norm_sq += float(p.grad.data.norm().item() ** 2)

                param_norm_before = float(param_norm_before_sq ** 0.5)
                grad_param_norm = float(grad_param_norm_sq ** 0.5)

                delta_param_norm_approx = float(a_k * grad_param_norm)
                rel_update_approx = float(delta_param_norm_approx / (param_norm_before + 1e-12))
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
            
            # --- Actual delta after projection ---
            with torch.no_grad():
                delta_sq = 0.0
                param_norm_after_sq = 0.0
                i = 0
                for p in policy.parameters():
                    if not p.requires_grad:
                        continue
                    dp = p.detach() - params_before[i]
                    delta_sq += float(dp.norm().item() ** 2)
                    param_norm_after_sq += float(p.data.norm().item() ** 2)
                    i += 1

                delta_param_norm_actual = float(delta_sq ** 0.5)
                param_norm_after = float(param_norm_after_sq ** 0.5)
                rel_update_actual = float(delta_param_norm_actual / (param_norm_before + 1e-12))

            episode_cost_mean = float(np.mean(episode_costs))
            episode_cost_std  = float(np.std(episode_costs))
            episode_length_mean = float(np.mean(episode_lengths))
            mean_loss = float(np.mean(losses)) if len(losses) > 0 else float("inf")
            replay_size = int(len(replay))
            train_used_mean = float(np.mean(train_used)) if len(train_used) > 0 else 0.0


            # --- Logging ---
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    k,
                    float(mean_loss),
                    episode_cost_mean,
                    episode_cost_std,
                    episode_length_mean,
                    fail_count,
                    fail_reason_counter["pos"],
                    fail_reason_counter["vel"],
                    fail_reason_counter["angle"],
                    fail_reason_counter["omega"],
                    fail_reason_counter["tilt"],
                    fail_reason_counter["quat_norm"],
                    float(a_k),
                    float(c_k),
                    int(replay_size),
                    float(train_used_mean),
                    float(param_norm),
                    float(grad_param_norm),
                    float(delta_param_norm_approx),
                    float(delta_param_norm_actual),
                    float(rel_update_approx),
                    float(rel_update_actual),
                    float(param_norm_before),
                    float(param_norm_after),
                ])
                
            print(f"[iter {k:03d}] mean loss = {mean_loss:.6f}")
            print(
                f"[iter {k:03d}] fails={fail_count} | "
                f"pos={fail_reason_counter['pos']} "
                f"vel={fail_reason_counter['vel']} "
                f"angle={fail_reason_counter['angle']} "
                f"omega={fail_reason_counter['omega']} "
                f"tilt={fail_reason_counter['tilt']}"
            )
            # best checkpoint
            if mean_loss < best_loss:
                best_loss = mean_loss
                save_checkpoint(best_ckpt, policy, k, extra={"best_loss": best_loss}, rng=rng)
                
            # latest checkpoint
            if (k % save_every) == 0:
                save_checkpoint(latest_ckpt, policy, k, extra={"best_loss": best_loss}, rng=rng)
                
    except KeyboardInterrupt:
        print("\n[INTERRUPT] saving checkpoint...")
        save_checkpoint(latest_ckpt, policy, k, extra={"best_loss": best_loss}, rng=rng)
        raise
    except Exception as e:
        print(f"\n[ERROR] {e} -> saving checkpoint...")
        save_checkpoint(latest_ckpt, policy, k, extra={"best_loss": best_loss}, rng=rng)
        raise


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
    gamma = 0.99
    # -----------------------------
    # Cost matrices
    # -----------------------------
    Qp = np.eye(3, dtype=np.float32) * 10.0
    r_u = 0.1
    R = r_u * np.eye(nu, dtype=np.float32)

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
        Q=Qp,
        device=device
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
    # SPSA training
    # -----------------------------
    spsa_train(
        policy=policy,
        Qp=Qp,
        R=R,
        T_SPSA=100,
        T_max=400,
        iters=30,
        K=2,
        KSPSA=2,
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
    