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
from tools.policy import ICNNPolicy, PolicyConfig
from tools.iccn_value import ICNNValue
from message_types.msg_delta import MsgDelta
from message_types.msg_controls import MsgControls
from controllers.tiltrotor_control import TiltRotorSwitcher, TiltConfig

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
    base[TILT] = np.array([1.3837, 1.4544]) # based on convergence parameters

    pairs = []
    for _ in range(K):
        # Initial state: random north/east in [0,50], down = 0
        init = base.copy()
        init[POS] = np.array([rng.uniform(0, 50),
                              rng.uniform(0, 50),
                              0.0])

        # Reference state: random north/east in [0,100], down in [-100,-20]
        ref = base.copy()
        ref[POS] = np.array([rng.uniform(0, 100),
                             rng.uniform(0, 100),
                             rng.uniform(-15, -5)])

        pairs.append((init, ref))

    return pairs


# ===============================================================
#  Utility
# ===============================================================
def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


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


def Q_hat_trajectory(policy, x0_np, xref_np, u0_override_np,
                     Qp, R, T=50, gamma=0.99, dt=SIM.ts_simulation):
    """
    Simulate trajectory under policy starting from x0_np to approximate
    our Action-Value function Q_hat(x0, u0_override_np)=∑ γ^t c(x_t, u_t).
    """
    # 1) Initialize simulation
    vtol = VtolDynamics(dt)
    vtol.external_set_state(x0_np.reshape(-1, 1))

    tilt_cfg = TiltConfig(trigger_height_m=2.0,
                              hysteresis_m=0.5,
                              transition_time_s=3.0,
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
    u_plus  = u_star_np + c_k * Delta_u
    u_minus = u_star_np - c_k * Delta_u

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
               c=0.1, a=1e-3, alpha=0.501, A=100, gamma=0.99, device="cpu"):
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

    dynamics_fn = lambda x, u: adapter.f_disc_torch(x, u)

    # -----------------------------
    # ICNN Value Function
    # -----------------------------
    hidden_dims = [64, 64]

    icnn_value = ICNNValue(
        input_dim=2 * nx,      # z = [x_next, x_ref]
        hidden_dims=hidden_dims
    ).to(device)

    # -----------------------------
    # Policy configuration
    # -----------------------------
    config = PolicyConfig(
        nx=nx,
        nu=nu,
        gamma=gamma,
        u_min=u_min,
        u_max=u_max,
        R=R,
        device=device,
        dt=dt,
    )

    # -----------------------------
    # Policy
    # -----------------------------
    policy = ICNNPolicy(
        config=config,
        icnn=icnn_value,
        dynamics_fn=dynamics_fn
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
        iters=5,
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
    torch.save(icnn_value.state_dict(), "icnn_value.pt")
    print("Training finished. ICNN value saved.")


if __name__ == "__main__":
    main()
    