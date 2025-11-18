import os, sys, random
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))
import numpy as np

from models.vtol_dynamics import VtolDynamics
import parameters.simulation_parameters as SIM
from models.vtol_dynamics_adapter import VtolDynamicsAdapter
from tools.icnn_policy import ICNNPolicy
from message_types.msg_delta import MsgDelta
from message_types.msg_controls import MsgControls
from controllers.tiltrotor_control import TiltRotorSwitcher, TiltConfig

# ---- Parameters ----
T_steps: int = 200
r_u: float = 1e-2           # r * ||u||^2
q_pos: tuple = (1.0, 1.0, 1.0)  # diag(Q) für Positionsfehler
gamma: float = 0.99         # discount factor

u_min = np.array([0.0, 0.0, 0.0, -0.785398, -0.785398], dtype=np.float32)
u_max = np.array([1.0, 1.0, 1.0,  0.785398,  0.785398], dtype=np.float32)
idx_fast = [0,1,2,3,4]     # thr_rear, thr_left, thr_right, elevon_left, elevon_right

# sample K random pairs of initial states and reference states
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

# ---- Helpers ---- 
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

def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)

def euler_discretize(Ac, Bc, bc, dt):
    n = Ac.shape[0]
    Ad = np.eye(n) + dt*Ac
    Bd = dt*Bc
    cd = dt*bc
    return Ad, Bd, cd
    
def linearize_fd(f, x0, u0, eps=1e-5):
    u0_vector = u0.extract_as_array()
    n = x0.size
    m = u0_vector.size
    f0 = f(x0, u0_vector)
    A = np.zeros((n,n))
    B = np.zeros((n,m))
    for i in range(n):
        dx = np.zeros_like(x0); dx[i] = eps
        A[:, i] = (f(x0+dx, u0_vector) - f(x0-dx, u0_vector)) / (2*eps)
    for j in range(m):
        du = np.zeros_like(u0_vector); du[j] = eps
        B[:, j] = (f(x0, u0_vector+du) - f(x0, u0_vector-du)) / (2*eps)
    b = f0 - A @ x0 - B @ u0_vector
    return A, B, b


def rollout(policy, seed, init_state, ref_state):
    seed_all(seed)

    # --- Reset dynamics and setup ---
    vtol = VtolDynamics(SIM.ts_simulation)
    vtol.external_set_state(init_state)
    adapter = VtolDynamicsAdapter(vtol)
    wind = np.zeros((6, 1), dtype=np.float32)
    u_prev = np.zeros(u_min.shape[0], dtype=np.float32)

    # --- Cost weights (position-only) ---
    Qp = np.diag(q_pos).astype(np.float32)
    r  = float(r_u)
    gamma_val = float(gamma)

    # --- Simulation bookkeeping ---
    Q_sum = 0.0
    sim_time = SIM.start_time
    steps = 0

    # --- Tilt control configuration ---
    tilt_cfg = TiltConfig(trigger_height_m=12.0, transition_time_s=3.0)
    tilt_ctrl = TiltRotorSwitcher(tilt_cfg)

    while steps < T_steps and sim_time < SIM.end_time:
        # 1) Current state
        xk = vtol._state.squeeze().astype(np.float32)
        
        # check for inf in state
        if not np.all(np.isfinite(xk)):
            print(f"[rollout] Non-finite state at step {steps}, aborting. xk =", xk)
            # large penalty
            Q_sum += 1e6
            break
        
        d_down = xk[2]  # NED Down position
        tilt_right, tilt_left = tilt_ctrl.step_ned(d_down, SIM.ts_simulation)

        # 2) Reference (desired position trajectory)
        xref = ref_state.copy()
        xref[13] = tilt_right
        xref[14] = tilt_left

        # 3) Linearize and discretize dynamics
        u_prev_full = pack_full_controls(u_prev, tilt_right, tilt_left)
        Ac, Bc_full, bc = linearize_fd(adapter.f, xk, u_prev_full)
        Ad, Bd_full, cd = euler_discretize(Ac, Bc_full, bc, SIM.ts_simulation)
        Bd = Bd_full[:, idx_fast]
        
        Qp=np.diag(q_pos).astype(np.float32) # shape (3,3)

        # 4) Compute control via convex policy
        u_star = policy(
            xk, xref, Ad, Bd, cd,
            Qp,
            r_u * np.eye(policy.nu, dtype=np.float32),
            gamma_val, u_min, u_max
        )
        u_np = np.asarray(u_star, dtype=np.float32).flatten()

        # Check dimensions
        if u_np.shape[0] != 5:
            print("Bad u_np shape:", u_np.shape, "prob.status:",
                  policy.prob.status, "u_star:", u_star)
        output = pack_full_controls(u_np, tilt_right, tilt_left)
        
        # 5) Apply control and propagate dynamics
        output = pack_full_controls(u_np, tilt_right, tilt_left)
        delta = MsgDelta(old=output)
        vtol.update(delta, wind)

        # 6) Compute stage cost: (p - pref)ᵀ Qp (p - pref) + r‖u‖²
        p = xk[0:3]
        p_ref = xref[0:3]
        epos = p - p_ref
        stage_cost = epos.T @ Qp @ epos + r * float(u_np.T @ u_np)

        Q_sum += (gamma_val ** steps) * stage_cost

        # 7) Update loop variables
        u_prev = u_np
        sim_time += SIM.ts_simulation
        steps += 1

    return Q_sum

# ---- SPSA Training loop ----

def spsa_train(policy, K=8, iters=10, seed=1, alpha=0.501, a = 0.0000001, A= 100, c=0.001, gamma_s=0.11):
    rng=np.random.default_rng(seed)
    
    pairs = sample_initial_and_reference_states(K, rng=rng)
    
    psi = policy.get_params()
    print("psi0 min/max/mean:", psi.min(), psi.max(), psi.mean())
    p= psi.size
    
    history = []
    for k in range(iters):
        # step size and pertubation size sequences
        a_k = a / ((k + 1 + A) ** alpha)
        c_k = c / ((k + 1) ** gamma_s)

        # Rademacher
        Delta = rng.choice([-1.0, 1.0], size=p)
        
        # two perturbations
        psi_plus  = psi + c_k * Delta
        psi_minus = psi - c_k * Delta
    
        # ---- Evaluate J(ψ⁺) ----
        policy.set_params(psi_plus)
        J_plus = 0.0
        for i, (x0, xref) in enumerate(pairs):
            J_plus += rollout(
                policy,
                seed=1000 + 37 * k + i,
                init_state=x0,
                ref_state=xref,
            )
        J_plus /= K

        # ---- Evaluate J(ψ⁻) ----
        policy.set_params(psi_minus)
        J_minus = 0.0
        for i, (x0, xref) in enumerate(pairs):
            J_minus += rollout(
                policy,
                seed=2000 + 41 * k + i,
                init_state=x0,
                ref_state=xref,
            )
        J_minus /= K
    
        # SPSA gradient estimate
        grad = (J_plus - J_minus) / (2.0 * c_k) * (1.0 / Delta)
        
        print("Jp, Jm, diff:", J_plus, J_minus, J_plus - J_minus)
        print("a_k, c_k:", a_k, c_k)
        print("grad max/mean:", np.max(grad), np.mean(grad))
        print("psi before update min/max:", psi.min(), psi.max())
        
        # parameter update + projection
        psi_new = psi - a_k * grad
        
        print("psi after update min/max:", psi_new.min(), psi_new.max())
        print("finite grad?", np.all(np.isfinite(grad)))
        print("finite psi_new?", np.all(np.isfinite(psi_new)))
        
        policy.set_params(psi_new)
        psi = policy.get_params() 
    
        # Logging
        J_mean = 0.5 * (J_plus + J_minus)
        history.append(
            dict(
                k=k,
                Jp=J_plus,
                Jm=J_minus,
                J=J_mean,
                a_k=a_k,
                c_k=c_k,
            )
        )
        print(f"[iter {k:03d}] J ≈ {J_mean:.4f}  (J+={J_plus:.4f}, J-={J_minus:.4f})  "
              f"a_k={a_k:.3e}, c_k={c_k:.3e}")

    # return final params and training log
    return psi, history

if __name__ == "__main__":
    # Create policy
    nx = 15
    nu = 5
    hidden = 64
    policy = ICNNPolicy(nx, nu, hidden)

    # Train with SPSA
    psi_final, log = spsa_train(policy, K=8, iters=20, seed=42)
    
    np.save("icnn_psi_final.npy", psi_final)
    np.save("spsa_training_log.npy", log)
    print("Training finished. Saved final parameters to icnn_psi_final.npy")