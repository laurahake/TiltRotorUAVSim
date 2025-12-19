import numpy as np
from message_types.msg_delta import MsgDelta
from models.vtol_dynamics import VtolDynamics
from models.vtol_dynamics_torch import VtolDynamicsTorch, VtolTorchConfig
import torch
from torch import nn

class VtolDynamicsAdapter:
    """
    Adapter to use VtolDynamics in a format suitable for fast timescale controller
    Use physics to calculate f(x,u)
    """
    def __init__(self, vtol: VtolDynamics, dt: float, device="cpu", dtype=torch.float32):
        self.vtol = vtol

        # Torch dynamics
        torch_cfg = VtolTorchConfig(dt=dt, device=device, dtype=dtype)
        self.vtol_torch = VtolDynamicsTorch(torch_cfg)

    # ---------------- NumPy path ----------------
    def _u_to_delta(self, u):
        # u layout: [T_rear, T_left, T_right, elevon_L, elevon_R, tilt_R, tilt_L]
        d = MsgDelta()
        d.throttle_rear  = float(u[0])
        d.throttle_left  = float(u[1])
        d.throttle_right = float(u[2])
        d.elevator = float(u[3] + u[4])   # elevon sum
        d.aileron  = float(u[3] - u[4])   # elevon diff
        d.rudder   = 0.0
        d.motor_right = float(u[5])       # tilt (rad)
        d.motor_left  = float(u[6])
        return d

    def f_cont_np(self, x, u, wind=None):
        if wind is None:
            wind = np.zeros((6,1))
        self.vtol.external_set_state(x.reshape(-1,1))
        self.vtol._update_velocity_data(wind)
        delta = self._u_to_delta(u)
        fm = self.vtol._forces_moments(delta)
        xdot = self.vtol._derivatives(self.vtol._state, fm, delta)
        return xdot.squeeze()

    # ---------------- Torch path ----------------
    def f_disc_torch(self, x_torch: torch.Tensor, u_torch: torch.Tensor, wind_torch: torch.Tensor | None = None):
        """
        Differentiable discrete step x_next = f_d(x,u) for ICNN policy / autograd.
        x_torch: shape (15,)
        u_torch: shape (7,)
        wind_torch: shape (6,) optional
        """
        return self.vtol_torch.f_disc(x_torch, u_torch, wind=wind_torch)
        
        