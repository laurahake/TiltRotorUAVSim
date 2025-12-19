from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import cvxpy as cp
import math
from cvxpylayers.torch import CvxpyLayer
from controllers.tiltrotor_control import TiltRotorSwitcher, TiltConfig


Tensor = torch.Tensor


@dataclass
class PolicyConfig:
    nx: int                     
    nu: int                    
    gamma: float                # discounting factor
    u_min: np.ndarray           # control constraints lower bound
    u_max: np.ndarray           # control constraints upper bound
    R: np.ndarray               # quadratic cost matrix, shape (nu, nu)
    dt: float                   # time step (for tilt switcher)
    device: str = "cpu"         #
    dtype: torch.dtype = torch.float32


class ICNNPolicy(nn.Module):
    """
    ICNN-based policy with DPP-konform CVXPYLayer-QP-Layer.

    structure:

        1) dynamics:   x_{k+1} = f(x_k, u_k; θ)
        2) Value:     V(x', x_ref; ψ)  approximated by ICNN
        3) Gradient:  ∇_{x'} V  via Autograd
        4) Bu = ∂f/∂u (x_k, u_k) via Autograd (Jacobian)
        5) g_k = γ * Bu^T ∇_{x'} V   
        6) QP: u_k* = argmin_u ( u^T R u + g_k^T u )  s.t. u_min ≤ u ≤ u_max

    Only R and g_k are Parameters in the QP layer.
    This makes the problem DPP-compliant.
    """

    def __init__(
        self,
        config: PolicyConfig,
        icnn: nn.Module,
        dynamics_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> None:
        """
        Parameters
        ----------
        config : PolicyConfig
            Configuration object with dimensions, cost matrix, bounds, etc.
        icnn : nn.Module
            ICNN that approximates the Value Function V(x', x_ref).
            Expects tensor of shape [x_next, x_ref] as input.
        dynamics_fn : Callable
            Torch-Function f(x, u) -> x_next.
            Must be fully differentiable in x and u.
        """
        super().__init__()

        self.cfg = config
        self.icnn = icnn
        self.dynamics_fn = dynamics_fn

        self.nx = config.nx
        self.nu = config.nu
        self.gamma = config.gamma
        self.R = config.R
        self.device = torch.device(config.device)
        self.dtype = config.dtype
        
        tilt_cfg = TiltConfig(trigger_height_m=2.0,
                              hysteresis_m=0.5,
                              transition_time_s=3.0,
                              vertical_angle=math.pi/2,
                              forward_angle=0.0,
                              angle_min=0.0,
                              angle_max=math.radians(115.0))
                              
        self.tilt_switcher = TiltRotorSwitcher(tilt_cfg)
        self.dt = config.dt

        # Cost matrix R as Torch buffer (constant during training)
        R_torch = torch.as_tensor(config.R, dtype=self.dtype, device=self.device)
        assert R_torch.shape == (self.nu, self.nu)
        self.register_buffer("R_torch", R_torch)

        # Input constraints as Torch tensors
        self.register_buffer(
            "u_min_torch",
            torch.as_tensor(config.u_min, dtype=self.dtype, device=self.device),
        )
        self.register_buffer(
            "u_max_torch",
            torch.as_tensor(config.u_max, dtype=self.dtype, device=self.device),
        )

        # Default nominal input (e.g., zero vector)
        self.register_buffer(
            "u_nominal_default",
            torch.zeros(self.nu, dtype=self.dtype, device=self.device),
        )

        # Define the QP layer (DPP-compliant!)
        self._build_qp_layer()

    # ------------------------------------------------------------------
    # 1. QP-Layer
    # ------------------------------------------------------------------
    def _build_qp_layer(self):
        nu = self.nu

        # Optimization variable
        u = cp.Variable(nu)

        # Parameters (MUST be Parameters for DPP)
        g = cp.Parameter(nu)
        u_min = cp.Parameter(nu)
        u_max = cp.Parameter(nu)

        R_const = cp.Constant(self.R)
        
        # Objective
        objective = cp.Minimize(0.5 * cp.quad_form(u, R_const) + g @ u)

        # Box constraints (DPP-safe)
        constraints = [
            u - u_min >= 0,
            u_max - u >= 0
        ]

        problem = cp.Problem(objective, constraints)
        
        # ---------------- DEBUG PRINTS ----------------
        print("\n=== QP DEBUG ===")
        print("is_dcp:", problem.is_dcp())
        print("is_dpp:", problem.is_dpp())
        print("is_qp :", problem.is_qp())
        print("num params:", len(problem.parameters()))
        print("params:", [(p.name(), p.shape) for p in problem.parameters()])
        print("num vars:", len(problem.variables()))
        print("vars  :", [(v.name(), v.shape) for v in problem.variables()])
        print("objective:", problem.objective)
        print("constraints:")
        for c in problem.constraints:
            print("  ", c)

        # Optional: isolate which expression breaks DPP
        try:
            print("objective expr DCP:", problem.objective.expr.is_dcp())
            print("objective expr DPP:", problem.objective.expr.is_dpp())
        except Exception as e:
            print("Could not query expr.is_dpp():", e)

        # Build CVXPYLayer
        self.qp_layer = CvxpyLayer(
            problem,
            parameters=[g, u_min, u_max],
            variables=[u],
        )

    # ------------------------------------------------------------------
    # 2. Helpers: Dynamics, Bu, Grad V, g_k
    # ------------------------------------------------------------------
    def dynamics(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Wrapper um dynamics_fn. Stellt sicher, dass alles im richtigen dtype/device ist.
        """
        return self.dynamics_fn(x, u)

    def compute_Bu_fast(
        self,
        x: Tensor,
        u_fast: Tensor,
        tilt_r: Tensor,
        tilt_l: Tensor,
    ) -> Tensor:
        """
        Compute ∂f/∂u_fast while treating tilt angles as constants.
        """

        x = x.detach()
        u_fast = u_fast.detach().requires_grad_(True)

        def f_fast(u_fast_var):
            u_full = torch.cat(
                [u_fast_var, tilt_r.view(1), tilt_l.view(1)],
                dim=0
            )
            return self.dynamics(x, u_full)

        Bu_fast = torch.autograd.functional.jacobian(
            f_fast,
            u_fast,
            create_graph=True
        )  # shape (nx, nu)

        return Bu_fast

    def compute_grad_V(self, x_next: Tensor, x_ref: Tensor) -> Tensor:
        """
        Computes ∇_{x'} V(x_next, x_ref; ψ).

        x_next : shape (nx,), must be part of autograd graph
        x_ref  : shape (nx,), treated as constant
        """

        # Ensure x_next participates in autograd
        if not x_next.requires_grad:
            x_next.requires_grad_(True)

        # x_ref is constant
        x_ref = x_ref.detach()

        icnn_input = torch.cat([x_next, x_ref], dim=-1)
        V = self.icnn(icnn_input)  # scalar

        (grad_V_x,) = torch.autograd.grad(
            V,
            x_next,
            create_graph=True,
            retain_graph=True,
        )

        return grad_V_x

    def compute_g(
        self,
        x: Tensor,
        u_fast_nominal: Tensor,
        x_ref: Tensor,
        tilt_r: Tensor,
        tilt_l: Tensor,
    ) -> Tensor:
        """
        Compute the linear term g_k = γ * B_u^T ∇_{x'} V
        for the fast-time-scale inputs only.

        x               : shape (nx,)
        u_fast_nominal  : shape (nu,)   (nu = 5)
        x_ref           : shape (nx,)
        tilt_r, tilt_l  : scalar tensors (exogenous, no gradients)

        g_k             : shape (nu,)
        """
        
        # 0) Build full input vector u = [u_fast, tilt_r, tilt_l]
        u_full = torch.cat([u_fast_nominal, tilt_r.view(1), tilt_l.view(1)], dim=0)
        # 1) get next state x'
        x_next = self.dynamics(x, u_full)
        
        # 2) Grad V w.r.t. x'
        grad_V_x = self.compute_grad_V(x_next, x_ref)  # (nx,)

        # 3) Bu = df/du w.r.t. fast inputs only
        Bu = self.compute_Bu_fast(x, u_fast=u_fast_nominal, tilt_r=tilt_r, tilt_l=tilt_l)            # (nx, nu)

        # 4) g_k = γ * Bu^T grad_V_x
        g_k = self.gamma * (Bu.transpose(0, 1) @ grad_V_x)  # (nu,)

        return g_k

    # ------------------------------------------------------------------
    # 3. Policy-Forward: x, x_ref -> u*
    # ------------------------------------------------------------------
    def forward_train(self, x: Tensor, x_ref: Tensor,
                u_nominal: Optional[Tensor] = None) -> Tensor:
        """
        Differentiable policy evaluation.
        Used ONLY during training.
        """

        x = x.to(self.device, self.dtype)
        x_ref = x_ref.to(self.device, self.dtype)

        if u_nominal is None:
            u_nominal = self.u_nominal_default
        else:
            u_nominal = u_nominal.to(self.device, self.dtype)

        # slow tilt dynamics
        d_down = float(x[2].detach().cpu().item())
        tilt_r, tilt_l = self.tilt_switcher.step_ned(d_down, self.dt)

        tilt_r_t = torch.tensor(tilt_r, device=self.device, dtype=self.dtype)
        tilt_l_t = torch.tensor(tilt_l, device=self.device, dtype=self.dtype)

        # ---- gradient-critical part ----
        g_k = self.compute_g(
            x=x,
            u_fast_nominal=u_nominal,
            x_ref=x_ref,
            tilt_r=tilt_r_t,
            tilt_l=tilt_l_t,
        )

        # ---- QP solve ----
        g_in = g_k.unsqueeze(0)
        u_fast_star_batch, = self.qp_layer(
            g_in,
            self.u_min_torch.unsqueeze(0),
            self.u_max_torch.unsqueeze(0),
        )

        return u_fast_star_batch.squeeze(0)
    
    def forward_eval(self, x: Tensor, x_ref: Tensor,
                 u_nominal: Optional[Tensor] = None) -> Tensor:
        """
        Non-differentiable policy evaluation.
        Used for rollouts, SPSA, logging.
        """

        x = x.to(self.device, self.dtype)
        x_ref = x_ref.to(self.device, self.dtype)

        if u_nominal is None:
            u_nominal = self.u_nominal_default
        else:
            u_nominal = u_nominal.to(self.device, self.dtype)

        d_down = float(x[2].item())
        tilt_r, tilt_l = self.tilt_switcher.step_ned(d_down, self.dt)

        tilt_r_t = torch.tensor(tilt_r, device=self.device, dtype=self.dtype)
        tilt_l_t = torch.tensor(tilt_l, device=self.device, dtype=self.dtype)
        
        g_k = self.compute_g(
            x=x,
            u_fast_nominal=u_nominal,
            x_ref=x_ref,
            tilt_r=tilt_r_t,
            tilt_l=tilt_l_t,
        )

        g_k = g_k.detach()
        
        g_in = g_k.unsqueeze(0)
        u_fast_star_batch, = self.qp_layer(
            g_in,
            self.u_min_torch.unsqueeze(0),
            self.u_max_torch.unsqueeze(0),
        )

        return u_fast_star_batch.squeeze(0)

    
    def act_np(self, x_np, xref_np):
        with torch.no_grad():
            x_t = torch.tensor(x_np, dtype=torch.float32)
            xref_t = torch.tensor(xref_np, dtype=torch.float32)
            u_t = self(x_t, xref_t)
        return u_t.cpu().numpy()
