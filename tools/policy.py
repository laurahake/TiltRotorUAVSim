from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import cvxpy as cp
import math
from cvxpylayers.torch import CvxpyLayer

from tools.iccn_value import CVXICNN2Layer, CVXICNN2LayerParams

Tensor = torch.Tensor


@dataclass
class PolicyConfig:
    nx: int
    nu: int
    ntheta: int                 # dimension of theta input to the ICNN
    gamma: float                # discounting factor
    u_min: np.ndarray
    u_max: np.ndarray
    R: np.ndarray               # (nu,nu), PD
    Q: np.ndarray               # (nx,nx), PD, position error weight

    # ICNN architecture (2-layer)
    h1: int = 64
    h2: int = 64
    relu_penalty: float = 1.0
    hard_relu: bool = True

    # training / numerics
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


class COCPICNNPolicy(nn.Module):
    """
    One-step lookahead convex optimal control policy (COCP) with ICNN value inside CVXPY.

    Solves each step:
        u* = argmin_u  u^T R u + gamma * V_psi( x_next(u), x_ref, theta )
        s.t. u_min <= u <= u_max

    where x_next(u) = Ad x + Bd u + cd is an affine surrogate (from linearization + discretization).

    Key properties:
    - DPP-friendly: ICNN is built with auxiliary variables (per ReLU layer) in CVXPY.
    """

    def __init__(
        self,
        config: PolicyConfig,
        dynamics_linearize_discretize_fn: Callable[[Tensor, Tensor, Tensor], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> None:
        """
        Parameters
        ----------
        config : PolicyConfig
        dynamics_linearize_discretize_fn :
            Callable (x, u0, theta) -> (Ad, Bd, cd), all numpy arrays.
            Must return:
                Ad: (nx,nx)
                Bd: (nx,nu)
                cd: (nx,1) or (nx,)
        """
        super().__init__()

        self.cfg = config
        self.nx = config.nx
        self.nu = config.nu
        self.ntheta = config.ntheta
        self.gamma = float(config.gamma)

        self.device = torch.device(config.device)
        self.dtype = config.dtype

        self.dyn_lin_disc = dynamics_linearize_discretize_fn


        # Buffers: bounds + R
        self.register_buffer("u_min_torch", torch.as_tensor(config.u_min, dtype=self.dtype, device=self.device).view(self.nu, 1))
        self.register_buffer("u_max_torch", torch.as_tensor(config.u_max, dtype=self.dtype, device=self.device).view(self.nu, 1))

        R_t = torch.as_tensor(config.R, dtype=self.dtype, device=self.device)
        assert R_t.shape == (self.nu, self.nu)
        self.register_buffer("R_torch", R_t)

        # Default linearization input u0
        self.register_buffer("u_nominal_default", torch.zeros(self.nu, dtype=self.dtype, device=self.device).view(self.nu, 1))

        # ----- Trainable ICNN parameters (psi) as torch.nn.Parameter -----
        # These are passed into cvxpylayers as Parameters -> gradients flow into them.
        input_dim = 2 * self.nx + self.ntheta
        self.h1 = int(config.h1)
        self.h2 = int(config.h2)

        # Layer 1: y1 = W1 s + b1
        self.W1 = nn.Parameter(0.01 * torch.randn(self.h1, input_dim, device=self.device, dtype=self.dtype))
        self.b1 = nn.Parameter(torch.zeros(self.h1, 1, device=self.device, dtype=self.dtype))

        # Layer 2: y2 = W2z z1 + W2x s + b2
        # ICNN-style: W2z >= 0 (enforced by projection)
        self.W2z = nn.Parameter(0.01 * torch.randn(self.h2, self.h1, device=self.device, dtype=self.dtype))
        self.W2x = nn.Parameter(0.01 * torch.randn(self.h2, input_dim, device=self.device, dtype=self.dtype))
        self.b2  = nn.Parameter(torch.zeros(self.h2, 1, device=self.device, dtype=self.dtype))

        # Output: V = Woutz z2 + Woutx s + bout
        # ICNN-style: Woutz >= 0 (enforced by projection)
        self.Woutz = nn.Parameter(0.01 * torch.randn(1, self.h2, device=self.device, dtype=self.dtype))
        self.Woutx = nn.Parameter(0.01 * torch.randn(1, input_dim, device=self.device, dtype=self.dtype))
        self.bout  = nn.Parameter(torch.zeros(1, 1, device=self.device, dtype=self.dtype))

        # Build the CVXPYLayer (one-step COCP)
        self._build_cocp_layer()

        # Initial projection to satisfy ICNN nonneg constraints
        self.project_icnn_parameters()

    # ------------------------------------------------------------------
    # DPP-safe ICNN weight projection
    # ------------------------------------------------------------------
    @torch.no_grad()
    def project_icnn_parameters(self) -> None:
        """
        Enforce nonnegativity on the 'z-path' weights to preserve ICNN-style convexity.
        """
        self.W2z.clamp_(min=0.0)
        self.Woutz.clamp_(min=0.0)

    # ------------------------------------------------------------------
    # CVXPY: One-step COCP layer
    # ------------------------------------------------------------------
    def _build_cocp_layer(self) -> None:
        nx, nu, ntheta = self.nx, self.nu, self.ntheta

        # Decision variable u (column vector)
        u = cp.Variable((nu, 1), name="u")

        # Parameters
        x    = cp.Parameter((nx, 1), name="x")
        xref = cp.Parameter((nx, 1), name="xref")
        theta = cp.Parameter((ntheta, 1), name="theta")

        umin = cp.Parameter((nu, 1), name="u_min")
        umax = cp.Parameter((nu, 1), name="u_max")

        Ad = cp.Parameter((nx, nx), name="Ad")
        Bd = cp.Parameter((nx, nu), name="Bd")
        cd = cp.Parameter((nx, 1), name="cd")
        
        # ---- LIFT parameters into variables ----
        xv     = cp.Variable((nx, 1), name="xv")
        xrefv  = cp.Variable((nx, 1), name="xrefv")
        thetav = cp.Variable((ntheta, 1), name="thetav")

        x_next = cp.Variable((nx, 1), name="x_next")

        s = cp.Variable((2*nx + ntheta, 1), name="s")
        
        lift_constraints = [
            xv == x,
            xrefv == xref,
            thetav == theta,
            x_next == Ad @ xv + Bd @ u + cd,
            s == cp.vstack([x_next, xrefv, thetav]),
        ]


        # Build ICNN block
        icnn = CVXICNN2Layer(
            input_dim=2 * nx + ntheta,
            h1=self.h1,
            h2=self.h2,
            relu_penalty=float(self.cfg.relu_penalty),
            use_hard_relu_constraints=bool(self.cfg.hard_relu),
            name="icnn",
        )
        cvx_params = icnn.make_parameters()
        V, cons_icnn, pen_icnn, _ = icnn.build(s, params=cvx_params)
        
        # position stage cost on NEXT state (Option A) ---
        p_next = x_next[0:3]
        p_ref  = xrefv[0:3]
        epos   = p_next - p_ref
        Qp_pos_const = cp.Constant(self.cfg.Q)

        # Objective: u^T R u + gamma * V + penalty
        R_const = cp.Constant(self.cfg.R)
        objective = cp.Minimize(cp.quad_form(epos, Qp_pos_const) + cp.quad_form(u, R_const) + self.gamma * V + pen_icnn)

        # Box constraints
        constraints = [
           u >= umin,
           u <= umax,
           *lift_constraints,
           *cons_icnn,
        ]

        problem = cp.Problem(objective, constraints)

        # Debug
        print("\n=== COCP DEBUG ===")
        print("is_dcp:", problem.is_dcp())
        print("is_dpp:", problem.is_dpp())
        print("num params:", len(problem.parameters()))
        print("num vars:", len(problem.variables()))

        assert problem.is_dpp(), "COCP problem is not DPP – ICNN formulation must be refactored."

        # Build CvxpyLayer: parameters in the exact order we will pass torch tensors
        self.cocp_layer = CvxpyLayer(
            problem,
            parameters=[
                x, xref, theta, umin, umax, Ad, Bd, cd,
                cvx_params.W1, cvx_params.b1,
                cvx_params.W2z, cvx_params.W2x, cvx_params.b2,
                cvx_params.Woutz, cvx_params.Woutx, cvx_params.bout,
            ],
            variables=[u],
        )

    # ------------------------------------------------------------------
    # Utilities: shape helpers
    # ------------------------------------------------------------------
    def _as_col(self, t: Tensor, dim: int) -> Tensor:
        t = t.to(self.device, self.dtype)
        if t.numel() != dim:
            raise ValueError(f"Expected tensor with {dim} elements, got {t.numel()}.")
        return t.view(dim, 1)

    def _batch(self, t: Tensor) -> Tensor:
        # cvxpylayers expects leading batch dimension
        return t.unsqueeze(0)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------
    def forward_train(self, x: Tensor, x_ref: Tensor, theta, u_nominal: Optional[Tensor] = None) -> Tensor:
        """
        Differentiable policy evaluation (used during training).
        Returns u* (nu,) torch tensor.
        """
        x = self._as_col(x, self.nx)
        x_ref = self._as_col(x_ref, self.nx)

        if u_nominal is None:
            u0 = self.u_nominal_default
        else:
            u0 = self._as_col(u_nominal, self.nu)

        # Linearize + discretize externally -> numpy arrays
        Ad_np, Bd_np, cd_np = self.dyn_lin_disc(x.view(-1), u0.view(-1), theta.view(-1))
        Ad_t = torch.as_tensor(Ad_np, dtype=self.dtype, device=self.device).view(self.nx, self.nx)
        Bd_t = torch.as_tensor(Bd_np, dtype=self.dtype, device=self.device).view(self.nx, self.nu)
        cd_t = torch.as_tensor(cd_np, dtype=self.dtype, device=self.device).view(self.nx, 1)

        # Solve COCP
        (u_star_batch,) = self.cocp_layer(
            self._batch(x),
            self._batch(x_ref),
            self._batch(theta),
            self._batch(self.u_min_torch),
            self._batch(self.u_max_torch),
            self._batch(Ad_t),
            self._batch(Bd_t),
            self._batch(cd_t),
            self._batch(self.W1),
            self._batch(self.b1),
            self._batch(self.W2z),
            self._batch(self.W2x),
            self._batch(self.b2),
            self._batch(self.Woutz),
            self._batch(self.Woutx),
            self._batch(self.bout),
        )

        # u_star_batch shape: (1, nu, 1) -> (nu,)
        return u_star_batch.squeeze(0).squeeze(-1)

    @torch.no_grad()
    def forward_eval(self, x: Tensor, x_ref: Tensor, theta, u_nominal: Optional[Tensor] = None) -> Tensor:
        """
        Non-differentiable policy evaluation (rollouts, SPSA, logging).
        """
        x = self._as_col(x, self.nx)
        x_ref = self._as_col(x_ref, self.nx)

        if u_nominal is None:
            u0 = self.u_nominal_default
        else:
            u0 = self._as_col(u_nominal, self.nu)

        Ad_np, Bd_np, cd_np = self.dyn_lin_disc(x.view(-1), u0.view(-1), theta.view(-1))
        Ad_t = torch.as_tensor(Ad_np, dtype=self.dtype, device=self.device).view(self.nx, self.nx)
        Bd_t = torch.as_tensor(Bd_np, dtype=self.dtype, device=self.device).view(self.nx, self.nu)
        cd_t = torch.as_tensor(cd_np, dtype=self.dtype, device=self.device).view(self.nx, 1)

        (u_star_batch,) = self.cocp_layer(
            self._batch(x),
            self._batch(x_ref),
            self._batch(theta),
            self._batch(self.u_min_torch),
            self._batch(self.u_max_torch),
            self._batch(Ad_t),
            self._batch(Bd_t),
            self._batch(cd_t),
            self._batch(self.W1),
            self._batch(self.b1),
            self._batch(self.W2z),
            self._batch(self.W2x),
            self._batch(self.b2),
            self._batch(self.Woutz),
            self._batch(self.Woutx),
            self._batch(self.bout),
        )
        return u_star_batch.squeeze(0).squeeze(-1)

    def act_np(self, x_np: np.ndarray, xref_np: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper for numpy usage (eval mode).
        """
        x_t = torch.tensor(x_np, dtype=self.dtype, device=self.device)
        xref_t = torch.tensor(xref_np, dtype=self.dtype, device=self.device)
        u_t = self.forward_eval(x_t, xref_t)
        return u_t.cpu().numpy()
