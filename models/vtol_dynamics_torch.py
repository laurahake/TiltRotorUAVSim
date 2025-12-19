from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

import parameters.convergence_parameters as VTOL


Tensor = torch.Tensor


def quaternion_to_rotation_torch(q: Tensor) -> Tensor:
    """
    q = [e0, e1, e2, e3] with scalar-first convention.
    Returns rotation matrix R (3x3) from body to inertial (same as your numpy helper).
    """
    e0, e1, e2, e3 = q[0], q[1], q[2], q[3]

    # Precompute products
    e0e0 = e0 * e0
    e1e1 = e1 * e1
    e2e2 = e2 * e2
    e3e3 = e3 * e3

    e0e1 = e0 * e1
    e0e2 = e0 * e2
    e0e3 = e0 * e3
    e1e2 = e1 * e2
    e1e3 = e1 * e3
    e2e3 = e2 * e3

    R = torch.stack([
        torch.stack([e0e0 + e1e1 - e2e2 - e3e3, 2.0*(e1e2 - e0e3),           2.0*(e1e3 + e0e2)]),
        torch.stack([2.0*(e1e2 + e0e3),           e0e0 - e1e1 + e2e2 - e3e3, 2.0*(e2e3 - e0e1)]),
        torch.stack([2.0*(e1e3 - e0e2),           2.0*(e2e3 + e0e1),           e0e0 - e1e1 - e2e2 + e3e3]),
    ])
    return R


@dataclass
class VtolTorchConfig:
    dt: float
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


class VtolDynamicsTorch(nn.Module):
    """
    torch version of 15-state VTOL dynamics.:

      x = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r, right_motor, left_motor]
      u_vec layout (wie in deinem Adapter):
        [throttle_rear, throttle_left, throttle_right, elevon_L, elevon_R, tilt_R_cmd, tilt_L_cmd]
    """

    def __init__(self, cfg: VtolTorchConfig):
        super().__init__()
        self.cfg = cfg
        self.dt = cfg.dt
        self.device = torch.device(cfg.device)
        self.dtype = cfg.dtype

        # Cache some constants as torch scalars (helps device/dtype consistency)
        self.mass = torch.tensor(VTOL.mass, device=self.device, dtype=self.dtype)
        self.gravity = torch.tensor(VTOL.gravity, device=self.device, dtype=self.dtype)
        self.rho = torch.tensor(VTOL.rho, device=self.device, dtype=self.dtype)

        # inertias & gamma params
        self.Jy = torch.tensor(VTOL.Jy, device=self.device, dtype=self.dtype)
        self.gamma1 = torch.tensor(VTOL.gamma1, device=self.device, dtype=self.dtype)
        self.gamma2 = torch.tensor(VTOL.gamma2, device=self.device, dtype=self.dtype)
        self.gamma3 = torch.tensor(VTOL.gamma3, device=self.device, dtype=self.dtype)
        self.gamma4 = torch.tensor(VTOL.gamma4, device=self.device, dtype=self.dtype)
        self.gamma5 = torch.tensor(VTOL.gamma5, device=self.device, dtype=self.dtype)
        self.gamma6 = torch.tensor(VTOL.gamma6, device=self.device, dtype=self.dtype)
        self.gamma7 = torch.tensor(VTOL.gamma7, device=self.device, dtype=self.dtype)
        self.gamma8 = torch.tensor(VTOL.gamma8, device=self.device, dtype=self.dtype)

        # aero/geom constants (same names as in your numpy code)
        self.S_wing = torch.tensor(VTOL.S_wing, device=self.device, dtype=self.dtype)
        self.b = torch.tensor(VTOL.b, device=self.device, dtype=self.dtype)
        self.c = torch.tensor(VTOL.c, device=self.device, dtype=self.dtype)
        self.e = torch.tensor(VTOL.e, device=self.device, dtype=self.dtype)
        self.AR = torch.tensor(VTOL.AR, device=self.device, dtype=self.dtype)

        self.C_L_0 = torch.tensor(VTOL.C_L_0, device=self.device, dtype=self.dtype)
        self.C_L_alpha = torch.tensor(VTOL.C_L_alpha, device=self.device, dtype=self.dtype)
        self.C_L_q = torch.tensor(VTOL.C_L_q, device=self.device, dtype=self.dtype)
        self.C_L_delta_e = torch.tensor(VTOL.C_L_delta_e, device=self.device, dtype=self.dtype)

        self.C_D_p = torch.tensor(VTOL.C_D_p, device=self.device, dtype=self.dtype)
        self.C_D_q = torch.tensor(VTOL.C_D_q, device=self.device, dtype=self.dtype)
        self.C_D_delta_e = torch.tensor(VTOL.C_D_delta_e, device=self.device, dtype=self.dtype)

        self.C_Y_0 = torch.tensor(VTOL.C_Y_0, device=self.device, dtype=self.dtype)
        self.C_Y_beta = torch.tensor(VTOL.C_Y_beta, device=self.device, dtype=self.dtype)
        self.C_Y_p = torch.tensor(VTOL.C_Y_p, device=self.device, dtype=self.dtype)
        self.C_Y_r = torch.tensor(VTOL.C_Y_r, device=self.device, dtype=self.dtype)
        self.C_Y_delta_a = torch.tensor(VTOL.C_Y_delta_a, device=self.device, dtype=self.dtype)
        self.C_Y_delta_r = torch.tensor(VTOL.C_Y_delta_r, device=self.device, dtype=self.dtype)

        self.C_m_0 = torch.tensor(VTOL.C_m_0, device=self.device, dtype=self.dtype)
        self.C_m_alpha = torch.tensor(VTOL.C_m_alpha, device=self.device, dtype=self.dtype)
        self.C_m_q = torch.tensor(VTOL.C_m_q, device=self.device, dtype=self.dtype)
        self.C_m_delta_e = torch.tensor(VTOL.C_m_delta_e, device=self.device, dtype=self.dtype)

        self.C_ell_0 = torch.tensor(VTOL.C_ell_0, device=self.device, dtype=self.dtype)
        self.C_ell_beta = torch.tensor(VTOL.C_ell_beta, device=self.device, dtype=self.dtype)
        self.C_ell_p = torch.tensor(VTOL.C_ell_p, device=self.device, dtype=self.dtype)
        self.C_ell_r = torch.tensor(VTOL.C_ell_r, device=self.device, dtype=self.dtype)
        self.C_ell_delta_a = torch.tensor(VTOL.C_ell_delta_a, device=self.device, dtype=self.dtype)
        self.C_ell_delta_r = torch.tensor(VTOL.C_ell_delta_r, device=self.device, dtype=self.dtype)

        self.C_n_0 = torch.tensor(VTOL.C_n_0, device=self.device, dtype=self.dtype)
        self.C_n_beta = torch.tensor(VTOL.C_n_beta, device=self.device, dtype=self.dtype)
        self.C_n_p = torch.tensor(VTOL.C_n_p, device=self.device, dtype=self.dtype)
        self.C_n_r = torch.tensor(VTOL.C_n_r, device=self.device, dtype=self.dtype)
        self.C_n_delta_a = torch.tensor(VTOL.C_n_delta_a, device=self.device, dtype=self.dtype)
        self.C_n_delta_r = torch.tensor(VTOL.C_n_delta_r, device=self.device, dtype=self.dtype)

        self.M = torch.tensor(VTOL.M, device=self.device, dtype=self.dtype)
        self.alpha0 = torch.tensor(VTOL.alpha0, device=self.device, dtype=self.dtype)

        # rotor/servo params
        self.k_servo = torch.tensor(VTOL.k_servo, device=self.device, dtype=self.dtype)
        self.V_max = torch.tensor(VTOL.V_max, device=self.device, dtype=self.dtype)

        # rotor positions (3,) tensors
        self.rear_rotor_pos = torch.tensor(VTOL.rear_rotor_pos.reshape(3,), device=self.device, dtype=self.dtype)
        self.right_rotor_pos = torch.tensor(VTOL.right_rotor_pos.reshape(3,), device=self.device, dtype=self.dtype)
        self.left_rotor_pos = torch.tensor(VTOL.left_rotor_pos.reshape(3,), device=self.device, dtype=self.dtype)

        # prop constants (rear vs front) stored as python floats in VTOL; we use on the fly below.

    # ---------------------------
    # Public API
    # ---------------------------
    def f_cont(self, x: Tensor, u_vec: Tensor, wind: Optional[Tensor] = None) -> Tensor:
        """
        Continuous-time dynamics xdot = f(x,u).
        wind: optional Tensor shape (6,) = [steady_n, steady_e, steady_d, gust_x, gust_y, gust_z]
              convention like your numpy: steady in inertial, gust in body.
        """
        x = x.to(self.device, self.dtype)
        u_vec = u_vec.to(self.device, self.dtype)
        if wind is None:
            wind = torch.zeros(6, device=self.device, dtype=self.dtype)
        else:
            wind = wind.to(self.device, self.dtype)

        # unpack state
        pn, pe, pd = x[0], x[1], x[2]
        u_b, v_b, w_b = x[3], x[4], x[5]
        e0, e1, e2, e3 = x[6], x[7], x[8], x[9]
        p, q, r = x[10], x[11], x[12]
        right_motor, left_motor = x[13], x[14]

        # unpack controls (same as adapter mapping)
        thr_rear = u_vec[0]
        thr_left = u_vec[1]
        thr_right = u_vec[2]
        elevon_L = u_vec[3]
        elevon_R = u_vec[4]
        motor_right_cmd = u_vec[5]  # tilt command (rad)
        motor_left_cmd = u_vec[6]

        elevator = elevon_L + elevon_R
        aileron = elevon_L - elevon_R
        rudder = torch.zeros((), device=self.device, dtype=self.dtype)

        # compute forces/moments
        Fx, Fy, Fz, Mx, My, Mz, Va, alpha, beta, v_air_body = self.forces_moments(
            x=x,
            elevator=elevator,
            aileron=aileron,
            rudder=rudder,
            thr_rear=thr_rear,
            thr_left=thr_left,
            thr_right=thr_right,
            wind=wind,
        )

        # kinematics: inertial position derivative
        R_bi = quaternion_to_rotation_torch(torch.stack([e0, e1, e2, e3]))
        vel_body = torch.stack([u_b, v_b, w_b])
        pos_dot = R_bi @ vel_body
        pn_dot, pe_dot, pd_dot = pos_dot[0], pos_dot[1], pos_dot[2]

        # translational dynamics in body
        u_dot = r * v_b - q * w_b + Fx / self.mass
        v_dot = p * w_b - r * u_b + Fy / self.mass
        w_dot = q * u_b - p * v_b + Fz / self.mass

        # quaternion kinematics
        e0_dot = 0.5 * (-p * e1 - q * e2 - r * e3)
        e1_dot = 0.5 * ( p * e0 + r * e2 - q * e3)
        e2_dot = 0.5 * ( q * e0 - r * e1 + p * e3)
        e3_dot = 0.5 * ( r * e0 + q * e1 - p * e2)

        # rotational dynamics
        p_dot = self.gamma1 * p * q - self.gamma2 * q * r + self.gamma3 * Mx + self.gamma4 * Mz
        q_dot = self.gamma5 * p * r - self.gamma6 * (p * p - r * r) + My / self.Jy
        r_dot = self.gamma7 * p * q - self.gamma1 * q * r + self.gamma4 * Mx + self.gamma8 * Mz

        # motor servo dynamics
        right_motor_dot = self.k_servo * (motor_right_cmd - right_motor)
        left_motor_dot  = self.k_servo * (motor_left_cmd  - left_motor)

        xdot = torch.stack([
            pn_dot, pe_dot, pd_dot,
            u_dot, v_dot, w_dot,
            e0_dot, e1_dot, e2_dot, e3_dot,
            p_dot, q_dot, r_dot,
            right_motor_dot, left_motor_dot
        ])
        return xdot

    def f_disc(self, x: Tensor, u_vec: Tensor, wind: Optional[Tensor] = None) -> Tensor:
        """
        Discrete Euler foward step:
          x_{k+1} = x_k + dt * f_cont(x_k, u_k)
        plus quaternion normalization.
        """
        x = x.to(self.device, self.dtype)
        xdot = self.f_cont(x, u_vec, wind=wind)
        x_next = x + self.dt * xdot

        # normalize quaternion
        q = x_next[6:10]
        q = q / (torch.linalg.norm(q) + 1e-12)
        x_next = x_next.clone()
        x_next[6:10] = q
        return x_next

    # ---------------------------
    # Forces and moments (torch port)
    # ---------------------------
    def forces_moments(
        self,
        x: Tensor,
        elevator: Tensor,
        aileron: Tensor,
        rudder: Tensor,
        thr_rear: Tensor,
        thr_left: Tensor,
        thr_right: Tensor,
        wind: Tensor,
    ):
        """
        Returns:
          Fx,Fy,Fz,Mx,My,Mz plus some aero intermediates (Va, alpha, beta, v_air_body)
        """
        # state unpack
        u_b, v_b, w_b = x[3], x[4], x[5]
        q_state = x[6:10]
        p, q, r = x[10], x[11], x[12]
        right_motor = x[13]
        left_motor = x[14]

        # wind handling (like numpy _update_velocity_data):
        steady = wind[0:3]                 # inertial
        gust_body = wind[3:6]              # body
        R_bi = quaternion_to_rotation_torch(q_state)  # body->inertial
        R_ib = R_bi.transpose(0, 1)

        wind_body = R_ib @ steady + gust_body
        v_air_body = torch.stack([u_b, v_b, w_b]) - wind_body

        Va = torch.linalg.norm(v_air_body)
        ur, vr, wr = v_air_body[0], v_air_body[1], v_air_body[2]

        # alpha, beta (match numpy logic)
        alpha = torch.where(
            torch.abs(ur) < 1e-12,
            torch.sign(wr) * (torch.pi / 2.0),
            torch.atan2(wr, ur),
        )
        beta = torch.where(
            Va < 1e-12,
            torch.sign(vr) * (torch.pi / 2.0),
            torch.asin(torch.clamp(vr / (Va + 1e-12), -1.0, 1.0)),
        )

        # gravitational force in body frame
        f_g_inertial = torch.stack([torch.zeros_like(self.gravity), torch.zeros_like(self.gravity), self.gravity * self.mass])
        f_g_body = R_ib @ f_g_inertial
        fx, fy, fz = f_g_body[0], f_g_body[1], f_g_body[2]

        # aerodynamic intermediates
        qbar = 0.5 * self.rho * Va * Va
        ca = torch.cos(alpha)
        sa = torch.sin(alpha)

        # nondim rates
        p_nd = torch.where(Va > 1.0, p * self.b / (2.0 * Va), torch.zeros_like(p))
        q_nd = torch.where(Va > 1.0, q * self.c / (2.0 * Va), torch.zeros_like(q))
        r_nd = torch.where(Va > 1.0, r * self.b / (2.0 * Va), torch.zeros_like(r))

        # sigma blend (same as numpy)
        tmp1 = torch.exp(-self.M * (alpha - self.alpha0))
        tmp2 = torch.exp( self.M * (alpha + self.alpha0))
        sigma = (1.0 + tmp1 + tmp2) / ((1.0 + tmp1) * (1.0 + tmp2))

        CL_lin = self.C_L_0 + self.C_L_alpha * alpha
        CL = (1.0 - sigma) * CL_lin + sigma * 2.0 * torch.sign(alpha) * sa * sa * ca

        CD_lin = self.C_D_p + (CL_lin * CL_lin) / (torch.pi * self.e * self.AR)
        CD = (1.0 - sigma) * CD_lin + sigma * 2.0 * torch.sign(alpha) * sa

        # Lift/Drag forces
        F_lift = qbar * self.S_wing * (CL + self.C_L_q * q_nd + self.C_L_delta_e * elevator)
        F_drag = qbar * self.S_wing * (CD + self.C_D_q * q_nd + self.C_D_delta_e * elevator)

        # body forces (longitudinal)
        fx = fx + (-ca * F_drag + sa * F_lift)
        fz = fz + (-sa * F_drag - ca * F_lift)

        # lateral force
        fy = fy + qbar * self.S_wing * (
            self.C_Y_0
            + self.C_Y_beta * beta
            + self.C_Y_p * p_nd
            + self.C_Y_r * r_nd
            + self.C_Y_delta_a * aileron
            + self.C_Y_delta_r * rudder
        )

        # torques
        My = qbar * self.S_wing * self.c * (
            self.C_m_0 + self.C_m_alpha * alpha + self.C_m_q * q_nd + self.C_m_delta_e * elevator
        )

        Mx = qbar * self.S_wing * self.b * (
            self.C_ell_0 + self.C_ell_beta * beta + self.C_ell_p * p_nd + self.C_ell_r * r_nd
            + self.C_ell_delta_a * aileron + self.C_ell_delta_r * rudder
        )

        Mz = qbar * self.S_wing * self.b * (
            self.C_n_0 + self.C_n_beta * beta + self.C_n_p * p_nd + self.C_n_r * r_nd
            + self.C_n_delta_a * aileron + self.C_n_delta_r * rudder
        )

        # motor thrust/torque (rear)
        Va_rear = -v_air_body[2]  # matches [0,0,-1]^T @ v_air
        Thrust_rear, Torque_rear = self.motor_thrust_torque(Va_rear, thr_rear, is_rear=True)
        Force_rear = torch.stack([torch.zeros_like(Thrust_rear), torch.zeros_like(Thrust_rear), -Thrust_rear])
        Moment_rear = torch.stack([torch.zeros_like(Torque_rear), torch.zeros_like(Torque_rear), Torque_rear]) \
                      + torch.linalg.cross(self.rear_rotor_pos, Force_rear)

        fx = fx + Force_rear[0]
        fy = fy + Force_rear[1]
        fz = fz + Force_rear[2]
        Mx = Mx + Moment_rear[0]
        My = My + Moment_rear[1]
        Mz = Mz + Moment_rear[2]

        # right motor orientation in body frame depends on right_motor state angle
        ur_vec = torch.stack([torch.cos(right_motor), torch.zeros_like(right_motor), -torch.sin(right_motor)])
        Va_right = (ur_vec * v_air_body).sum()
        Thrust_r, Torque_r = self.motor_thrust_torque(Va_right, thr_right, is_rear=False)
        Force_right = Thrust_r * ur_vec
        Moment_right = (-Torque_r) * ur_vec + torch.linalg.cross(self.right_rotor_pos, Force_right)

        fx = fx + Force_right[0]
        fy = fy + Force_right[1]
        fz = fz + Force_right[2]
        Mx = Mx + Moment_right[0]
        My = My + Moment_right[1]
        Mz = Mz + Moment_right[2]

        # left motor
        ul_vec = torch.stack([torch.cos(left_motor), torch.zeros_like(left_motor), -torch.sin(left_motor)])
        Va_left = (ul_vec * v_air_body).sum()
        Thrust_l, Torque_l = self.motor_thrust_torque(Va_left, thr_left, is_rear=False)
        Force_left = Thrust_l * ul_vec
        Moment_left = (-Torque_l) * ul_vec + torch.linalg.cross(self.left_rotor_pos, Force_left)

        fx = fx + Force_left[0]
        fy = fy + Force_left[1]
        fz = fz + Force_left[2]
        Mx = Mx + Moment_left[0]
        My = My + Moment_left[1]
        Mz = Mz + Moment_left[2]

        return fx, fy, fz, Mx, My, Mz, Va, alpha, beta, v_air_body

    def motor_thrust_torque(self, Va: Tensor, delta_t: Tensor, is_rear: bool):
        """
        Torch-port of _motor_thrust_torque. Uses same VTOL constants.
        """
        if is_rear:
            C_Q0 = VTOL.C_Q0_rear; C_Q1 = VTOL.C_Q1_rear; C_Q2 = VTOL.C_Q2_rear
            C_T0 = VTOL.C_T0_rear; C_T1 = VTOL.C_T1_rear; C_T2 = VTOL.C_T2_rear
            D_prop = VTOL.D_prop_rear
            KQ = VTOL.KQ_rear
            R_motor = VTOL.R_motor_rear
            i0 = VTOL.i0_rear
        else:
            C_Q0 = VTOL.C_Q0_front; C_Q1 = VTOL.C_Q1_front; C_Q2 = VTOL.C_Q2_front
            C_T0 = VTOL.C_T0_front; C_T1 = VTOL.C_T1_front; C_T2 = VTOL.C_T2_front
            D_prop = VTOL.D_prop_front
            KQ = VTOL.KQ_front
            R_motor = VTOL.R_motor_front
            i0 = VTOL.i0_front

        # convert to torch scalars on correct device/dtype
        C_Q0 = torch.tensor(C_Q0, device=self.device, dtype=self.dtype)
        C_Q1 = torch.tensor(C_Q1, device=self.device, dtype=self.dtype)
        C_Q2 = torch.tensor(C_Q2, device=self.device, dtype=self.dtype)
        C_T0 = torch.tensor(C_T0, device=self.device, dtype=self.dtype)
        C_T1 = torch.tensor(C_T1, device=self.device, dtype=self.dtype)
        C_T2 = torch.tensor(C_T2, device=self.device, dtype=self.dtype)
        D_prop = torch.tensor(D_prop, device=self.device, dtype=self.dtype)
        KQ = torch.tensor(KQ, device=self.device, dtype=self.dtype)
        R_motor = torch.tensor(R_motor, device=self.device, dtype=self.dtype)
        i0 = torch.tensor(i0, device=self.device, dtype=self.dtype)

        # input voltage
        V_in = self.V_max * delta_t

        # Quadratic formula for Omega
        a = C_Q0 * self.rho * (D_prop ** 5) / ((2.0 * torch.pi) ** 2)
        b = (C_Q1 * self.rho * (D_prop ** 4) / (2.0 * torch.pi)) * Va + (KQ * KQ) / R_motor
        c = C_Q2 * self.rho * (D_prop ** 3) * (Va ** 2) - (KQ / R_motor) * V_in + KQ * i0

        disc = torch.clamp(b * b - 4.0 * a * c, min=0.0)
        Omega = (-b + torch.sqrt(disc)) / (2.0 * a + 1e-12)

        # advance ratio
        J = 2.0 * torch.pi * Va / (Omega * D_prop + 1e-12)

        # coefficients
        C_T = C_T2 * (J ** 2) + C_T1 * J + C_T0
        C_Q = C_Q2 * (J ** 2) + C_Q1 * J + C_Q0

        n = Omega / (2.0 * torch.pi)
        T_p = self.rho * (n ** 2) * (D_prop ** 4) * C_T
        Q_p = self.rho * (n ** 2) * (D_prop ** 5) * C_Q
        return T_p, Q_p
