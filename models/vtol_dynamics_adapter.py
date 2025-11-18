import numpy as np
from message_types.msg_delta import MsgDelta
from models.vtol_dynamics import VtolDynamics

class VtolDynamicsAdapter:
    """
        Adapter to use VtolDynamics in a format suitable for fast timescale controller
        Use physics to calulate f(x,u)
    """
    def __init__(self, vtol):
        self.vtol = vtol

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

    def f(self, x, u, wind=None):
        if wind is None:
            wind = np.zeros((6,1))
        self.vtol.external_set_state(x.reshape(-1,1))
        self.vtol._update_velocity_data(wind)
        delta = self._u_to_delta(u)
        fm = self.vtol._forces_moments(delta)
        xdot = self.vtol._derivatives(self.vtol._state, fm, delta)
        return xdot.squeeze()