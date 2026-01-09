import math
from dataclasses import dataclass

@dataclass
class TiltConfig:
    trigger_height_m: float = 10.0     # start transition once h >= this
    hysteresis_m: float = 1.0          # avoids flip-flop when hovering near threshold
    transition_time_s: float = 3.0     # time to rotate from π/2 to 0
    vertical_angle: float = math.pi/2  # vertical takeoff tilt [rad]
    forward_angle: float  = 0.0        # cruise tilt [rad]
    angle_min: float = 0.0             # safety clamp
    angle_max: float = math.radians(115.0)

class TiltState:
    VTO = 0          # vertical
    TRANSITION = 1   # rotating
    FORWARD = 2      # done
    

class TiltRotorSwitcher:
    def __init__(self, cfg: TiltConfig):
        self.cfg = cfg
        self.state = TiltState.VTO
        self.t_elapsed = 0.0

    def reset(self):
        self.state = TiltState.VTO
        self.t_elapsed = 0.0

    @staticmethod
    def _s_curve01(s):
        # cosine smoothstep on [0,1]
        s = max(0.0, min(1.0, s))
        return 0.5 - 0.5*math.cos(math.pi*s)

    def _blend_tilt(self, s):
        c = self.cfg
        tilt = (1.0 - s) * c.vertical_angle + s * c.forward_angle
        return max(c.angle_min, min(c.angle_max, tilt))

    def step_ned(self, d_down: float, dt: float):
        """
        NED input:
          d_down: 'Down' position (z in NED). Down is positive; altitude h = -d_down.
          dt:     timestep [s]
        Returns:
          (tilt_right, tilt_left) target angles [rad]
        """
        c = self.cfg
        h = -float(d_down)  # altitude from NED Down

        if self.state == TiltState.VTO:
            tilt = c.vertical_angle
            if h >= c.trigger_height_m:
                self.state = TiltState.TRANSITION
                self.t_elapsed = 0.0

        elif self.state == TiltState.TRANSITION:
            self.t_elapsed += dt
            s = self._s_curve01(self.t_elapsed / max(1e-6, c.transition_time_s))
            tilt = self._blend_tilt(s)
            if self.t_elapsed >= c.transition_time_s:
                self.state = TiltState.FORWARD
                tilt = self._blend_tilt(1.0)

        else:  # FORWARD
            if h <= c.trigger_height_m - c.hysteresis_m:
                self.state = TiltState.VTO
                tilt = c.vertical_angle
            else:
                tilt = c.forward_angle

        return tilt, tilt