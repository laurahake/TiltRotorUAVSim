import numpy as np
from scipy.interpolate import CubicSpline

class TrajectoryGenerator:
    def __init__(self, knots, speeds):
        """
        knots: np.array shape (n, 3) — 3D position waypoints like in equation(26)
        speeds: np.array shape (n,) — desired speed at each knot like in eqaution (27)
        """
        self.knots = knots
        self.speeds = speeds
        self.n = len(knots)

        self.s = np.arange(self.n)
        self.spline_x = CubicSpline(self.s, knots[:, 0], bc_type='natural')
        self.spline_y = CubicSpline(self.s, knots[:, 1], bc_type='natural')
        self.spline_z = CubicSpline(self.s, knots[:, 2], bc_type='natural')
        self.speed_interp = CubicSpline(self.s, speeds, bc_type='natural')

    def get_pos(self, s):
        """        
        Get the position at a given s value
        s: float or np.array — parameter along the trajectory
        Returns: np.array shape (3,) — position vector [x, y, z]
        """
        # based on equation (28)
        return np.array([
            self.spline_x(s),
            self.spline_y(s),
            self.spline_z(s)
        ])

    def get_vel(self, s):
        """
        Compute desired velocity vector at parameter s using Equation (29):
        - Speed is linearly interpolated between knot speeds.
        - Direction is taken from the tangent of the position spline.
        """
        # 1. Compute the tangent vector dP/ds
        dx = self.spline_x.derivative()(s)
        dy = self.spline_y.derivative()(s)
        dz = self.spline_z.derivative()(s)
        tangent = np.array([dx, dy, dz])

        # 2. Normalize the tangent vector
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm == 0:
            return np.zeros(3)
        direction = tangent / tangent_norm

        # 3. Compute s_knot and s_star (local segment)
        s_knot = int(np.floor(s))
        s_star = s - s_knot

        # Clamp s_knot to valid range
        if s_knot >= self.n - 1:
            s_knot = self.n - 2
            s_star = 1.0
        elif s_knot < 0:
            s_knot = 0
            s_star = 0.0

        # 4. Perform linear interpolation of speed (Equation 29)
        v0 = self.speeds[s_knot]
        v1 = self.speeds[s_knot + 1]
        speed = (1 - s_star) * v0 + s_star * v1

        # 5. Scale direction by interpolated speed
        # based on equation (35)
        return speed * direction
    
    
    def get_desired_state(self, s):
        """
        Returns desired position, velocity, and orientation (as quaternion) at parameter s.
        Returns:
            p_des: np.array (3,) — desired position
            v_des: np.array (3,) — desired velocity
            quat_des: np.array (4,) — desired quaternion [w, x, y, z]
        """
        # 1. Desired position
        p_des = self.get_pos(s)

        # 2. Desired velocity
        v_des = self.get_vel(s)

        # 3. Get direction from spline tangent
        # derivative of spline at s - corresponding to equation (30)
        dx = self.spline_x.derivative()(s)
        dy = self.spline_y.derivative()(s)
        dz = self.spline_z.derivative()(s)
        tangent = np.array([dx, dy, dz])
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm == 0:
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            tangent = tangent / tangent_norm

        # 4. Desired yaw from tangent
        yaw = np.arctan2(tangent[1], tangent[0])
        # we intend the UAV to be near level flight, so the desired roll and pitch are zero
        pitch = 0.0
        roll = 0.0

        # 5. Convert Euler angles to quaternion [w, x, y, z]
        # instead of equation (31), we use quaternions
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        quat_des = np.array([qw, qx, qy, qz])

        return p_des, v_des, quat_des