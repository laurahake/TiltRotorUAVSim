import numpy as np
from scipy.linalg import solve_continuous_are
from TiltRotorUAV import TiltRotorUAV, UAVStateQuat
from scipy.spatial.transform import Rotation

class StateDependentLQRController:
    def __init__(self, uav, Q=None, R=None):
        self.uav = uav
        self.Q = Q if Q is not None else np.diag([.1]*6 + [2]*3)
        self.R = R if R is not None else np.diag([1, 1, 10, 10, 150])

    def error_state(self, desired, actual):
        # based on equation (40)
        p_error = actual.p - desired.p
        v_error = actual.v - desired.v

        if np.linalg.norm(actual.quat) < 1e-8:
            print("actual quaternion is nearly zero — invalid")
        if np.linalg.norm(desired.quat) < 1e-8:
            print("desired quaternion is nearly zero — invalid")
        R = actual.rotation_matrix()
        R_d = desired.rotation_matrix()
        R_err = R.T @ R_d
        rotvec_err = UAVStateQuat.rotation_to_vector(R_err)
        return np.concatenate([p_error, v_error, rotvec_err])
    

    def so3_log(self, R):
        """
        Logarithm map from SO(3) to so(3)
        Converts rotation matrix to rotation vector (axis-angle representation)

        Args:
            R (3x3 np.ndarray): Rotation matrix

        Returns:
            np.ndarray: 3D rotation vector (theta)
        """
        if np.isnan(R).any() or np.isinf(R).any():
            print("Rotation matrix has NaN or Inf:", R)
            
        cos_theta = (np.trace(R) - 1) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure numerical stability
        theta = np.arccos(cos_theta)
        if np.isnan(theta) or np.isinf(theta):
            print("Invalid theta from rotation matrix:", theta)
            return np.zeros(3)

        if np.isclose(theta, 0.0):
            return np.zeros(3)

        skew = (R - R.T) / (2 * np.sin(theta))
        return theta * np.array([
            skew[2,1],
            skew[0,2],
            skew[1,0]
        ])


    def reduce_to_9d(self, dx_vec, desired_state, x_err):
        """
        Reduce full 15D derivative vector to 9D error state derivative.
        Assumes dx_vec is [dp, dv, dR_flat] from simplified_derivatives.
        """
        dp = dx_vec[0:3]
        dv = dx_vec[3:6]
        dR = dx_vec[6:15].reshape(3, 3)

        # Get the current actual rotation from error state
        R_des = desired_state.rotation_matrix()
        delta_theta = x_err[6:9]
        R_actual = R_des @ Rotation.from_rotvec(delta_theta).as_matrix()
        
        # Explicitly ensure orthogonality of R_actual to avoid numerical issues
        U, _, Vt = np.linalg.svd(R_actual)
        R_actual = U @ Vt  # re-orthogonalized

        # Compute rotation error derivative properly
        R_err_dot = R_des.T @ R_actual @ dR
        omega_err = self.so3_log(R_err_dot)

        return np.concatenate([dp, dv, omega_err])

    def linearize_error_dynamics(self, desired_state, actual_state, u):
        x_err = self.error_state(desired_state, actual_state)
        
        def error_dynamics_fn(x_err_local, desired_state_local, u_tilde):
            # Feedforward control (nominal input at desired state)
            u_hat = self.compute_feedforward_control(desired_state)

            # Total input to use in dynamics = feedforward - error input
            u_total = u_hat - u_tilde
            
            # Apply x_err to desired_state to get a full actual state
            x_actual_local = self.apply_error(desired_state_local, x_err_local)

            # Call simplified_derivatives
            dx_vec = self.uav.simplified_derivatives(x_actual_local, u_total)

            dx_err = self.reduce_to_9d(dx_vec, desired_state_local, x_err_local)
            return dx_err

        # Use Nielsen-style numerical Jacobian
        A, B = self.central_difference_jacobians(error_dynamics_fn, x_err, desired_state, u)
        ctrb = np.hstack([B] + [A @ B] + [A @ A @ B] + [A @ A @ A @ B])  # or use control.ctrb()
        rank = np.linalg.matrix_rank(ctrb)
    
        return A, B
    
    
    def apply_error(self, desired_state: UAVStateQuat, x_err: np.ndarray) -> UAVStateQuat:
        """
        Apply error vector to desired state to reconstruct the actual state.
        This is the inverse of error_state(...).
        """
        p_d = desired_state.p
        v_d = desired_state.v
        R_d = desired_state.rotation_matrix()

        p = p_d + x_err[0:3]
        v = v_d + x_err[3:6]

        # Orientation: use exponential map (hat operator)
        delta_theta = x_err[6:9]
        delta_R = Rotation.from_rotvec(delta_theta).as_matrix()
        R = R_d @ delta_R  # actual orientation

        quat = Rotation.from_matrix(R).as_quat()
        quat = quat / np.linalg.norm(quat)

        return UAVStateQuat(
            p=p,
            v=v,
            quat=quat,
            omega=np.zeros(3),
            theta=np.zeros(2)
        )
    

    def compute_control(self, desired_state, actual_state):
        u_hat = self.compute_feedforward_control(desired_state)
        x_err = self.error_state(desired_state, actual_state)
        A, B = self.linearize_error_dynamics(desired_state, actual_state, u_hat)
        P = solve_continuous_are(A, B, self.Q, self.R)
        K = np.linalg.inv(self.R) @ B.T @ P
        u_tilde = -K @ x_err
        return u_hat + u_tilde
    
    
    def compute_feedforward_control(self, desired_state):
        """
        Compute nominal feedforward control input (û) based on desired velocity.
        This is used as the base input around which we linearize.
        """
        v_I = desired_state.v
        v_h = np.linalg.norm(v_I[:2])
        v_v = v_I[2]
        v_mag = np.linalg.norm(v_I)
        g = self.uav.params['gravity']

        if v_mag == 0:
            a_x = 0.0
            a_z = g
        elif v_v > 0:
            a_x = 0.4 * (v_h / v_mag)
            a_z = 0.8 * abs(v_v / v_mag)
        else:
            a_x = 0.4 * 5.0 * (v_h / v_mag)
            a_z = 0.8 * abs(v_v / v_mag)

        return np.array([a_x, a_z, 0.0, 0.0, 0.0])  # û
    
    
    def central_difference_jacobians(self, error_dynamics_fn, x_err, desired_state, u, epsilon=1e-5):
        """
        Computes A = ∂f/∂x_err and B = ∂f/∂u using central difference method,
        following Nielsen's method on page 97/98.

        Parameters:
            error_dynamics_fn: function (x_err, desired_state, u) -> dx_err/dt
            x_err: current 9D error state vector
            desired_state: desired UAVStateQuat
            u: control input (5D vector)
            epsilon: small perturbation for differencing

        Returns:
            A: Jacobian ∂f/∂x_err (9x9)
            B: Jacobian ∂f/∂u (9x5)
        """
        n = x_err.shape[0]  # should be 9
        m = u.shape[0]      # should be 5

        A = np.zeros((n, n))
        B = np.zeros((n, m))

        # Compute A matrix
        for j in range(n):
            e = np.zeros(n)
            e[j] = epsilon

            f_plus = error_dynamics_fn(x_err + e, desired_state, u)
            f_minus = error_dynamics_fn(x_err - e, desired_state, u)
            A[:, j] = (f_plus - f_minus) / (2 * epsilon)

        # Compute B matrix
        for k in range(m):
            e = np.zeros(m)
            e[k] = epsilon

            f_plus = error_dynamics_fn(x_err, desired_state, u + e)
            f_minus = error_dynamics_fn(x_err, desired_state, u - e)
            B[:, k] = (f_plus - f_minus) / (2 * epsilon)

        if np.isnan(A).any() or np.isnan(B).any():
            print("Rotation matrix has NaN or Inf:", A, B)
        return A, B

