import numpy as np
import scipy.integrate as spi
from scipy.spatial.transform import Rotation


class UAVStateQuat:
    def __init__(self, 
                 p=np.zeros(3), 
                 v=np.zeros(3), 
                 quat=np.array([0, 0, 0, 1]),  # [x, y, z, w]
                 omega=np.zeros(3), 
                 theta=np.zeros(2)):
        """
        UAV full nonlinear state (quaternion-based).
        - p: position in inertial frame (3,)
        - v: velocity in body frame (3,)
        - quat: orientation as quaternion [x, y, z, w]
        - omega: angular velocity in body frame (3,)
        - theta: front rotor servo angles (2,)
        """
        self.p = p.astype(float)
        self.v = v.astype(float)
        self.quat = quat.astype(float)  # unit quaternion
        self.omega = omega.astype(float)
        self.theta = theta.astype(float)

    def as_vector(self):
        """Flatten the state to a 1D numpy array (15,)"""
        return np.concatenate([
            self.p,      # 3
            self.v,      # 3
            self.quat,   # 4
            self.omega,  # 3
            self.theta   # 2
        ])

    @classmethod
    def from_vector(cls, vec):
        """Reconstruct state from a 1D array of shape (15,)"""
        assert len(vec) == 15
        p = vec[0:3]
        v = vec[3:6]
        quat = vec[6:10]
        omega = vec[10:13]
        theta = vec[13:15]
        return cls(p, v, quat, omega, theta)

    @classmethod
    def from_reduced_vector(cls, vec):
            """
            Reconstruct a UAVStateQuat-like object from a 9D vector:
            [px, py, pz, vx, vy, vz, wx, wy, wz]
            """
            assert len(vec) == 9

            obj = cls()
            obj.p = vec[0:3]
            obj.v = vec[3:6]
            obj.omega = vec[6:9]

            # Leave quat and theta at defaults (they're not used in linearization)
            return obj

    def rotation_matrix(self):
        """Return the rotation matrix corresponding to the current quaternion"""
        return Rotation.from_quat(self.quat).as_matrix()

    def normalize_quat(self):
        """Ensure the quaternion stays normalized"""
        self.quat = self.quat / np.linalg.norm(self.quat)
        
    @staticmethod
    def rotation_to_vector(R):
        """Convert a rotation matrix to a rotation vector (Lie algebra so(3))"""
        return Rotation.from_matrix(R).as_rotvec()


class TiltRotorUAV:
    def __init__(self, params=None):
        self.state = UAVStateQuat()
        self.params = params or self.default_params()
        
    def default_params(self):
        """Default parameters for the UAV."""
        return {
            'mass': 0.771,
            'gravity': 9.81,
            'J': np.array([[0.0165, 0.0,     0.000048],
                        [0.0,     0.0128, 0.0],
                        [0.000048, 0.0,   0.0282]]),
            'rho': 1.2682,
            'S': 0.2589,
            'b': 1.4224,
            'c': 0.3305,
            'es': 0.9,
            'AR': 1.4224**2 / 0.2589, #Aspect Ratio b^2 / S

            # Front motor parameters
            'Vmax_front': 11.1,
            'D_front': 0.18,
            'KQ_front': 0.0066,
            'Rm_front': 0.3,
            'I0_front': 0.83,
            'CQ_front': [0.0088, 0.0129, -0.0216],
            'CT_front': [0.1167, 0.0144, -0.1480],

            # Rear motor parameters
            'Vmax_rear': 11.1,
            'D_rear': 0.14,
            'KQ_rear': 0.0062,
            'Rm_rear': 0.4,
            'I0_rear': 0.6,
            'CQ_rear': [0.0216, 0.0292, -0.0368],
            'CT_rear': [0.2097, 0.0505, -0.1921],

            # Longitudinal aerodynamic coefficients
            'CL0': 0.005,
            'CL_alpha': 2.819,
            'CL_q': 3.242,
            'CL_delta_e': 0.2,
            'CD0': 0.0022,
            'CD_alpha': 0.003,
            'CD_delta_e': 0.005,
            'Cm0': 0.0,
            'Cm_alpha': -0.185,
            'Cm_q': -1.093,
            'Cm_delta_e': -0.387,

            # Lateral aerodynamic coefficients
            'CY0': 0.0,
            'CY_beta': -0.318,
            'CY_p': 0.078,
            'CY_r': 0.288,
            'CY_delta_e': 0.000536,
            'Cl0': 0.0,
            'Cl_beta': -0.032,
            'Cl_p': -0.207,
            'Cl_r': 0.036,
            'Cl_delta_e': 0.018,
            'Cn0': 0.0,
            'Cn_beta': 0.112,
            'Cn_p': -0.053,
            'Cn_r': -0.104,
            'Cn_delta_e': -0.00328,

            # Blending function parameters
            'alpha0': 0.47,
            'd_blend': 50,
        }


    def sigma(self, alpha):
        # based on equation (18)
        d = self.params['d_blend']
        alpha0 = self.params['alpha0']
        return (1 + np.exp(-d * (alpha - alpha0)) + np.exp(d * (alpha + alpha0))) / \
            ((1 + np.exp(-d * (alpha - alpha0))) * (1 + np.exp(d * (alpha + alpha0))))

    def CL(self, alpha):
        # based on equation (16)
        return (1 - self.sigma(alpha)) * (self.params['CL0'] + self.params['CL_alpha'] * alpha) + \
            self.sigma(alpha) * (2 * np.sign(alpha) * np.sin(alpha)**2 * np.cos(alpha))

    def CD(self, alpha):
        #based on equation (17)
        CL_basic = self.params['CL0'] + self.params['CL_alpha'] * alpha
        return self.params['CD0'] + (CL_basic ** 2) / (np.pi * self.params['es'] * self.params['AR'])

    def R_alpha(self, alpha):
        #based on equation (11)
        return np.array([
            [np.cos(alpha), 0, -np.sin(alpha)],
            [0, 1, 0],
            [np.sin(alpha), 0, np.cos(alpha)]
        ])

    def compute_aerodynamic_forces(self, va, omega, delta_e):
        rho, S, b, c = self.params['rho'], self.params['S'], self.params['b'], self.params['c']
        alpha = np.arctan2(va[2], va[0])
        va_norm = np.linalg.norm(va)
        if va_norm < 1e-6:
            beta = 0.0
        else:
            beta = np.arcsin(va[1] / va_norm)
        R_alpha_mat = self.R_alpha(alpha)
        
        #based on equation (13)
        F0 = 0.5 * rho * np.linalg.norm(va)**2 * S * R_alpha_mat @ np.array([
            -c * self.CD(alpha),
            b * (self.params['CY0'] + self.params['CY_beta'] * beta),
            -c * self.CL(alpha)
        ])

        #based on equation (14)
        Fomega = 0.25 * rho * np.linalg.norm(va) * S * R_alpha_mat @ np.array([
            0,
            b**2 * self.params['CY_p'] * omega[0] + b**2 * self.params['CY_r'] * omega[2],
            c**2 * self.params['CL_q'] * omega[1]
        ])

        #based on equation (15)
        Fdelta = 0.5 * rho * np.linalg.norm(va)**2 * S * R_alpha_mat @ np.array([
            c * self.params['CD_delta_e'] * delta_e[0],
            b * self.params['CY_delta_e'] * delta_e[0],
            c * self.params['CL_delta_e'] * delta_e[0]
        ])

        return F0 + Fomega + Fdelta
    
    
    def compute_aerodynamic_moments(self, va, omega, delta_e):
        rho, S, b, c = self.params['rho'], self.params['S'], self.params['b'], self.params['c']
        alpha = np.arctan2(va[2], va[0])
        if np.linalg.norm(va) < 1e-6:
            beta = 0.0
        else:
            beta = np.arcsin(va[1] / np.linalg.norm(va))

        # based on equation (8)
        M0 = 0.5 * rho * np.linalg.norm(va)**2 * S * np.array([
            b * (self.params['Cl0'] + self.params['Cl_beta'] * beta),
            c * (self.params['Cm0'] + self.params['Cm_alpha'] * alpha),
            b * (self.params['Cn0'] + self.params['Cn_beta'] * beta)
        ])

        # based on equation (9)
        Momega = 0.25 * rho * np.linalg.norm(va) * S * np.array([
            b**2 * self.params['Cl_p'] * omega[0] + b**2 * self.params['Cl_r'] * omega[2],
            c**2 * self.params['Cm_q'] * omega[1],
            b**2 * self.params['Cn_p'] * omega[0] + b**2 * self.params['Cn_r'] * omega[2]
        ])

        # based on equation (10)
        Mdelta = 0.5 * rho * np.linalg.norm(va)**2 * S * np.array([
            self.params['Cl_delta_e'] * delta_e[0],
            self.params['Cm_delta_e'] * delta_e[0],
            self.params['Cn_delta_e'] * delta_e[0]
        ])

        return M0 + Momega + Mdelta
    
    
    def propeller_thrust_torque(self, delta_r, va_i, D, KQ, Rm, I0, Vmax, CT, CQ):
        CT0, CT1, CT2 = CT
        CQ0, CQ1, CQ2 = CQ
        rho = self.params['rho']
        # based on equations (21), (22) and (23)
        a = rho * D**5 / (4 * np.pi**2) * CQ0
        b = rho * D**4 / (2 * np.pi) * CQ1 * va_i + KQ**2 / Rm
        c = rho * D**3 * CQ2 * va_i**2 - (KQ / Rm) * (Vmax * delta_r) + KQ * I0

        omega = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        if np.isnan(omega) or np.isinf(omega):
            print("Invalid omega calculation")
        
        # based on equation (19)
        Tp = (rho * D**4 * CT0 / (4 * np.pi**2)) * omega**2 + \
            (rho * D**3 * CT1 * va_i / (2 * np.pi)) * omega + \
            (rho * D**2 * CT2 * va_i**2)

        # based on equation (20)
        Qp = (rho * D**5 * CQ0 / (4 * np.pi**2)) * omega**2 + \
            (rho * D**4 * CQ1 * va_i / (2 * np.pi)) * omega + \
            (rho * D**3 * CQ2 * va_i**2)

        return Tp, Qp

    def compute_propeller_forces_and_moments(self, va, delta_r, theta_r, q_rotors):
        """
        Compute forces and moments from propellers.
        va: velocity vector in body frame (3,) 
        delta_r: control inputs for rotors (3,)
        theta_r: tilt angles for front rotors (2,) and fixed rear rotor (1,)
        q_rotors: positions of rotors in body frame (3, 3)
        Returns: force vector (3,) and moment vector (3,) in body frame
        """
        # based on equations (24) and (25)
        
        forces = []
        torques = []

        for i in range(3):
            if i < 2:  # front rotors
                D = self.params['D_front']
                KQ = self.params['KQ_front']
                Rm = self.params['Rm_front']
                I0 = self.params['I0_front']
                Vmax = self.params['Vmax_front']
                CT = self.params['CT_front']
                CQ = self.params['CQ_front']
                theta = theta_r[i]
            else:  # rear rotor
                D = self.params['D_rear']
                KQ = self.params['KQ_rear']
                Rm = self.params['Rm_rear']
                I0 = self.params['I0_rear']
                Vmax = self.params['Vmax_rear']
                CT = self.params['CT_rear']
                CQ = self.params['CQ_rear']
                theta = np.pi / 2  # fixed rear rotor

            s = np.array([np.cos(theta), 0, -np.sin(theta)])
            vai = va @ s
            if abs(vai) < 1e-3:
                print(f"[Warning] va_i too small: {vai}, setting to 1e-3")
                vai = 1e-3
            delta_r = np.maximum(delta_r, 1e-6) # replace small delta_r with 1e-6 to avoid division by zero
            Tp, Qp = self.propeller_thrust_torque(delta_r[i], vai, D, KQ, Rm, I0, Vmax, CT, CQ)

            q = q_rotors[i]  # position of rotor in body frame
            forces.append(Tp * s)
            torques.append(-Qp * s + Tp * np.cross(q, s))

        Fp = np.sum(forces, axis=0)
        Mp = np.sum(torques, axis=0)
        return Fp, Mp
    
    
    def compute_total_force(self, va, omega, delta_e, delta_r, theta_r, q_rotors):
        Fa = self.compute_aerodynamic_forces(va, omega, delta_e)
        Fp, _ = self.compute_propeller_forces_and_moments(va, delta_r, theta_r, q_rotors)
        return Fa + Fp


    def compute_total_moment(self, va, omega, delta_e, delta_r, theta_r, q_rotors):
        Ma = self.compute_aerodynamic_moments(va, omega, delta_e)
        _, Mp = self.compute_propeller_forces_and_moments(va, delta_r, theta_r, q_rotors)
        return Ma + Mp


    def compute_omega_dot(self, omega, M):
        J = self.params['J']
        omega_cross = np.cross(omega, J @ omega)
        return np.linalg.inv(J) @ (M - omega_cross)


    def compute_theta_dot(self, theta, theta_cmd, ks):
        return ks * (theta_cmd - theta)
     
    
    def quaternion_derivative(self, quat, omega):
        """ 
        Compute the derivative of the quaternion based on angular velocity.
        quat: current quaternion (4,)  
        omega: angular velocity (3,)
        Returns: quaternion derivative (4,) 
        """
        q = quat
        wx, wy, wz = omega
        Omega = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
        return 0.5 * Omega @ q
    
    
    def derivatives(self, t, x_vec, u_func):
        """
        Compute the derivatives of the UAV state.
        t: current time
        x_vec: flattened state vector (15,)
        u_func: function to get control inputs at time t
        Returns: derivatives of the state vector (15,)
        """
        x = UAVStateQuat.from_vector(x_vec)
        u = u_func(t, x)
        
        norm_q = np.linalg.norm(x.quat)
        if norm_q > 1e-8:
            x.quat = x.quat / norm_q
        else:
            x.quat = np.array([1.0, 0.0, 0.0, 0.0])  # fallback

        # Assume a control input format: [delta_e1, delta_r1, delta_r2, delta_r3, theta_cmd1, theta_cmd2]
        # e.g. u = np.array([...]) with shape (6,)
        delta_e = np.array([u[0]])  # single deflection for both elevons
        delta_r = u[1:4]            # rotor commands
        theta_cmd = u[4:6]          # front servo angles
        theta_r = np.append(theta_cmd, np.pi/2)  # rear rotor tilt fixed at 90 deg

        # Assume constant rotor positions in body frame
        q_rotors = np.array([
            [0.2,  0.3, 0.0],  # front-left
            [0.2, -0.3, 0.0],  # front-right
            [-0.5, 0.0, 0.0]   # rear
        ])

        # Extract states
        R_ib = x.rotation_matrix()      # From body to inertial
        v = x.v                         # Body-frame linear velocity (used as va)
        omega = x.omega
        quat = x.quat
        theta = x.theta

        # Dynamics from Eq. (2)–(6)
        dp = R_ib @ v                                           # Eq. (2)
        g_vec = np.array([0, 0, -self.params['gravity']])
        g_b = R_ib.T @ g_vec # convert to body frame
        F = self.compute_total_force(v, omega, delta_e, delta_r, theta_r, q_rotors)
        dv = g_b + F / self.params['mass'] - np.cross(omega, v)  # Eq. (2)
        dquat = self.quaternion_derivative(quat, omega)         # Eq. (3)
        
        domega = self.compute_omega_dot(omega,
                    self.compute_total_moment(v, omega, delta_e, delta_r, theta_r, q_rotors))  # Eq. (4)
        dtheta = self.compute_theta_dot(theta, theta_cmd, ks=5.0)  # Eq. (6)

        return np.concatenate([dp, dv, dquat, domega, dtheta])
    
    
    
    def simplified_derivatives(self, x: UAVStateQuat, u):
        """
        Implements the simplified dynamics from Equation (42) in the paper.

        Args:
            x: UAVStateQuat state [p, v, quat, omega, theta]
            u: control input [a_x, a_z, ω_x, ω_y, ω_z]

        Returns:
            np.array: derivative of simplified state [dp, dv, dR (flattened)] of the dimension 15.
            When computing the error state and linearizing, this state has to be converted to a 9D vector
        """
        # Unpack state
        p = x.p
        v = x.v
        R = x.rotation_matrix()

        # Control inputs
        a_body = np.array([u[0], 0, u[1]])  # simplified body-frame acceleration
        omega = np.array(u[2:5])            # instantaneous angular velocities

        # Constants
        g = self.params['gravity']
        mass = self.params['mass']
        e3 = np.array([0.0, 0.0, 1.0])

        # Simplified dynamics (Equation 42 from the paper)
        dp = R @ v
        dv = np.cross(omega, v) + R.T @ (g * e3) + a_body
        dR = R @ self.skew(omega)

        # Flatten rotation matrix derivative
        dR_flat = dR.flatten()

        return np.concatenate([dp, dv, dR_flat])

    @staticmethod
    def skew(w):
        """ Returns skew-symmetric matrix for vector w """
        return np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ])