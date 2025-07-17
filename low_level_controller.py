import numpy as np
from pid import PID

class LowLevelController:
    def __init__(self, pid_params, mix_params):
        self.pids = {
            'x': PID(*pid_params['x']),
            'y': PID(*pid_params['y']),
            'z': PID(*pid_params['z'])
        }
        self.N = mix_params['N']
        self.Va0 = mix_params['Va0']

    def compute(self, u_c, omega, Va, dt):
        ax, az = u_c[0], u_c[1]
        omega_c = u_c[2:]

        # PID rate control
        # based on equation (45)
        tau = np.zeros(3)
        for i, axis in enumerate(['x', 'y', 'z']):
            tau[i] = self.pids[axis].update(omega_c[i] - omega[i], dt)

        # Build W = [ax, az, tau_x, tau_y, tau_z]
        W = np.array([ax, az, *tau])

        # Airspeed-based mixing based on equation (46)-(48)
        rho = 1.0 / (1.0 + np.exp(-self.N * (Va - self.Va0)))
        G = self.build_mixing_matrix(rho)
        Lambda = G @ W

        # Map Lambda to actuator commands
        delta_r1 = np.hypot(Lambda[0], Lambda[1])
        theta_r1 = np.arctan2(Lambda[1], Lambda[0])
        delta_r2 = np.hypot(Lambda[2], Lambda[3])
        theta_r2 = np.arctan2(Lambda[3], Lambda[2])
        delta_r3 = Lambda[4]
        delta_e1 = Lambda[5]
        delta_e2 = Lambda[6]

        return {
            'delta_e': [delta_e1, delta_e2],
            'delta_r': [delta_r1, delta_r2, delta_r3],
            'theta_r': [theta_r1, theta_r2]
        }

    def build_mixing_matrix(self, rho):
        return np.array([
            [0, 1, 0, -(1 - rho), 0],
            [1, 0, 0, 0, -1],
            [0, 1, -(1 - rho), 1 - rho, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1 - rho, 1 - rho, 0],
            [0, 0, -rho, -rho, 0],
            [0, 0, rho, -rho, 0],
        ])
