from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from TiltRotorUAV import TiltRotorUAV

def simulate(uav, u_func, t_span=(0, 10), dt=0.01):
    x0 = uav.state.as_vector()
    t_eval = np.arange(t_span[0], t_span[1], dt)

    def rhs(t, x): return uav.derivatives(t, x, u_func)

    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval, rtol=1e-6, atol=1e-8)
    return sol


def hover_control(t, x):
    # This control should match the inputs expected by TiltRotorUAV.derivatives
    # which are: [ax, az, wx, wy, wz]
    return np.array([0, 9.81, 0, 0, 0])


if __name__ == "__main__":
    uav = TiltRotorUAV()
    sol = simulate(uav, hover_control)
    t = sol.t
    y = sol.y

    fig, axs = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

    axs[0].plot(t, y[0], label='x')
    axs[0].plot(t, y[1], label='y')
    axs[0].plot(t, y[2], label='z')
    axs[0].set_ylabel("Position [m]")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(t, y[3], label='vx')
    axs[1].plot(t, y[4], label='vy')
    axs[1].plot(t, y[5], label='vz')
    axs[1].set_ylabel("Velocity [m/s]")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(t, y[6], label='qx')
    axs[2].plot(t, y[7], label='qy')
    axs[2].plot(t, y[8], label='qz')
    axs[2].plot(t, y[9], label='qw')
    axs[2].set_ylabel("Quaternion")
    axs[2].legend()
    axs[2].grid()

    axs[3].plot(t, y[10], label='wx')
    axs[3].plot(t, y[11], label='wy')
    axs[3].plot(t, y[12], label='wz')
    axs[3].set_ylabel("Angular Velocity [rad/s]")
    axs[3].legend()
    axs[3].grid()

    axs[4].plot(t, y[13], label='theta1')
    axs[4].plot(t, y[14], label='theta2')
    axs[4].set_ylabel("Servo Angles [rad]")
    axs[4].set_xlabel("Time [s]")
    axs[4].legend()
    axs[4].grid()

    plt.suptitle("Tilt-Rotor UAV Full State Simulation")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

