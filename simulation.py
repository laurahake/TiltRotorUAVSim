import numpy as np
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from TiltRotorUAV import TiltRotorUAV, UAVStateQuat
from SD_LQR import StateDependentLQRController
from trajectory_generator import TrajectoryGenerator
from trajectory_tracker import TrajectoryTracker
from low_level_controller import LowLevelController

# 1. Setup
dt = 0.05
T_final = 0.5
timesteps = int(T_final / dt)
#t_vals = np.linspace(0, T_final, timesteps)
t_vals = np.linspace(0, T_final, int(T_final / 0.1))  # evaluate every 0.1s


uav = TiltRotorUAV()
sd_lqr = StateDependentLQRController(uav)
low_level = LowLevelController(
    pid_params={'x': (2.0, 0.0, 0.2), 'y': (2.0, 0.0, 0.2), 'z': (2.0, 0.0, 0.2)},
    mix_params={'N': 1.0, 'Va0': 4.0}
)

# Define trajectory
# move 10 meters foward
# then turn right 90 degrees and move 10 meters
# then turn 90 degrees right again
knots = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]])
speeds = np.array([3, 3, 3, 3]) # const
traj_gen = TrajectoryGenerator(knots, speeds)
tracker = TrajectoryTracker(traj_gen)

# Initial state
x = UAVStateQuat()
state_history = []
time_history = []

# 2. Simulation loop
def control_func(t, x_vec):
    if isinstance(x_vec, UAVStateQuat):
        x_vec = x_vec.as_vector()
    x_obj = UAVStateQuat.from_vector(x_vec)
    # get closest point on the path to the current UAV position
    s = tracker.find_nearest_s(x_obj.p)
    # get desired position, direction and speed to fly , and orientation to face along the path
    p_des, v_des, quat_des = traj_gen.get_desired_state(s)

    # Form desired state
    desired_state = UAVStateQuat(
        p=p_des,
        v=v_des,
        quat=quat_des,
        omega=np.zeros(3),
        theta=np.zeros(2)
    )

    # High-level control
    # u_c consists of [ax, az, wx, wy, wz]
    # where ax is acceleration in x(throttle), az is acceleration in z(vertical thrust), and w how fast to rotate around each axis
    u_c = sd_lqr.compute_control(desired_state, x_obj)

    # Current angular velocity
    omega = x_obj.omega
    # Estimate airspeed (just norm of body-frame velocity)
    Va = np.linalg.norm(x_obj.v)

    # Low-level control: actuator commands
    actuators = low_level.compute(u_c, omega, Va, dt)

    # Final actuator vector
    delta_e = actuators['delta_e']  # elevon deflections
    delta_r = actuators['delta_r']  # rotor thrusts
    theta_r = actuators['theta_r']  # rotor tilts
    return np.array([*delta_e, *delta_r, *theta_r])

# 3. Integrate dynamics
def dynamics(t, x_vec):
    return uav.derivatives(t, x_vec, control_func)

x_vec0 = x.as_vector()
#sol = solve_ivp(dynamics, [0, T_final], x_vec0, t_eval=t_vals, method='RK45', max_step=dt)
sol = solve_ivp(
    dynamics,
    [0, T_final],
    x_vec0,
    t_eval=t_vals,
    method='RK23',
    max_step=dt,
    rtol=1e-2, atol=1e-4    # loosen tolerances
)

actual_positions = sol.y[:3].T  # Extract actual positions from solution
desired_positions = []

for pos in actual_positions:
    # Find the closest s for the actual position
    s = tracker.find_nearest_s(pos, full_search=True)
    # Get the desired position at this s
    p_des, _, _ = traj_gen.get_desired_state(s)
    desired_positions.append(p_des)

# 4. Process results
states = [UAVStateQuat.from_vector(xi) for xi in sol.y.T]

# 5. Plot results
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(actual_positions[:, 0], actual_positions[:, 1], actual_positions[:, 2],
        label="Actual Trajectory", linestyle='--', linewidth=2)
ax.plot(desired_positions[:, 0], desired_positions[:, 1], desired_positions[:, 2],
        label="Desired Trajectory", linewidth=2)

ax.set_title("3D Trajectory: Actual vs. Desired")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
