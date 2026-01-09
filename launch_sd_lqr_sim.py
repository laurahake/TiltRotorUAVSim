"""
vtolsim_lqr
    - simultion showing trajectory following using full state LQR
    - Update history:
        5/8/2019 - R.W. Beard
        2/1/2024 - RWB
        3/12/2024 - RWB
"""
import os, sys
# insert parent directory at beginning of python search path
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[1]))
import numpy as np
from scipy.spatial import cKDTree
import parameters.simulation_parameters as SIM
import parameters.spline_parameters as SPLP
from message_types.msg_convert import *
from models.vtol_dynamics import VtolDynamics
from controllers.lqr.lqr_control import LqrControl
from controllers.low_level_control import LowLevelControl
from planners.trajectory.spline_trajectory import SplineTrajectory
from planners.trajectory.differential_flatness import DifferentialFlatness
from message_types.msg_delta import MsgDelta
from viewers.view_manager import ViewManager
from tools import rotations 

from datetime import datetime

SAVE_DATA = True
SAVE_DIR = Path(__file__).resolve().parent / "data_rollouts"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
STATE_DB_PATH = SAVE_DIR / f"state_db_{RUN_TAG}.npy"
TILT_INDEX_PATH = SAVE_DIR / f"tilt_index_1d_{RUN_TAG}.npz"

IDX_PD = 2
IDX_TILT_R = 13
IDX_TILT_L = 14

# initialize elements of the architecture
wind = np.array([[0., 0., 0., 0., 0., 0.]]).T
vtol = VtolDynamics(SIM.ts_simulation)
viewers = ViewManager(animation=True, data=True)

# initialize trajectory
# df_traj = DifferentialFlatness(XYZSinusoid(150., 150., 75., 600., 300., 600., -np.pi/2, np.pi, 0.*np.pi/2))
# df_traj = DifferentialFlatness(HANDTraj())
df_traj = DifferentialFlatness(SplineTrajectory(SPLP.pts, SPLP.vels)) # points, max velocity
step =  0.*df_traj.traj.s_max/500
viewers.vtol_view.addTrajectory(df_traj.traj.getPList(df_traj.traj.getP, 0., df_traj.traj.s_max + step, df_traj.traj.s_max/500))

#initialize controllers
low_ctrl = LowLevelControl(M=0.5, Va0=2.0, ts=SIM.ts_simulation)
lqr_ctrl = LqrControl(SIM.ts_simulation)

# initialize the simulation time
sim_time = SIM.start_time

# ---- buffers for logged data ----
state_log = []   # list of np arrays (15,)
time_log  = [] 

# main simulation loop
print("Press Command-Q to exit...")
while sim_time < SIM.end_time:
    #-------observer-------------
    measurements = vtol.sensors()  # get sensor measurements
    estimated_state = vtol._state  # estimated state is current state
    
    # ------ Trajectory follower
    desired_state, desired_input = df_traj.desiredState_fromX(estimated_state[0:10])
    
    #------- High Level controller-------------
    force_des, omega_des = lqr_ctrl.update(
        estimated_state[0:10], 
        desired_state, 
        desired_input, 
        df_traj)
    
    #------- Low Level Controller -------------
    delta = low_ctrl.update(force_des, omega_des, vtol.true_state)
    
    #-------update physical system-------------
    vtol.update(delta, wind)  # propagate the MAV dynamics
    
    # ---- log states for dataset ----
    if SAVE_DATA:
        msg = vtol.true_state
        pos  = np.asarray(msg.pos).reshape(-1)              # (3,)
        vel  = np.asarray(msg.vel).reshape(-1)              # (3,)
        quat = rotations.rotation_to_quaternion(msg.R).reshape(-1)  # (4,)
        omg  = np.asarray(msg.omega).reshape(-1)            # (3,)
        tilt = np.asarray(msg.motor_angle).reshape(-1)      # (2,)

        x = np.concatenate([pos, vel, quat, omg, tilt]).astype(np.float32)
        state_log.append(x)
        time_log.append(sim_time)

    #-------update viewers-------------
    viewers.update(
        sim_time,
        vtol.true_state,  # true states
        vtol.true_state,  # estimated states
        vtol.true_state,  # commanded states
        delta,  # inputs to aircraft
        None,  # measurements
    )
    
    #-------increment time-------------
    sim_time += SIM.ts_simulation
    
# ---- save dataset + 1D tilt index ----
X_db = np.stack(state_log, axis=0).astype(np.float32)  # (N,15)
np.save(STATE_DB_PATH, X_db)
print(f"[DATA] Saved state DB: {STATE_DB_PATH} with shape {X_db.shape}")

# --- 2D features: height + tilt ---
h = (-X_db[:, IDX_PD]).astype(np.float32)  # height in meters
tilt = (0.5 * (X_db[:, IDX_TILT_R] + X_db[:, IDX_TILT_L])).astype(np.float32)

feats = np.stack([h, tilt], axis=1)  # (N,2)

# normalize features
mu = feats.mean(axis=0)
sig = feats.std(axis=0) + 1e-8
feats_n = (feats - mu) / sig

# build KD-tree
tree = cKDTree(feats_n)

# save everything needed to reuse later
np.savez(TILT_INDEX_PATH,
            feats_n=feats_n.astype(np.float32),
            mu=mu.astype(np.float32),
            sig=sig.astype(np.float32))
print(f"[DATA] Saved 2D (height,tilt) index: {TILT_INDEX_PATH} (N={feats_n.shape[0]})")   
input("Press a key to exit")
