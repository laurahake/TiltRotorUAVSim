# TiltRotorUAVSim

This repository is a streamlined version of the [BYU MAGICC vtolsim](https://github.com/byu-magicc/vtolsim) simulator, adapted to demonstrate **State-Dependent LQR Control** for a Tilt-Rotor UAV based on the paper:

> **"State-Dependent LQR Control for a Tilt-Rotor UAV"**  
> Eren, Serdar; Ehsani, Amirhossein; Beard, Randal W.  
> _IEEE Control Systems Letters, 2020_  
> [Link to publication](https://ieeexplore.ieee.org/document/9147931)

## ✈️ Overview

The goal of this simulation is to reproduce the trajectory tracking performance of a tilt-rotor UAV using the **state-dependent LQR (SD-LQR)** strategy. The model, dynamics, and control structure follow the equations and architecture laid out in the referenced publication.

## 📁 Repository Structure

- `launch_sd_lqr_sim.py` — Main entry point for running the SD-LQR trajectory tracking simulation
- `models/` — Implements the tilt-rotor UAV dynamics
- `controllers/` — Includes SD-LQR and low-level control implementations
- `planners/` — Trajectory generation using splines and differential flatness
- `viewers/` — Real-time visualization tools
- `parameters/` — Configuration files for UAV and simulation parameters
- `message_types/` — Standardized message classes for interfacing system components
- `tools/` — Utility functions (e.g., math and interpolation helpers)

## ▶️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/laurahake/TiltRotorUAVSim.git
cd TiltRotorUAVSim
```

### 2. Install Dependencies

This project uses Python 3. Install the required packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not included, you may need the following key libraries manually:
- `numpy`
- `scipy`
- `pyqtgraph` (for visualization)
- `PyQt5`

### 3. Run the Simulation

```bash
python launch_sd_lqr_sim.py
```

A real-time viewer will display the UAV trajectory and relevant plots.

## 📖 Reference

If you use this codebase in your own work or research, please cite the original paper:

```bibtex
@article{eren2020sd_lqr,
  author={Eren, Serdar and Ehsani, Amirhossein and Beard, Randal W.},
  journal={IEEE Control Systems Letters}, 
  title={State-Dependent LQR Control for a Tilt-Rotor UAV}, 
  year={2020},
  volume={5},
  number={1},
  pages={79--84},
  doi={10.1109/LCSYS.2020.3004504}
}
```

## 📌 Notes

- This repository contains only the essential components for simulating SD-LQR as described in the paper.
- It **includes actuator dynamics and quaternion-based orientation**, as in the original repo. These components are essential for realistic modeling and are preserved in this version.

---

📝 _This README was generated with the help of ChatGPT._
