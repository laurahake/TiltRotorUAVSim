# TiltRotorUAVSim

This repository contains a physics-based simulation of a tilt-rotor UAV, based entirely on the nonlinear dynamic model described in the paper:

> Eren, U., Hamer, M., Graichen, K., Zierer, B., Faessler, M., & Scaramuzza, D. (2017). *State-Dependent LQR Control for a Tilt-Rotor UAV*. In 2017 IEEE International Conference on Robotics and Automation (ICRA), pp. 1236–1241.

The simulation numerically solves the full nonlinear state equations using SciPy's `solve_ivp` integrator. It includes models for translational and rotational dynamics, gravity, aerodynamic forces, propeller thrust, control surfaces, and servo tilt angles.

---

## 🚁 Features

- Full 20-state nonlinear dynamics (position, velocity, rotation, angular velocity, rotor tilt)
- Gravity and Coriolis effects in the body frame
- Servo tilt dynamics for front rotors
- Modular UAV model class
- Custom control inputs (e.g. hover control)
- Multi-axis state plots

---

## 📁 Repository Structure

- TiltRotorUAV.py # Class implementation of the UAV model
- simulation.py # Main simulation script using solve_ivp
- README.md # You're here

---

## ▶️ How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/laurahake/TiltRotorUAVSim.git
    cd TiltRotorUAVSim
    ```

2. Install dependencies:
    ```bash
    pip install numpy scipy matplotlib
    ```

3. Run the simulation:
    ```bash
    python simulation.py
    ```

4. A plot will display the evolution of the UAV's state over time (position, velocity, orientation, etc.)

---

## 📚 Model Reference

All physics, parameters, and modeling equations are derived from:

> Eren et al., *State-Dependent LQR Control for a Tilt-Rotor UAV*, ICRA 2017.  
> [IEEE Link](https://ieeexplore.ieee.org/document/7989145)

---

## 🤖 Acknowledgements

Certain portions of the code and documentation were developed with the assistance of [ChatGPT](https://openai.com/chatgpt), based on the referenced paper.

---

## 📜 License

MIT License — feel free to use, modify, and share with attribution.