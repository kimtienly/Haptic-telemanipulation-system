# Haptic telemanipulation

This repository contains the implementation of a haptic telemanipulation system. A Touch haptic device controls a simulated Franka Emika panda robot arm remotely via a TCP socket.

Specifically, the operations includes:

1. Position-force end-effector control.
2. Intent detection using Deep Q-Learning.
3. Trajectories modelling using Probabilistic Movement Primitives (ProMP).

## Requirements

- This implement is used with Python 3. An environment requirements.txt file is provided.
- Touch haptic device (Ethernet version) with its driver.
- H3DAPI

## Usage

Run main.py in the required environment:

```sh
python main.py.
```

Then, when the haptic device is connected to the system, run this in command line to initialize device:

```sh
h3dload haptic/haptic.x3d
```
