"""
Robot Arm Environment for reinforcement learning model.
"""

import numpy as np
from trajectory import TrajectoryGuidance
import math


class ArmEnv:

    def __init__(self, jacobian, trajectories, target_list):
        self.state_dim = 3
        self.action_number = 6

        self.J = jacobian
        self.trajectory_plans = trajectories
        self.target_list = target_list
        self.state = np.zeros((3,))
        self.actions = {0: [-1, 0, 0], 1: [1, 0, 0], 2: [0, -
                                                         1, 0], 3: [0, 1, 0], 4: [0, 0, -1], 5: [0, 0, 1]}

    def step(self, action, current_env=[]):
        """
        Generate the next state of agent.
        param action: action for agent to take
        return: states if the agent takes action, which includes expected state, reward value, done state
                and a list containing target and action if it reaches the target
        """
        done = False

        dt = 0.05
        s = self.state+np.asarray(self.actions[action])*dt
        bestchoice = {"target": None, "phase": None, "reward": None}

        for target in self.target_list:
            if not (target in current_env):
                for phase_num in range(10):
                    q_lower, q_upper = self.trajectory_plans.object_trajectory[target].GetMeanRange(
                        phase_num)
                    x_lower = self.J.dot(q_lower)
                    x_upper = self.J.dot(q_upper)
                    inside = True
                    for i in range(3):
                        if x_lower[i] < x_upper[i]:
                            if s[i] < x_lower[i] or s[i] > x_upper[i]:
                                inside = False
                                break
                        else:
                            if s[i] > x_lower[i] or s[i] < x_upper[i]:
                                inside = False
                                break

                    # If the agent new state is inside the range of one trajectory
                    if inside:
                        done = True
                        des_q = self.trajectory_plans.object_trajectory[target].GetTrajectory()[
                            phase_num]
                        des_x = self.J.dot(des_q)
                        distance = math.sqrt((des_x[0]-s[0])**2+(des_x[1] -
                                                                 s[1])**2+(des_x[2]-s[2])**2)

                        bestchoice["reward"] = -distance
                        bestchoice["target"] = target
                        bestchoice["phase"] = phase_num
                        break
            if done:
                break

        if done:
            r = bestchoice["reward"]
        else:
            r = 0
        self.state = s
        return s, r, done, [bestchoice["target"], bestchoice["phase"]]

    def reset(self):
        """
        Reset the agent position in the environment
        return: random position within specified workspace
        """
        self.state[0] = np.random.uniform(-1, -0.4)
        self.state[1] = np.random.uniform(-0.2, 0.1)
        self.state[2] = np.random.uniform(-1, -0.3)
        return self.state

    def setstate(self, position):
        """
        Set new position state for the robot's position
        """
        self.state = position
        return self.state
