"""
Intent detection class for recalling intent detection model in thread.
Loop through DQN model and return the end point with target, phase and the number of steps without affecting the main loop.

"""

import time
import threading
import numpy as np
from collections import defaultdict
import pickle
import time
from intent_detection import *


class IntentDetection(threading.Thread):
    def __init__(self, RL, rl_env):
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._active = True
        self.RL = RL
        self.rl_env = rl_env
        self.ref_position = None
        self.detected = False
        self.result = None
        self.number_of_steps = 0

    def start_stream(self):
        self._active = True
        self.start()

    def stop_stream(self):
        self._active = False

    def run(self):
        while self._active:
            self.detect()

    def detect(self):
        """
        Loop through a sequence of actions and states to detect the intended goal at the final step.
        """
        if not(self.ref_position is None):
            observation = self.rl_env.setstate(
                self.ref_position)
            inloop = 0
            while True:
                action = self.RL.choose_action(observation)
                # RL take action and get next observation and reward
                observation_, reward, done, info = self.rl_env.step(
                    action, self.current_env)
                # swap observation
                observation = observation_
                if done:
                    self.detected = True
                    self.result = info
                    self.number_of_steps = inloop
                    break
                if inloop == 5:
                    break
                inloop += 1

    def update(self, position, current_env):
        """
        Update knowledge of the robot state
        param position: robot's end-effector position
        param current_env: list of finished tasks
        """
        self.ref_position = position
        self.current_env = current_env
        self.detected = False

    def isDetected(self):
        """
        return: the intent detection result that includes target, phase and the number of steps
        """
        return self.detected, self.result, self.number_of_steps
