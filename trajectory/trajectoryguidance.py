"""
TrajectoryGuidance class to define, manage a block of ProMP plans within the same context.
Aiming at implementing guidance via points to target. 
"""

from trajectory.promp import *
import numpy as np
import os


class TrajectoryGuidance:
    def __init__(self, datafolder, objectlist):
        self.object_trajectory = {}
        for elem in objectlist:
            self.object_trajectory[elem] = self.initProMP(
                os.path.join(datafolder, elem+".npz"))

    def initProMP(self, file):
        """ 
        Create and learn a ProMP model for each trajectory.
        param file: the path to .npz file that stores all demonstrations of a plan
        """
        df_generated, N = PrepareData(file)
        params = {'D': 7, 'K': 5, 'N': N}
        trajectorymodel = ProMP(
            TrainingData=df_generated, params=params)
        trajectorymodel.RegularizedLeastSquares()
        return trajectorymodel

    def get_trajectory_to_target(self, viapoint, target, phase=0):
        """ 
        Find trajectory based on ProMP via point conditioning 
        param viapoint: via point as conditions
        param target: the target object (which is in the list of trajectories)
        """
        conditioned_trajectory = self.object_trajectory[target].Condition_JointSpace(
            Qtarget=[viapoint, self.object_trajectory[target].GetTrajectory()[-1]], Ztarget=[phase*0.1, 1])
        return conditioned_trajectory[phase:]
