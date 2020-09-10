"""
ProMP class to construct trajectory model using training data which contains all demonstrations.
"""

import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import pickle as p
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize, NonlinearConstraint, BFGS
import math


class ProMP:
    def __init__(self, TrainingData=None, params=None):
        self.D = params['D']  # Number of degrees of freedom
        # Number of basis functions (should be > 2, since it uses 1 polynomial function and >1)
        self.K = params['K']
        self.N = params['N']  # Number of Demonstrations currently attached.
        self.TrainingData = TrainingData
        # W is vector of dimension K*D. We initialize it here.
        self.meanw = np.zeros(self.K*self.D)
        # This dataframe stores the already-trained w vectors.
        self.W = pd.DataFrame(columns=['W'])
        self.examplestrained = 0  # This accounts to the number of rows of this dataframe
        self.estimate_m = None
        self.estimate_sd = None

    def GBasis(self, z, c, h):
        """"
        Returns: gaussian basis.
        """

        return np.exp(-((z-c)**2)/(2*h))

    def BasisVector(self, z):
        """
        return: the K x 1 basis vector for the ProMP.
        """
        K = self.K
        BaVe = np.zeros(K)
        # the width of the basis functions is selected to consistently divide the entire phase interval
        h = 0.1*1/(K-1)
        # The interval between the first and the last basis functions.
        Interval = [-2*h, 1+2*h]
        # Effective distance between each basis
        dist_int = (Interval[1]+abs(Interval[0]))/(K-1)
        c = Interval[0]
        for k in range(K):
            BaVe[k] = self.GBasis(z, c, h)
            c = c+dist_int
        return BaVe/np.sum(BaVe)

    def BasisMatrix(self, z):
        """
        Generates the basis matrix for the joints at the phase level z.
        param z: the basis vectors for each degree of freedom (For simplicity, it is the same for all DoFs)
        return: DxKD matrix at phase Z=z 
        """
        D = self.D
        K = self.K
        BaMa = np.zeros((D, K*D))

        index_block = 0  # On each row of the basis matrix, the Basis Vector should start on a different index. The first one starts at 0
        for d in range(D):
            # Computes the basis vector, which is assumed to be the same for each joint
            BaMa[d, index_block:index_block+K] = self.BasisVector(z)
            index_block = index_block+K

        return BaMa

    def BigMatrix(self, demonstration):
        """
        Generates the basis matrics 'PSI' for all the joints over all the phase stamps.
        param demonstration: a sample of training data
        return: a DZxKD block matrix that stacks in the diagonal
        """
        Z = demonstration["Times"].shape[0]
        D = self.D
        K = self.K
        BFM = np.zeros((Z*D, K*D))
        index_phase = 0  # On each row within the same DoF, the Basis Vector should start on a different index. The first one starts at 0
        index_block = 0  # On each DoF of the basis matrix, the Basis Vector should start on a different index. The first one starts at 0

        for d in range(D):
            for idz, z in enumerate(demonstration["Times"]):
                # Computes the basis vector, which is assumed to be the same for each joint
                BFM[index_phase, index_block:index_block +
                    K] = self.BasisVector(z)
                index_phase = index_phase+1
            index_block = index_block+K

        return BFM

    def RegularizedLeastSquares(self, l=1e-12):
        """
        This uses the regularized least squares method to train the ProMP.
        param l: is the regularization parameter.
        """
        D = self.D
        K = self.K

        Qlist = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']  # list of q strings
        wmatrix = np.array([])
        df = self.TrainingData
        for index, row in df.iterrows():
            PSI = self.BigMatrix(row)
            Y = row[Qlist[0]]
            for q in range(1, D):
                Y = np.hstack((Y, row[Qlist[q]]))
            ######
            RegMat = l*np.eye(K*D)

            factor1 = npl.inv((np.matmul(PSI.T, PSI)+RegMat))
            factor2 = np.dot(PSI.T, Y)
            w = np.dot(factor1, factor2)
            self.examplestrained = self.examplestrained+1

            # The exemplified w's are stored as a dataframe. But first, they are
            # stacked on a matrix such that we can compute easily the covariance.
            if index == 0:
                wmatrix = w
            else:
                wmatrix = np.vstack((wmatrix, w))

            self.W = self.W.append({'W': w}, ignore_index=True)

        self.estimate_m = np.mean(self.W.values)
        self.estimate_sd = np.std(self.W.values)
        self.estimate_sigma = np.cov(wmatrix.T)

    def Condition_JointSpace(self, Qtarget, Ztarget):
        """Qtarget and Ztarget contain one or more target points.

        This script conditions the distribution and generates a nice plot over it.
        returns only the mean.
        T"""
        K = self.K
        D = self.D
        N = self.N
        Ey_des = 0.001*np.eye(D)  # Accuracy matrix
        Ew = self.estimate_sigma
        Muw = self.estimate_m

        for idxq, q in enumerate(Qtarget):
            z = Ztarget[idxq]

            Psi = self.BasisMatrix(z).T

            L = Ew.dot(Psi).dot(npl.inv(Ey_des+Psi.T.dot(Ew).dot(Psi)))
            Muw = Muw+L.dot(q.T-Psi.T.dot(Muw))
            Ew = Ew-L.dot(Psi.T.dot(Ew))

        path = self.GetTrajectory(
            w=Muw)
        return path

    def GetTrajectory(self, w=None):
        ''' Predicts the mean trajectory for a given w.
        In case no w is given, the mean from all the demonstrations is plotted'''
        if w is None:
            w = self.estimate_m
        Z = np.arange(0, 1, 0.1)
        Y = np.array([])
        for z in Z:
            if z == 0:
                Y = np.dot(self.BasisMatrix(z), w)
            else:
                Y = np.vstack(
                    (Y, np.dot(self.BasisMatrix(z), w)))
        return Y

    def GetMeanRange(self, phase, dt=0.1, factor=1):
        """
        param phase: the phase to find range (phase should be in [0,10])
        return: the range of learnt trajectories
        """
        Ylower = np.dot(self.BasisMatrix(
            phase*dt), self.estimate_m-factor*self.estimate_sd)
        Yupper = np.dot(self.BasisMatrix(
            phase*dt), self.estimate_m+factor*self.estimate_sd)
        return Ylower, Yupper

    def PlotBasisFunctions(self):
        """
        Plots the basis functions that are currently used
        """
        plt.figure()
        q = np.arange(-0.1, 1.1, 0.005)
        for Q in q:

            for idx, basis in enumerate(self.BasisVector(Q)):
                plt.plot(Q, basis, 'b*')
                plt.title('Basis Functions')

    def ConditionedAndStdPredictionPlot(self, wconditioned, factor=1, Qtarget=None, Ztarget=None):
        """
        Produces a plot of a conditioned trajectory.
        param wconditioned: the weight vector calculated from conditions
        """
        if Qtarget is not None and Ztarget is not None:
            PlotViaPoints = True
        else:
            PlotViaPoints = False

        Z = np.arange(0, 1, 0.1)
        Qlist = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']  # list of q strings
        Y = np.array([])
        Ycond = np.array([])

        for z in Z:
            if z == 0:
                Y = np.dot(self.BasisMatrix(z), self.estimate_m)
                Ycond = np.dot(self.BasisMatrix(z), wconditioned)

            else:
                Y = np.vstack(
                    (Y, np.dot(self.BasisMatrix(z), self.estimate_m)))
                Ycond = np.vstack(
                    (Ycond, np.dot(self.BasisMatrix(z), wconditioned)))

        Ylower = np.array([])
        for z in Z:
            if z == 0:
                Ylower = np.dot(self.BasisMatrix(
                    z), self.estimate_m-factor*self.estimate_sd)
            else:
                Ylower = np.vstack((Ylower, np.dot(self.BasisMatrix(
                    z), self.estimate_m-factor*self.estimate_sd)))
        Yupper = np.array([])
        for z in Z:
            if z == 0:
                Yupper = np.dot(self.BasisMatrix(
                    z), self.estimate_m+factor*self.estimate_sd)
            else:
                Yupper = np.vstack((Yupper, np.dot(self.BasisMatrix(
                    z), self.estimate_m+factor*self.estimate_sd)))

        plt.figure()
        for idq, q in enumerate(Qlist):
            plt.subplot(1, 7, idq+1)

            plt.fill_between(Z, Yupper[:, idq], Ylower[:, idq], alpha=0.6)
            plt.plot(Z, Y[:, idq], 'k', LineWidth=2)
            plt.plot(Z, Ycond[:, idq], 'r', LineWidth=2)
            plt.title(Qlist[idq])
            if idq > 0:
                plt.yticks([])
            else:
                plt.yticks([-np.pi, 0, np.pi],
                           ['$-\pi$', '$0$', '$\pi$'])
            plt.xticks([0, 1])
            if idq == 6:

                plt.legend(
                    ('Mean trajectory', 'Mean trajectory after conditioning to via points'), loc=4)

            if PlotViaPoints:
                for idx, qtarg in enumerate(Qtarget):
                    plt.plot(Ztarget[idx], qtarg[idq], 'r*')
            plt.ylim(-np.pi, np.pi)
        plt.suptitle('Conditioned trajectory')
        plt.show()


def PrepareData(filename):
    """
    Loads the dataframe, which is in a .npz file, and puts it on the necessary shape to get in the ProMP class.
    """

    columns = ['Times', 'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']
    df_generated = pd.DataFrame(columns=columns)  # Empty dataframe

    # # load dict of arrays
    dict_data = np.load(filename)
    # extract the first array
    ds = dict_data['arr_0']
    dt = 1/(len(ds[0])-1)
    duration = 1.0
    duration += dt  # to get time points to line up correctly
    t = np.arange(0.0, duration, dt)
    N = len(ds)
    if t[len(t)-1] > 1:
        t = t[:-1]
    t = np.tile(t, (N, 1))
    for i in range(N):
        q = ds[i]
        timestamps = t[i]
        df_generated = df_generated.append(pd.DataFrame(data={'Times': [timestamps],
                                                              'q0': [q[:, 0]],
                                                              'q1': [q[:, 1]],
                                                              'q2': [q[:, 2]],
                                                              'q3': [q[:, 3]],
                                                              'q4': [q[:, 4]],
                                                              'q5': [q[:, 5]],
                                                              'q6': [q[:, 6]]}), ignore_index=True)

    return df_generated, N
