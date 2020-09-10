""" 
Input class to connect, receive and transmit data to teleoperator.

"""

import time
import threading
import numpy as np
from collections import defaultdict
import socket
# import datetime
import pickle

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432
timeStep = 1. / 100.


class BaseInputController(threading.Thread):
    """ Base input controller class that runs a input device controller daemon thread."""

    def __init__(self):
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._active = True

    def run(self):
        while self._active:
            self._input_to_command()

    def _input_to_command(self):
        raise NotImplementedError

    def get_commands(self):
        raise NotImplementedError

    def start_controller(self):
        self._active = True
        self.start()

    def stop_controller(self):
        self._active = False
        # self.join()


class InputControllerViaSocket(BaseInputController):
    """ Input controller using TCP socket """

    def __init__(self):
        super().__init__()
        self.force_feedback = np.zeros((3,))
        self.desired_cart_pos = np.zeros((3,))
        self.desired_ori = np.zeros((3,))
        self.desired_gripper = 0

        # Initialize connection
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((HOST, PORT))
        self.s.listen()
        self.conn, self.addr = self.s.accept()
        if self.conn:
            print('Connected by', self.addr)

    def get_initial_states(self, defaultValue=True):
        return self.desired_cart_pos, self.desired_ori

    def _input_to_command(self):
        self.ReceiveandTransmitData()

    def ReceiveandTransmitData(self):
        """
        Data transmission between the master and the slave.
        Receive haptic position, rotation and gripper states.
        Send force feedback to haptic device.
        """
        try:
            # Receive stylus position
            data = self.conn.recv(2048)
            received = pickle.loads(data)
            recv_pos = received["Position"].split()
            recv_ori = received["Rotation"].split()
            recv_button1 = received["Button1"]
            recv_button2 = received["Button2"]

            # Send force feedback
            data_to_send = ""
            for i in self.force_feedback:
                data_to_send += str(i)+" "
            self.conn.sendall(data_to_send.encode('utf-8'))

            # Scale and convert coordinates
            self.desired_cart_pos[0] = -float(recv_pos[0])*2.5
            self.desired_cart_pos[1] = (float(recv_pos[2])+0.25)*2
            self.desired_cart_pos[2] = (float(recv_pos[1])+0.3)*3

            self.desired_ori[0] = (float(recv_ori[2])+6.65)*0.5
            self.desired_ori[1] = ((float(recv_ori[0]))+1)*0.6
            self.desired_ori[2] = (-(float(recv_ori[1]))+2.941)*0.5

            self.desired_gripper = float(
                recv_button1)-float(recv_button2)

        except:
            print("Cannot connect to haptic device!")
            self.stop_controller()

    def get_commands(self):
        """
        return: command signals including desired Cartesian position, orientation and gripper.
        """
        return self.desired_cart_pos, self.desired_ori, self.desired_gripper

    def send_force_feedback(self, data):
        """
        Get force feedback from the robot arm.
        param data: force value
        """
        # haptic has different coordinates order
        convert = [-data[1], -data[2], -data[0]]
        self.force_feedback = convert
