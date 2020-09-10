import socket
from H3DInterface import *
from H3DUtils import *
import random
import time as python_time
import datetime
import pickle
random.seed(python_time.time())

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

x, = references.getValue()  # Group node on x3d


class DeviceChangeField(AutoUpdate(TypedField(SFBool, (SFBool, SFBool, SFVec3f, SFRotation)))):
    def __init__(self):
        AutoUpdate(TypedField(SFBool, (SFBool, SFBool, SFVec3f, SFRotation))
                   ).__init__(self)

        self.node, self.dn = createX3DNodeFromString("""\
            <ForceField DEF="FORCE"/>""")
        x.children.push_back(
            self.node)  # make the node visible by adding it to the parent
        self.force = [0, 0, 0]

    def update(self, event):
        """
        Update states changes of haptic stylus's position, orientation and 2 buttons.
        Send latest data to robot arm using TCP socket.
        Receive force feedback from robot arm.
        Generate force feedback.
        """
        # Read stylus states
        button1, button2, pos, ori = self.getRoutesIn()

        mydict = {"Position": str(pos.getValue()), "Rotation": str(ori.getValue(
        ).toEulerAngles()), "Button1": str(button1.getValue()), "Button2": str(button2.getValue())}

        # Send stylus states to robot arm
        data_to_send = pickle.dumps(mydict)
        s.sendall(data_to_send)

        # Receive force feedback
        received = s.recv(2048)
        if received:
            self.force = list(received.split(" "))
            for i in range(3):
                self.force[i] = float(self.force[i])

        # Render force on stylus
        self.dn["FORCE"].force.setValue(
            Vec3f(self.force[0], self.force[1], self.force[2]))

        return True


device = getHapticsDevice(0)
if not device:
    di = createX3DNodeFromString(
        """<DeviceInfo><AnyDevice/></DeviceInfo>""")[0]
    device = getHapticsDevice(0)

position_change = DeviceChangeField()
if device:
    device.mainButton.routeNoEvent(position_change)
    device.secondaryButton.routeNoEvent(position_change)
    device.devicePosition.routeNoEvent(position_change)
    device.trackerOrientation.routeNoEvent(position_change)

    # Connect to TCP socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
