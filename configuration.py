"""
This file contains all variables definition.
"""
MODE = 1  # 0: manual control, #1: auto detect

GRAVITY = 9.8

# Visual view point
cameraDistance = 1.
cameraYaw = 180
cameraPitch = -30
cameraTargetPosition = [0., 0.35, 0.615]

# scene
ROBOT_LOCATION = [0.0, 0.0, 0.615]
PLANE_LOCATION = [0, 0, 0]
PLANE_PATH = "plane.urdf"
TABLE_LOCATION = [0, 0.35, 0]
TABLE_PATH = "table/table.urdf"

OBJECT = {"duck": [-0.1, 0.35, 0.65], "cuboid": [0, 0.5, 0.65],
          "tool": [0.2, 0.36, 0.65], "lego": [0.2, 0.53, 0.65]}
# OBJECT = {"duck": [-0.1, 0.35, 0.65], "cuboid": [0, 0.5, 0.65],
#           "tool": [0.2, 0.5, 0.65], "lego": [-0.2, 0.53, 0.65]}
OBJECT_FOLDER = "/simulation_data/objects/"
COLOR = ["red", "yellow"]
TRAY = {"tray_red": [0.5, 0.4, 0.63], "tray_yellow": [-0.5, 0.4, 0.63]}
TRAY_FOLDER = "/simulation_data/tray/"

# Trajectory
TRAJECTORY_PATH_PICK = "trajectory_data/pick"
TRAJECTORY_PATH_PLACE = "trajectory_data/place"

# Reinforcement learned model
RL_PATH_PICK = "./intent_detection/pick_model/"
RL_PATH_PLACE = "./intent_detection/place_model/"

timeStep = 1. / 60.
MAX_FORCE = 1.5
MIN_FORCE = 0.3
DEFAULT_FORCE_COEF = 6  # greater brings stronger force
BREAK_LOOP = 90  # easy to break for small value

TASK_RECORD_FILE = "result/task_record.csv"
LOGGER_FOLDER = "result/robotlogger"
