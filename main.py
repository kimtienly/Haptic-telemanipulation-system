"""
Main function.

"""
from configuration import *
from haptic import *
from intent_detection import *
from simulation.environment import TaskEnvironment
from simulation.robots import PandaArm
from trajectory import TrajectoryGuidance
import pybullet as p
import pybullet_data
import time
import datetime
import numpy as np
from csv import writer
import random


def force_guidance(trajectory, force_coef=DEFAULT_FORCE_COEF, th_force=0.2, dx=0.05, allowbreak=False, inplace=False):
    """
    Loop through a sequence of steps for all phases of a trajectory to the target.
    """
    if not inplace:
        # adaptive for moving towards target
        th_force = MIN_FORCE+dx*(len(trajectory)-1)

    step = len(trajectory)
    # Trajectory force
    for phase in trajectory:
        x = False
        if step <= 1:
            x = True
        result = generate_force(
            J.dot(phase.T), force_coef, th_force,  allowbreak=allowbreak, inplace=inplace, final=x)
        if not result:
            break
        if not inplace:
            th_force -= dx
        step -= 1


def move_to_phase(rl_intent, rl_env_intent, force_coef=DEFAULT_FORCE_COEF, th_force=MIN_FORCE, allowbreak=False, inplace=False):
    """
    Feed position into reinforcement learning model to get a a sequence of steps to the nearest phase.   
    """
    observation = rl_env_intent.setstate(np.asarray(J.dot(panda.curr_q)))
    step = 0
    breaked = False
    while True:
        action = rl_intent.choose_action(observation)
        observation_, reward, done, info = rl_env_intent.step(
            action, env.finishedObjects)
        observation = observation_
        result = generate_force(
            observation_, force_coef+step, th_force, allowbreak=allowbreak, inplace=inplace)

        # if done or (not result):
        if done:
            break
        if not result:
            done = False
            break
        step += 1
    if done:
        return True, info
    else:
        return False, None


def generate_force(target, force_coef, th_force, allowbreak=False, inplace=False, break_loop=BREAK_LOOP, final=False):
    """
    Generate guidance force on haptic device.
    """
    loop = 0
    breaked = False
    after = target
    before = J.dot(panda.curr_q)
    force = np.asarray(before-after)*force_coef
    pre_ = 0
    while not np.all(abs(force) <= th_force):
        controller.send_force_feedback(force)
        panda.move_via_ee_controller(
            *controller.get_commands())
        p.stepSimulation()
        before = J.dot(panda.curr_q)
        force = np.asarray(before-after)*force_coef
        force = safety_filter(force)

        if (allowbreak and loop == break_loop) or (inplace ^ panda.isGrasping()):
            breaked = True
            pre_ = force

        if breaked:
            pre_ = pre_*0.3
            force = pre_

        loop += 1
    if allowbreak:
        return not breaked  # False if can not reach because of breaking mechanism
    else:
        return True  # reached position


def safety_filter(force):
    """
    Filter the force to avoid exceed limits of the haptic device.
    """
    if np.any(abs(force) > MAX_FORCE):
        loc = (np.where(abs(force) > MAX_FORCE)[0]).tolist()
        for pos in loc:
            force[pos] = np.sign(force[pos])*MAX_FORCE
    return force


def record_data(file_name, list_of_elem):
    """
    Record task execution data.
    """
    # Open file in read mode to count line
    with open(file_name) as file:
        testID = sum(1 for line in file)
    # Open file in append mode
    with open(file_name, 'a+', newline='') as file:
        data = [testID]+list_of_elem
        # Create a writer object from csv module
        csv_writer = writer(file)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(data)
        print("Data saved: "+str(data))
        return testID


# Pybullet interface configuration
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setTimeStep(timeStep)
p.setGravity(0, 0, -GRAVITY)
p.resetDebugVisualizerCamera(cameraDistance=cameraDistance,
                             cameraYaw=cameraYaw,
                             cameraPitch=cameraPitch,
                             cameraTargetPosition=cameraTargetPosition)


# Draw scene
groundId = p.loadURDF(PLANE_PATH, PLANE_LOCATION)
tableId = p.loadURDF(TABLE_PATH, TABLE_LOCATION)
traysID = []
for tray_ in TRAY:
    trayID = p.loadURDF(TRAY_FOLDER+tray_+".urdf", TRAY[tray_])
    traysID.append(trayID)
objectsID = {}
object_list = []
for object_ in OBJECT:
    r = random.randint(0, 1)
    objectID = p.loadURDF(OBJECT_FOLDER+object_+"_" +
                          COLOR[r]+".urdf", OBJECT[object_])
    objectsID[object_] = objectID
    object_list.append(object_)

# Initialize Panda robot arm
panda = PandaArm(p)
panda.load_robot(ROBOT_LOCATION)
J = panda.get_Jacobian()

# Manage environment objects (check done tasks)
env = TaskEnvironment(p, objectsID, traysID)

# Initial trajectories and intent detection for force guidance mode
if MODE != 0:
    # Picking intent detection configuration
    trajectories = TrajectoryGuidance(
        TRAJECTORY_PATH_PICK, object_list)
    rl_env = ArmEnv(J, trajectories, object_list)
    RL = DeepQNetwork(rl_env.action_number, rl_env.state_dim,
                      path=RL_PATH_PICK
                      )
    picking_detection = IntentDetection(RL, rl_env)
    picking_detection.start_stream()

    # Placing intent detection configuration
    place_dict = {}
    for object_ in object_list:
        place_dict[object_] = {}
        place_dict[object_]["list"] = [object_+"1", object_+"2"]
        place_dict[object_]["trajectories"] = TrajectoryGuidance(
            TRAJECTORY_PATH_PLACE, place_dict[object_]["list"])
        place_dict[object_]["rl_env"] = ArmEnv(
            J, place_dict[object_]["trajectories"], place_dict[object_]["list"])
        place_dict[object_]["rl"] = DeepQNetwork(place_dict[object_]["rl_env"].action_number, place_dict[object_]["rl_env"].state_dim,
                                                 path=RL_PATH_PLACE+object_
                                                 )


# Connect to operator
print("Ready to connect to master!")
controller = InputControllerViaSocket()
controller.start_controller()
desired_cart_pos, desired_ori = controller.get_initial_states()

# Move robot to original position that matches haptic
k = 0
while k <= 60:
    panda.move_via_ee_controller(desired_cart_pos, desired_ori, 0)
    p.stepSimulation()
    k += 1

print("Start manipulating...")
panda.startLogging()
timer_grasp = 0
start = time.time()
cnt_grasp = 0
pre_graspstate = False
loop = 0

# Main loop
while True:
    force_feedback = panda.get_force()
    force_feedback = safety_filter(force_feedback)
    controller.send_force_feedback(force_feedback)
    panda.move_via_ee_controller(*controller.get_commands())
    p.stepSimulation()
    env.update()
    # Force guidance mode
    if MODE != 0:
        # Intent detection for picking task
        if (not panda.isGrasping()) and (loop == 5):
            detected, info, number_of_steps = picking_detection.isDetected()
            if detected:
                target = info[0]
                if (not env.checkFinishedObject(target)):
                    if int(info[1]) >= 1:
                        atphase = move_to_phase(RL, rl_env,
                                                DEFAULT_FORCE_COEF-number_of_steps, allowbreak=True)
                        if atphase:
                            trajectory = trajectories.get_trajectory_to_target(
                                viapoint=np.asarray(panda.curr_q), target=info[0], phase=int(info[1]))
                            force_guidance(trajectory, allowbreak=True)

        # Intent detection for placing task
        if panda.isGrasping() and timer_grasp > 50:
            for object_ in objectsID:
                if (panda.objectTouch(objectsID[object_])):
                    atphase, trayinfo = move_to_phase(
                        place_dict[object_]["rl"], place_dict[object_]["rl_env"], DEFAULT_FORCE_COEF/2, allowbreak=True, inplace=True)

                    if atphase:
                        trajectory = place_dict[object_]["trajectories"].get_trajectory_to_target(
                            viapoint=np.asarray(panda.curr_q), target=trayinfo[0], phase=int(trayinfo[1]))
                        force_guidance(
                            trajectory, force_coef=DEFAULT_FORCE_COEF/2, allowbreak=True, inplace=True)
        if loop == 5:
            loop = 0
        loop += 1

        picking_detection.update(np.asarray(
            J.dot(panda.curr_q)), env.finishedObjects)

    # Count number of grasp (There may be a continuous grasping behavior through several loops)
    if panda.isGrasping():
        if timer_grasp == 0:
            cnt_grasp += 1
        timer_grasp += 1
    if pre_graspstate and not panda.isGrasping():  # reset timer if the object has been placed
        timer_grasp = 0
    pre_graspstate = panda.isGrasping()

    # Ends when all tasks are done or controller disconnects
    if ((env.isDone()) or (not controller._active)):
        break

    time.sleep(timeStep)


panda.stopLogging()
# panda.logger.plot()
end = time.time()
data_to_save = [datetime.datetime.now().strftime(
    "%d/%m/%Y %H:%M:%S"), MODE, end-start, env.totalFinishedObjects(), cnt_grasp, np.sum(panda.logger.getEnergy())]
# [Timestamp, mode, execution time, number of finished object, number of grasp, energy]
testID = record_data(TASK_RECORD_FILE, data_to_save)
panda.logger.save(LOGGER_FOLDER+"/"+str(testID)+".csv")
