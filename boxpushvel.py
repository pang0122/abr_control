#imports
from cProfile import label

#import mujoco
#import sys
import traceback
from abr_control.arms.mujoco_config import MujocoConfig
from abr_control.interfaces.mujoco import Mujoco
import mujoco_py
from abr_control.controllers import oscVel, Damping
from abr_control.utils import transformations
import numpy as np
import glfw
up = np.array([0,0,1])

arm_model = ("jaco2ModVel") #jaco2 type arm modified to have a .05 cube
robot_config = MujocoConfig(arm_model)
dt = 0.001
interface = Mujoco(robot_config, dt=dt)
#startup command from force_osc_xyz.py, name joints sequentially
interface.connect(joint_names=[f"joint{ii}" for ii in range(len(robot_config.START_ANGLES))])
#send default angles to the arm
interface.send_target_angles(robot_config.START_ANGLES)
# damp the movements of the arm
#damping = Damping(robot_config, kv=40)
force_track = []
box_vel_x = []
box_vel_y = []
des_vel_x = []
des_vel_y = []
desired_velocity = []
def calc_tangent_angle(ee_xyz, target_xyz):
    direction = np.subtract(target_xyz, ee_xyz)
    z_axis = direction / np.linalg.norm(direction)
    # print(direction_unit)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    R = np.column_stack((x_axis, y_axis, z_axis))

    # convert to euler for OSC
    return transformations.euler_from_matrix(R, axes="rxyz")
vmax = [0.5, 10]
ctrlr = oscVel.OSCVel(
    robot_config,
    kp=60,
    kv=10,
    ko=30,
    #null_controllers=[damping],
    vmax=vmax,  # [m/s, rad/s]
    # control (x, y, z) out of [x, y, z, alpha, beta, gamma]
    ctrlr_dof=[True, True, True, True, True, True],
    correcting_force = 120,
    dt=dt
)
def get_distance_to_box():
    """
    :return: distance between center of EE and center of box. For reference, exactly touching the box is around 0.105
    """
    return np.linalg.norm((interface.get_xyz("box") - interface.get_xyz("EE")))
def prep(target_velocity):
    """
    Moves the EE in a position where pushing the box involves going forward at the target direction.
    :param target_velocity: velocity you wish to push the box in.
    """
    count=0
    interface.model.eq_active[0] = 0
    while (count < 250):
        direction = np.array([0, -1, 0])#Due to welding, starting at different directions causes issues -(target_velocity / np.linalg.norm(target_velocity))
        box_xyz = interface.get_xyz("box")
        start_target = box_xyz.copy()
        start_target += direction * 0.15 #0.15m away from center of box
        feedback = interface.get_feedback()
        ee_xyz = interface.get_xyz("EE")
        target = np.hstack(
            [
                start_target[:3],
                calc_tangent_angle(ee_xyz, box_xyz)
            ]
        )
        u = ctrlr.generate(
            q=feedback["q"],
            dq=feedback["dq"],
            target=target,
            target_velocity=None
        ) #note: Q is angles, dq is angular velocity (rad/s)
        interface.send_forces(u)
        #compensate for the box falling as the sim starts
        interface.set_mocap_xyz(name="target", xyz=start_target)
        error = np.linalg.norm(interface.get_xyz("EE") - start_target[:3])

        #graphing
        box_vel = interface.data.sensor("box_velocity").data
        box_vel_x.append(box_vel[0])
        box_vel_y.append(box_vel[1])
        des_vel_x.append(desired_velocity[0])
        des_vel_y.append(desired_velocity[1])
        #graphing
        force = interface.data.sensor("contact_force").data
        force_track.append(np.linalg.norm(force))
        if error < 0.01:
            count += 1
        if glfw.window_should_close(interface.viewer.window):
            break
    print("starting level reached")
    ctrlr.simple_pid_x.reset()
    ctrlr.simple_pid_y.reset()
    return

def push(duration, target_velocity):
    welded = False
    count = 0
    box_vel = interface.data.sensor("box_velocity").data
    while count < duration:
        if welded:
            vel_offset = None
        else:
            direction = interface.get_xyz("box") - interface.get_xyz("EE")
            direction[2] = 0
            vel_offset = -(direction/np.linalg.norm(direction) * (vmax[0]-0.03)) #if at rest this is fine
            vel_offset += box_vel[:3]
            vel_offset = np.hstack((vel_offset, (0,0,0)))

        if not welded and get_distance_to_box() < 0.102: #This is just barely large enough that it only activates
                #the weld on the surface facing the arm
            interface.model.eq_active[0] = 1
            welded = True
            print("weld")
        feedback = interface.get_feedback()
        ee_xyz = interface.get_xyz("EE")
        box_xyz = interface.get_xyz("box")
        tangent_angle = calc_tangent_angle(ee_xyz, box_xyz)
        box_xyz[1] -= 0.02
        target = np.hstack((box_xyz, tangent_angle))

        u = ctrlr.generate(
            q=feedback["q"],
            dq=feedback["dq"],
            target=target,
            box_velocity=box_vel,
            target_velocity = vel_offset,
            desired_box_velocity=target_velocity,
            override_task=welded
        )
        interface.send_forces(u)
        force = interface.data.sensor("contact_force").data
        force_track.append(np.linalg.norm(force))
        box_vel = interface.data.sensor("box_velocity").data
        box_vel_x.append(box_vel[0])
        box_vel_y.append(box_vel[1])
        des_vel_x.append(desired_velocity[0])
        des_vel_y.append(desired_velocity[1])
        if welded:
            count += 1
    interface.model.eq_active[0] = 0
    return
try:
    desired_velocity = [-0.1, 0.1, 0]
    prep(desired_velocity)
    push(2000, desired_velocity)
    desired_velocity = [0, -0.5, 0]
    ctrlr.simple_pid_x.reset()
    ctrlr.simple_pid_y.reset()
    push(300, desired_velocity)
    desired_velocity = [0.2, 0, 0]
    ctrlr.simple_pid_x.reset()
    ctrlr.simple_pid_y.reset()
    push(1000, desired_velocity)

    #prep(desired_velocity)
    #push(1000, desired_velocity)
    while True:
        feedback=interface.get_feedback()
        u = ctrlr.generate(
        q=feedback["q"],
        dq=feedback["dq"],
        target=[0,0,1,0,0,0] #only exists so sim doesn't immediately exit out
        )
        interface.send_forces(u)
        if glfw.window_should_close(interface.viewer.window):
            break
except:
  print("Something bad happened")
  print(traceback.format_exc())

finally:
  interface.disconnect()
  print("Graphing")
  import matplotlib.pyplot as plt
  fig = plt.figure(figsize=(8,12))
  ax1 = fig.add_subplot(211)
  ax1.set_ylabel("force (N)")
  ax1.set_xlabel("Time (ms)")
  ax1.plot(force_track, label="force on EE")
  ax1.legend()
  ax2 = fig.add_subplot(212)
  ax2.set_ylabel("direction velocity (m/s)")
  ax2.set_xlabel("Time(ms)")
  ax2.plot(box_vel_x, label="x velocity")
  ax2.plot(box_vel_y, label="y velocity")
  ax2.plot(des_vel_x, label="target x velocity")
  ax2.plot(des_vel_y, label="target y velocity")
  ax2.legend()
  plt.show()