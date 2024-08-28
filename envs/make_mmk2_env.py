from typing import List
from envs.airbot_mmk_env import AirbotMmkEnv, MMK2Robot
from robots.ros_robots.ros_robot_config import EXAMPLE_CONFIG, JOINT_POSITION
from std_msgs.msg import Float32MultiArray, Int32
from sensor_msgs.msg import JointState


def make_environment(env_config):
    robot_num = env_config["robot_num"]
    # TODO: change this to start state and start action
    start_joint = env_config["start_joint"]
    arm_joints_num = 7
    arms_end = arm_joints_num * 2

    # config start(default) act
    EXAMPLE_CONFIG["reset"]["arm"]["left"][JOINT_POSITION] = JointState(
        position=start_joint[:arm_joints_num]
    )
    EXAMPLE_CONFIG["reset"]["arm"]["right"][JOINT_POSITION] = JointState(
        position=start_joint[arm_joints_num:arms_end]
    )
    EXAMPLE_CONFIG["reset"]["head"][JOINT_POSITION] = Float32MultiArray(
        data=start_joint[arms_end : arms_end + 2]
    )
    EXAMPLE_CONFIG["reset"]["spine"][JOINT_POSITION] = Int32(
        data=start_joint[arms_end + 2 :]
    )

    robots = []
    for i in range(robot_num):
        robots.append(MMK2Robot(EXAMPLE_CONFIG))

    return AirbotMmkEnv(
        robot_instances=robots,
        camera_names=env_config["camera_names"],
    )
