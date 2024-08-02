import numpy as np
from threading import Thread
from typing import Dict
from functools import partial
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Float32MultiArray, Int32
from ros_tools import Lister
from convert_all import flatten_dict


# action and observation types
JOINT = "joint"
JOINT_MIT = "joint_mit"  # joint position and velocity
JOINT_POSITION = "joint_position"
JOINT_VELOCITY = "joint_velocity"
JOINT_EFFORT = "joint_effort"
EEF_POSE = "pose"
EEF_POSE_POSITION = "pose_position"
EEF_POSE_ORIENTATION = "pose_orientation"
EEF_VELOCITY = "eef_velocity"
EEF_EFFORT = "eef_effort"

SIMPLE_KEY_MAP = {
    JOINT: "a",
    JOINT_MIT: "pv",
    JOINT_POSITION: "p",
    JOINT_VELOCITY: "v",
    JOINT_EFFORT: "e",
}

# Initiate All Available Configs for Actions and Observations
ACTIONS_TOPIC_CONFIG = {
    "arm": {
        "left": {
            JOINT_POSITION: ("/control/arms/left/joint_states", JointState),
            EEF_POSE: ("/control/arms/left/pose", Pose),
        },
        "right": {
            JOINT_POSITION: ("/control/arms/right/joint_states", JointState),
            EEF_POSE: ("/control/arms/left/pose", Pose),
        },
    },
    "head": {
        JOINT_POSITION: ("/control/head/move_to", Float32MultiArray),
        JOINT_VELOCITY: ("/control/head/move", Float32MultiArray),
    },
    "spine": {
        JOINT_POSITION: ("/control/spine/move_to", Int32),
        JOINT_VELOCITY: ("/control/spine/move", Int32),
    },
}
OBSERVATIONS_TOPIC_CONFIG = {
    "arm": {
        "left": {
            JOINT_POSITION: (
                "/follower_arms/left/joint_states",
                JointState,
            ),
            EEF_POSE: ("/follower_arms/left/current_pose", Pose),
        },
        "right": {
            JOINT_POSITION: (
                "/follower_arms/right/joint_states",
                JointState,
            ),
            EEF_POSE: ("/follower_arms/right/current_pose", Pose),
        },
    },
    "head": {
        JOINT_POSITION: (
            "/state/head/pos",
            Float32MultiArray,
        )
    },
    "spine": {
        JOINT_POSITION: (
            "/state/spine/pos",
            Int32,
        )
    },
}

mmk_config = {
    # only can have one type for each obs, action and reset
    "observation": {
        "arm": {"left": JOINT_POSITION, "right": JOINT_POSITION},
        "head": JOINT_POSITION,
        "spine": JOINT_POSITION,
    },
    "action": {
        "arm": {"left": JOINT_POSITION, "right": JOINT_POSITION},
        "head": JOINT_VELOCITY,
        "spine": JOINT_VELOCITY,
    },
    "reset": {  # will also be used to get the dim of each action
        # the reset actions type can be different from those of step
        "arm": {
            "left": {JOINT_POSITION: JointState(position=[0] * 7)},
            "right": {JOINT_POSITION: JointState(position=[0] * 7)},
        },
        "head": {JOINT_POSITION: Float32MultiArray(data=[0.5541, 0.0498])},
        "spine": {JOINT_POSITION: Int32(data=-97443)},
    },
    # can have multi types for each action and observation but only one will be used for preprocess and lister
    "preprocess": {
        # spine pos to float[-2π, 0]
        "observation": {
            "spine": {
                JOINT_POSITION: lambda x: (np.array(x) / 120000.0 * 2 * np.pi).tolist()
            }
        },
        "action": {
            "spine": {
                # pos to int[-120000, 0]
                JOINT_POSITION: lambda x: (np.array(x) * 120000.0 / 2 / np.pi).tolist(),
                # vel to int[-1, 1]
                JOINT_VELOCITY: lambda x: [np.round(min(max(v, -1), 1)) for v in x],
            },
            # elements of vel tuple to float[-1, 1]
            "head": {JOINT_VELOCITY: lambda x: [min(max(v, -1), 1) for v in list(x)]},
        },
    },
    "lister": {
        "observation": {  # convert all current data to list
            "arm": {
                "left": {JOINT_POSITION: Lister.joint_state_to_list},
                "right": {JOINT_POSITION: Lister.joint_state_to_list},
            },
            "head": {
                JOINT_POSITION: Lister.data_field_to_list,
                JOINT_VELOCITY: Lister.data_field_to_list,
            },
            "spine": {
                JOINT_POSITION: Lister.data_field_to_list,
                JOINT_VELOCITY: Lister.data_field_to_list,
            },
        },
        "action": {  # convert action list to target data (type) for publishing
            "arm": {
                "left": {
                    JOINT_POSITION: partial(
                        Lister.list_to_joint_state, SIMPLE_KEY_MAP[JOINT_POSITION]
                    ),
                    EEF_POSE: Lister.list_to_pose,
                },
                "right": {
                    JOINT_POSITION: partial(
                        Lister.list_to_joint_state, SIMPLE_KEY_MAP[JOINT_POSITION]
                    ),
                    EEF_POSE: Lister.list_to_pose,
                },
            },
            "head": {
                JOINT_POSITION: partial(
                    Lister.list_to_given_data_field,
                    ACTIONS_TOPIC_CONFIG["head"][JOINT_POSITION][1],
                ),
                JOINT_VELOCITY: partial(
                    Lister.list_to_given_data_field,
                    ACTIONS_TOPIC_CONFIG["head"][JOINT_VELOCITY][1],
                ),
            },
            "spine": {
                JOINT_POSITION: partial(
                    Lister.list_to_given_data_field,
                    ACTIONS_TOPIC_CONFIG["spine"][JOINT_POSITION][1],
                ),
                JOINT_VELOCITY: partial(
                    Lister.list_to_given_data_field,
                    ACTIONS_TOPIC_CONFIG["spine"][JOINT_VELOCITY][1],
                ),
            },
        },
    },
    "param": {
        "control_freq": 200,
        # the dim of each action must be set since the interface type and dims may be different
        # from the obersevations and reset actions. These dims will be used to convert the action
        # list to target data for publishing
        "actions_dim": {
            "arm": {
                "left": {JOINT_POSITION: 7, EEF_POSE: 7},
                "right": {JOINT_POSITION: 7, EEF_POSE: 7},
            },
            "head": {JOINT_POSITION: 2, JOINT_VELOCITY: 2},
            "spine": {JOINT_POSITION: 1, JOINT_VELOCITY: 1},
        },
        # the tolerance for reaching the target of each action
        # if None, will publish the target for 1s at control_freq
        # if int or float, will be applied to all the dims
        # if list, will be applied to each dim
        "reach_tolerance": None,
    },
}


class AssembledROS1Robot(object):
    """Use the keys and values in config as the keys to get all the configuration"""

    def __init__(self, config: Dict[str, dict] = None) -> None:
        self.params = config["param"]
        self.actions_dim = flatten_dict(self.params["actions_dim"])
        reach_tolerance = self.params["reach_tolerance"]
        if isinstance(reach_tolerance, (int, float)):
            reach_tolerance = {
                key: [reach_tolerance] * value
                for key, value in self.actions_dim.items()
            }
        self.reach_tolerance = reach_tolerance
        # Initiate Preprocess Functions
        obs_pre_funcs = flatten_dict(config["preprocess"]["observation"])
        self.act_pre_funcs = flatten_dict(config["preprocess"]["action"])
        # Initiate Lister Functions
        self.obs_listers = flatten_dict(config["lister"]["observation"])
        self.act_listers = flatten_dict(config["lister"]["action"])
        # Initiate Observation Subscribers and Current Data
        subs_configs = flatten_dict(OBSERVATIONS_TOPIC_CONFIG)
        self.obs_config = flatten_dict(config["observation"])
        self.obs_subs: Dict[str, rospy.Subscriber] = {}
        self.current_data = {}
        for key, value in self.obs_config.items():
            new_key = f"{key}/{value}"
            topic, msg_type = subs_configs[new_key]
            self.obs_subs[new_key] = rospy.Subscriber(
                topic,
                msg_type,
                self._current_state_callback,
                (
                    new_key,
                    self.obs_listers[new_key],
                    obs_pre_funcs.get(new_key, lambda x: x),
                ),
            )
            self.current_data[new_key] = None
            rospy.loginfo(f"Subscribe to {topic} with type {msg_type} as observation")
        # Initiate Action Publishers and Target Data
        pubs_configs = flatten_dict(ACTIONS_TOPIC_CONFIG)
        self.act_config = flatten_dict(config["action"])
        self.act_pubs: Dict[str, rospy.Publisher] = {}
        self.target_data = {}
        for key, value in self.act_config.items():
            new_key = f"{key}/{value}"
            topic, msg_type = pubs_configs[new_key]
            self.act_pubs[new_key] = rospy.Publisher(topic, msg_type, queue_size=10)
            # init target data to None so it won't be published until being set
            self.target_data[new_key] = None
            rospy.loginfo(
                f"Publish to {topic} with type {msg_type} as step/reset action"
            )
        # Initiate Reset Publishers
        self.reset_pubs: Dict[str, rospy.Publisher] = {}
        self.reset_config = flatten_dict(config["reset"])
        self.reset_data = {}
        self.reset_act_com = []
        for key, value in self.reset_config.items():
            if key not in self.act_pubs:
                topic, msg_type = pubs_configs[key]
                self.reset_pubs[key] = rospy.Publisher(topic, msg_type, queue_size=10)
                rospy.loginfo(
                    f"Publish to {topic} with type {msg_type} as just reset action"
                )
            else:
                self.reset_pubs[key] = self.act_pubs[key]
                self.reset_act_com.append(key)
            self.reset_data[key] = value
        # Initiate Basic Parameters200 # TODO: remove these
        self.end_effector_open = 1
        self.end_effector_close = 0
        self.all_joints_num = 17  # 7 for each of the 2 arms, 2 for head, 1 for spine
        # TODO: change all_joints_num to action and observation dim
        Thread(target=self._target_cmd_pub_thread, daemon=True).start()

    def _target_cmd_pub_thread(self):
        """Publish thread for all publishers"""
        rate = rospy.Rate(self.params["control_freq"])
        while not rospy.is_shutdown():
            for key, pub in self.act_pubs.items():
                target_data = self.target_data[key]
                if target_data is None:
                    continue
                # if TypeError: expected [int32] but got [std_msgs/Float32MultiArray]
                # check your lister configuration
                pub.publish(target_data)
            rate.sleep()

    def _current_state_callback(self, data, args):
        """Callback function used for all subcribers"""
        key, lister, preprocess = args
        self.current_data[key] = preprocess(lister(data))
        rospy.logdebug(f"Current data: {self.current_data[key]}")

    @staticmethod
    def get_dim(data, interface):
        """Get the dim of the data according to the type and interface"""
        if isinstance(data, JointState):
            dim = len(Lister.joint_state_to_list(data))
        elif isinstance(data, (Pose, PoseStamped)):
            if interface == EEF_POSE_POSITION:
                dim = 3
            elif interface == EEF_POSE_ORIENTATION:
                dim = 4
            else:
                dim = 7
        else:
            v = getattr(data, "data", None)
            if v is not None:
                if isinstance(v, (int, float)):
                    dim = 1
                else:
                    dim = len(v)
            else:
                raise NotImplementedError
        return dim

    @staticmethod
    def get_interface(key: str):
        """The strint after last / is the interface of states and control"""
        last_slash_index = key.rfind("/")
        if last_slash_index != -1:
            last_slash_substring = key[last_slash_index + 1 :]  # 从'/'之后开始切片
            return last_slash_substring
        else:
            raise ValueError("No interface found")

    def set_target_states(self, qpos, qvel=None, blocking=False):
        if not blocking:
            start = 0
            for key in self.target_data:
                dim = self.actions_dim[key]
                end = int(start + dim)
                target = qpos[start:end]
                if key in self.act_pre_funcs:
                    target = self.act_pre_funcs[key](target)
                self.target_data[key] = self.act_listers[key](target)
                start = end
        else:
            # TODO: change the outer env to use the reset funtion for reseting
            self.reset()

    def get_current_states(self) -> list:
        current_data = []
        for key in self.current_data:
            current_data.extend(self.current_data[key])
        return current_data

    def wait_for_current_states(self) -> None:
        rospy.loginfo("Waiting for current states")
        while not rospy.is_shutdown():
            for value in self.current_data.values():
                if value is None:
                    break
            else:
                break

    def reset(self) -> list:
        res_keys = set(self.reset_data.keys()).difference(self.reset_act_com)
        for key in res_keys:
            self.target_data[key] = self.reset_data[key]
        # TODO: add wait for reset
        max_pub = 25
        period = 0.04
        for _ in range(max_pub):
            for key in res_keys:
                self.reset_pubs[key].publish(self.reset_data[key])
            rospy.sleep(period)
        return self.get_current_states()


if __name__ == "__main__":

    rospy.init_node("test_mmk")
    mmk = AssembledROS1Robot(mmk_config)
    mmk.wait_for_current_states()
    current = mmk.get_current_states()
    print("Current states:", current)
    mmk.set_target_states(current)
    print("Reseting")
    print(mmk.reset())
    print("All Done")
    rospy.spin()
