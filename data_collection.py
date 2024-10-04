from airbot_client import Robot
# from airbot_python_sdk.airbot import Robot
import argparse
import time
from threading import Thread
import os
# import keyboard

def parse_args()-> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aloha data collection")
    parser.add_argument("--leader_number", type=int, default=1, help="Number of the leader")
    parser.add_argument("--follower_number", type=int, default=1, help="Number of the follower")
    parser.add_argument("--leader_arm_type", type=list[str], default=["play_long"], help="Type of the leader's arm")
    parser.add_argument("--follower_arm_type", type=list[str], default=["play_long"], help="Type of the follower's arm")
    parser.add_argument("--leader_end_effector", type=list[str], default=["E2B"], help="End effector of the leader's arm")
    parser.add_argument("--follower_end_effector", type=list[str], default=["G2"], help="End effector of the follower's arm")
    parser.add_argument("--leader_can_interface", type=list[str], default=["can0"], help="Can interface of the leader's arm")
    parser.add_argument("--follower_can_interface", type=list[str], default=["can1"], help="Can interface of the follower's arm")
    parser.add_argument("--leader-domain-id", type=list[int], default=[50], help="Domain id of the leader")
    parser.add_argument("--follower-domain-id", type=list[int], default=[100], help="Domain id of the follower")
    parser.add_argument("--frequency", type=int, default=25, help="Frequency of the data collection")
    parser.add_argument("--start-episode", type=int, default=0, help="Start episode")
    parser.add_argument("--end-episode", type=int, default=100, help="End episode")
    parser.add_argument("--task-name", type=str, default="aloha", help="Task name")
    parser.add_argument("--start-joint-position", type=list, default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], help="Start joint position")
    parser.add_argument("--start-end-effector-position", type=float, default=0.0, help="Start end effector position")
    assert len(parser.parse_args().leader_arm_type) == parser.parse_args().leader_number
    assert len(parser.parse_args().follower_arm_type) == parser.parse_args().follower_number
    assert len(parser.parse_args().leader_end_effector) == parser.parse_args().leader_number
    assert len(parser.parse_args().follower_end_effector) == parser.parse_args().follower_number
    assert len(parser.parse_args().leader_can_interface) == parser.parse_args().leader_number
    assert len(parser.parse_args().follower_can_interface) == parser.parse_args().follower_number
    assert len(parser.parse_args().leader_domain_id) == parser.parse_args().leader_number
    assert len(parser.parse_args().follower_domain_id) == parser.parse_args().follower_number
    return parser.parse_args()

class data_collection:
    def __init__(self, args):
        self.args = args
        self.init_robot()
        self.data = []

    def init_robot(self):
        args = self.args
        leader_robot = []
        follower_robot = []
        for i in range(args.leader_number):
            leader_robot.append(Robot(arm_type=args.leader_arm_type[i], end_effector=args.leader_end_effector[i], can_interface=args.leader_can_interface[i], domain_id=args.leader_domain_id[i]))
        for i in range(args.follower_number):
            follower_robot.append(Robot(arm_type=args.follower_arm_type[i], end_effector=args.follower_end_effector[i], can_interface=args.follower_can_interface[i], domain_id=args.follower_domain_id[i]))
        self.leader_robot = leader_robot
        self.follower_robot = follower_robot
        self.reset_robot()

    def reset_robot(self):
        args = self.args
        leader_robot = self.leader_robot
        follower_robot = self.follower_robot
        for i in range(args.leader_number):
            if args.leader_arm_type[i] == "replay":
                continue
            if leader_robot[i].get_current_state() != "ONLINE_TRAJ":
                assert leader_robot[i].online_idle_mode(), "Leader robot %d online idle mode failed" % i
                assert leader_robot[i].online_traj_mode(), "Leader robot %d online traj mode failed" % i
            time.sleep(0.5)
        for i in range(args.follower_number):
            if follower_robot[i].get_current_state() != "SLAVE_MOVING":
                assert follower_robot[i].online_idle_mode(), "Follower robot %d online idle mode failed" % i
                assert follower_robot[i].slave_waiting_mode(args.leader_domain_id[i]), "Follower robot %d slave waiting mode failed" % i
                time.sleep(0.5)
                assert follower_robot[i].slave_reaching_mode(), "Follower robot %d slave reaching mode failed" % i
                while follower_robot[i].get_current_state() != "SLAVE_REACHED":
                    time.sleep(0.5)
                assert follower_robot[i].slave_moving_mode(), "Follower robot %d slave moving mode failed" % i
        for i in range(args.leader_number):
            if args.leader_arm_type[i] == "replay":
                continue
            assert leader_robot[i].set_target_joint_q(args.start_joint_position), "Leader robot %d set target joint q failed" % i
            assert leader_robot[i].online_idle_mode(), "Leader robot %d online idle mode failed" % i

    def start_data_collection(self, round:int):
        args = self.args
        for i in range(args.leader_number):
            if args.leader_arm_type[i] == "replay":
                continue
            assert self.leader_robot[i].demonstrate_prep_mode(), "Leader robot %d demonstrate start mode failed" % i
        self.collect_data(round=round)

    def collect_data_once(self):
        args = self.args
        leader_robot = self.leader_robot
        follower_robot = self.follower_robot
        data = []
        data.append(time.time())
        for i in range(args.leader_number):
            data.append(leader_robot[i].get_current_joint_q())
            data.append(leader_robot[i].get_current_end())
        for i in range(args.follower_number):
            data.append(follower_robot[i].get_current_joint_q())
            data.append(follower_robot[i].get_current_end())
        self.data.append(data)
        print(data)

    def collect_data(self, round:int):
        print(round)
        for i in range(100):
            self.collect_data_once()
            time.sleep(1./self.args.frequency)
        print("Round", round, "finished")

def main():
    args = parse_args()
    data_coll = data_collection(args)
    print("Robot initialization finished! You have", args.leader_number, "leader(s) and", args.follower_number, "follower(s)")
    for cnt in range(args.start_episode, args.end_episode+1):
        print("Round: ", cnt, "Please press space to start current round")
        data_coll.start_data_collection(round=cnt)
        data_coll.reset_robot()

if __name__ == "__main__":
    main()
