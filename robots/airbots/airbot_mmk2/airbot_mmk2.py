from robots.airbots.airbot_mmk2.airbot_mmk2 import MMK2
from typing import Dict
import cv2


class AirbotMMK2(object):
    def __init__(self, config: Dict[str, dict] = None):
        self.robot = MMK2("mmk2", -1, "192.168.11.200")
        self.all_joints_num = 14

    def set_target_states(self, qpos, qvel=None, blocking=False):
        if not blocking:
            self.robot.set_arms_servo(qpos[0:6], "l", False)
            self.robot.set_arms_servo(qpos[7:13], "r", False)
            self.robot.set_gripper(qpos[6], "l")
            self.robot.set_gripper(qpos[13], "r")
            # print(qpos)
        else:
            # TODO: change the outer env to use the reset funtion for reseting
            self.reset()

    def get_image(self):
        image = self.robot.get_image("head", "color")
        return image

    def get_current_states(self) -> list:
        all_joint = self.robot.get_all_joint_states()
        return (
            list(all_joint[2])
            + list(all_joint[4])
            + list(all_joint[3])
            + list(all_joint[5])
        )

    def reset(self) -> list:
        self.robot.set_arms(
            [
                -0.2588311731815338,
                -0.7001983523368835,
                0.9470130205154419,
                1.9457160234451294,
                -1.099603295326233,
                -0.41065841913223267,
            ],
            "l",
            False,
        )
        self.robot.set_arms(
            [
                0.6288624405860901,
                -0.6555657386779785,
                0.9355688095092773,
                -1.9705119132995605,
                1.1411840915679932,
                0.368314653635025,
            ],
            "r",
            False,
        )
        self.robot.execute_trajectory(True)
        self.robot.set_head([0.03, -0.95])
        self.robot.set_spine(0.042)

        return self.get_current_states()


def main(args=object):
    robot = AirbotMMK2()
    # robot.set_target_states([0,0,0,0,0,0,1,0,0,0,0,0,0,1])
    # cap=cv2.imread(robot.get_image())

    cv2.imshow("Video", robot.get_image())
    cv2.waitKey(0)
    # 释放视频捕获对象并关闭所有窗口
    # cap.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
