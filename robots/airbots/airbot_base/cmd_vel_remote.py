import rclpy
from geometry_msgs.msg import Twist
from typing import Tuple


class CmdVelRemote(object):

    def __init__(self):
        self._node = rclpy.create_node("pedal_remote")
        self._logger = self._node.get_logger()
        self._cmd: Twist = None
        self._cmd_sub = self._node.create_subscription(
            Twist, "/cmd_vel", self._cmd_callback
        )
        self._logger.info("Waiting for cmd_vel topic")
        while self._cmd is None:
            rclpy.spin_once(self._node)
        self._logger.info("Received cmd_vel topic")
    
    def _cmd_callback(self, msg):
        # self._logger.info(f"Received cmd_vel: {msg}")
        self._cmd = msg

    def get_command(self) -> Tuple[float, float]:
        """Get the current command velocity (linear, angular)."""
        return self._cmd.linear.x, self._cmd.linear.y, self._cmd.angular.z