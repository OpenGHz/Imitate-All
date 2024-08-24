import logging, os, time
from typing import List

# moveit usage
from rclpy.impl.rcutils_logger import RcutilsLogger
from ament_index_python.packages import get_package_share_directory
from moveit.planning import (
    MoveItPy,
    PlanningComponent,
    PlanningSceneMonitor,
    TrajectoryExecutionManager,
)
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_msgs.msg import MotionPlanResponse
from moveit_configs_utils import MoveItConfigsBuilder
from moveit.core.robot_state import RobotState
from moveit.core.robot_model import RobotModel, JointModelGroup, VariableBounds
from moveit.core.planning_scene import PlanningScene
from moveit.core.kinematic_constraints import construct_joint_constraint

# from moveit.core.robot_trajectory import RobotTrajectory
from moveit.core.controller_manager import ExecutionStatus

# from moveit.core.planning_interface import MotionPlanResponse
from geometry_msgs.msg import Pose, PoseStamped


class AirbotPlayMoveit(object):

    def __init__(self, robot_name: str, node_name: str) -> None:
        # instantiate MoveItPy instance and get planning component
        moveit_config = (
            MoveItConfigsBuilder(robot_name)
            .robot_description(file_path="config/airbot.urdf.xacro")
            .joint_limits()
            .robot_description_kinematics()
            .trajectory_execution(file_path="config/moveit_controllers.yaml")
            .pilz_cartesian_limits()
            .planning_pipelines(
                pipelines=["ompl", "pilz_industrial_motion_planner", "stomp"]
            )
            .planning_scene_monitor(
                publish_planning_scene=True,
                publish_geometry_updates=True,
                publish_state_updates=True,
                publish_transforms_updates=True,
                publish_robot_description_semantic=True,
            )
            .moveit_cpp(  # must be set
                file_path=os.path.join(
                    get_package_share_directory("airbot_moveit_config"),
                    "config",
                    "moveit_cpp.yaml",
                )
            )
            .to_moveit_configs()
        ).to_dict()
        self.robot = MoveItPy(node_name=node_name, config_dict=moveit_config)
        self.arm: PlanningComponent = self.robot.get_planning_component("arm")
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.arm.planning_group_name
        self.plan_scene_monitor: PlanningSceneMonitor = (
            self.robot.get_planning_scene_monitor()
        )
        self.traj_manager: TrajectoryExecutionManager = (
            self.robot.get_trajactory_execution_manager()
        )
        self.robot_model: RobotModel = self.robot.get_robot_model()
        self.arm_joint_model_group: JointModelGroup = (
            self.robot_model.get_joint_model_group("arm")
        )
        self.logger.info(f"robot_model.end_effectors: {self.robot_model.end_effectors}")
        self.arm_links = self.arm_joint_model_group.link_model_names
        self.arm_active_joints = self.arm_joint_model_group.active_joint_model_names
        self.logger.info(f"all_arm_links: {self.arm_links}")
        self.pose_target_link = self.arm_links[-1]
        self.pose_reference_link = self.arm_links[0]
        # self.plan_scene.wait_for_current_robot_state(rclpy.clock.Clock().now().to_msg(), 1.0)  # TODO: error now
        self.arm_active_joints_num = len(self.arm_active_joints)
        self.logger.info("AIRBOT Play Moveit instance created")

    def go_home(self):
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(configuration_name="start")
        self.move()

    def move(self, wait=False, sleep_time=0):
        self.plan_and_execute(
            self.robot, self.arm, self.logger, wait=wait, sleep_time=sleep_time
        )

    def go(self, target, wait=False, sleep_time=0, is_pose=False):
        self.arm.set_start_state_to_current_state()
        if is_pose:
            pose = Pose()
            pose.position.x = target[0]
            pose.position.y = target[1]
            pose.position.z = target[2]
            pose.orientation.x = target[3]
            pose.orientation.y = target[4]
            pose.orientation.z = target[5]
            pose.orientation.w = target[6]
            pose_goal = PoseStamped()
            pose_goal.pose = pose
            pose_goal.header.frame_id = self.pose_reference_link
            self.arm.set_goal_state(
                pose_stamped_msg=pose_goal, pose_link=self.pose_target_link
            )
        elif isinstance(target, str):
            self.arm.set_goal_state(configuration_name=target)
        else:
            state = RobotState(self.robot_model)
            if not isinstance(target, dict):
                target = dict(zip(self.arm_active_joints, target))
            state.joint_positions = target
            joint_constraint = construct_joint_constraint(
                robot_state=state,
                joint_model_group=self.arm_joint_model_group,
            )
            self.arm.set_goal_state(motion_plan_constraints=[joint_constraint])
        self.move(wait, sleep_time)

    def get_current_pose(self):
        current_pose = None
        with self.plan_scene_monitor.read_only() as scene:
            scene: PlanningScene
            current_state: RobotState = scene.current_state
            current_state.update()
            pose: Pose = current_state.get_pose(self.pose_target_link)
            current_pose = [
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
        return current_pose

    def get_current_joint_positions(self):
        current_joint_positions = None
        with self.plan_scene_monitor.read_only() as scene:
            scene: PlanningScene
            current_state: RobotState = scene.current_state
            current_state.update()
            current_joint_positions: dict = current_state.joint_positions
        return list(current_joint_positions.values())

    def get_joint_limits(self):
        bounds: List[VariableBounds] = (
            self.arm_joint_model_group.active_joint_model_bounds
        )
        joint_bounds = {}
        for index, joint_name in enumerate(self.arm_active_joints):
            joint_bounds[joint_name] = {}
            joint_bounds[joint_name]["position"] = bounds[index].position_bounded
            joint_bounds[joint_name]["velocity"] = bounds[index].velocity_bounded
            joint_bounds[joint_name]["acceleration"] = bounds[
                index
            ].acceleration_bounded
            joint_bounds[joint_name]["jerk"] = bounds[index].jerk_bounded
        return joint_bounds

    def set_pose_target_link(self, link_name):
        assert (
            link_name in self.arm_links
        ), f"link_name: {link_name} not in all_arm_links"
        self.pose_target_link = link_name

    def wait(self):
        self.traj_manager.wait_for_execution()

    def stop(self):
        self.traj_manager.stop_execution()

    def get_last_execution_status(self) -> ExecutionStatus:
        return self.traj_manager.get_last_execution_status()

    @staticmethod
    def plan_and_execute(
        robot: MoveItPy,
        planning_component: PlanningComponent,
        logger: RcutilsLogger,
        single_plan_parameters=None,
        multi_plan_parameters=None,
        wait=False,
        sleep_time=0.0,
    ):
        """Helper function to plan and execute a motion."""
        # plan to goal
        if multi_plan_parameters is not None:
            plan_result = planning_component.plan(
                multi_plan_parameters=multi_plan_parameters
            )
        elif single_plan_parameters is not None:
            plan_result = planning_component.plan(
                single_plan_parameters=single_plan_parameters
            )
        else:
            plan_result = planning_component.plan()

        # execute the plan
        if plan_result.error_code.val == 1:
            plan_result: MotionPlanResponse
            logger.info("Executing plan")
            robot_trajectory = plan_result.trajectory
            robot.execute(robot_trajectory, controllers=[])
            if wait:
                robot.get_trajactory_execution_manager().wait_for_execution()
        else:
            logger.error("Planning failed")

        time.sleep(sleep_time)
