import numpy as np

np.set_printoptions(precision=3, suppress=True)
from numpy.linalg import norm, solve
import time
import casadi as cs

# import example_robot_data as robex
import copy
import os
import time



# Mujoco magic
import mujoco
import mujoco.viewer

# Adam and Liecasadi magic
import config as cfg
import os

dir_path = os.path.dirname(os.path.realpath(__file__))



import config as cfg



IT_MAX = 5
DT = 1e-2
damp = 1e-3
num_joints = int(cfg.robot_params['num_joints'])
damp_matrix = damp * np.eye(num_joints)


# Class for solving a generic inverse kinematics problem
class InverseKinematicsNumeric:
    def __init__(self) -> None:
        """
        This method initializes the inverse kinematics solver class.

        Args:

        """

        robot_dir = os.getcwd() + "/models/quadrupeds/"

        robot_name = cfg.robot

        robot_path = robot_dir + robot_name + "/" + robot_name + ".xml"

        # Load the MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(robot_path)
        self.data = mujoco.MjData(self.model)

        # Get feet body id
        self.feet_geom_id, self.feet_body_id = self.get_feet_body_id()

    # Get the feet body ids
    def get_feet_body_id(self):

        # Initialize a dictionary to store the feet positions
        feet_geom_id = {}
        feet_body_id = {}

        foot_names = cfg.toe_names_geom

        for leg in cfg.leg_names:
            mujoco_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, foot_names[leg])
            feet_geom_id[leg] = mujoco_id
            feet_body_id[leg] = self.model.geom_bodyid[mujoco_id]

        return feet_geom_id, feet_body_id

    # Get the joint positions in world frame
    def get_feet_positions_mujoco(self):
        """
        Get the feet positions in world frame using MuJoCo body xpos.
        Returns:
            dict: A dictionary containing the feet positions for each leg.
        """

        # Initialize a dictionary to store the feet positions
        feet_pos = {}

        foot_names = cfg.toe_names_geom

        for leg in cfg.leg_names:
            mujoco_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, foot_names[leg])
            feet_pos[leg] = self.data.geom_xpos[mujoco_id]


        return feet_pos
    
    # Get feet jacobians
    def get_feet_jacobians(self):
        

        feet_trans_jac = {}
        feet_rot_jac = {}

        # Get the feet positions in world frame
        feet_pos = self.get_feet_positions_mujoco()

        for leg in cfg.leg_names:
            feet_trans_jac[leg] = np.zeros((3, self.model.nv))
            feet_rot_jac[leg] = np.zeros((3, self.model.nv))
            mujoco.mj_jac(
                m=self.model,
                d=self.data,
                jacp=feet_trans_jac[leg],
                jacr=feet_rot_jac[leg],
                point=feet_pos[leg],
                body=self.feet_body_id[leg],
            )


        return feet_trans_jac, feet_rot_jac

    def compute_solution(
        self,
        q: np.ndarray,
        FL_foot_target_position: np.ndarray,
        FR_foot_target_position: np.ndarray,
        RL_foot_target_position: np.ndarray,
        RR_foot_target_position: np.ndarray,
    ) -> np.ndarray:
        """
        This method computes the forward kinematics from initial joint angles and desired foot target positions.

        Args:
            q (np.ndarray): The initial joint angles.
            FL_foot_target_position (np.ndarray): The desired position of the front-left foot.
            FR_foot_target_position (np.ndarray): The desired position of the front-right foot.
            RL_foot_target_position (np.ndarray): The desired position of the rear-left foot.
            RR_foot_target_position (np.ndarray): The desired position of the rear-right foot.

        Returns:
            np.ndarray: The joint angles that achieve the desired foot positions.
        """

        # Set the initial states
        self.data.qpos = q
        mujoco.mj_fwdPosition(self.model, self.data)

        for j in range(IT_MAX):
            feet_pos = self.get_feet_positions_mujoco()

            FL_foot_actual_pos = feet_pos["FL"]
            FR_foot_actual_pos = feet_pos["FR"]
            RL_foot_actual_pos = feet_pos["RL"]
            RR_foot_actual_pos = feet_pos["RR"]

            err_FL = FL_foot_target_position - FL_foot_actual_pos
            err_FR = FR_foot_target_position - FR_foot_actual_pos
            err_RL = RL_foot_target_position - RL_foot_actual_pos
            err_RR = RR_foot_target_position - RR_foot_actual_pos


            # Compute feet jacobian
            feet_jac, _ = self.get_feet_jacobians()


            J_FL = feet_jac["FL"][:, 6:]
            J_FR = feet_jac["FR"][:, 6:]
            J_RL = feet_jac["RL"][:, 6:]
            J_RR = feet_jac["RR"][:, 6:]

            total_jac = np.vstack((J_FL, J_FR, J_RL, J_RR))
            total_err = 100*np.hstack((err_FL, err_FR, err_RL, err_RR))

            # Solve the IK problem
            #dq = total_jac.T @ np.linalg.solve(total_jac @ total_jac.T + damp_matrix, total_err)
            damped_pinv = np.linalg.inv(total_jac.T @ total_jac + damp_matrix) @ total_jac.T
            dq = damped_pinv @ total_err

            # Integrate joint velocities to obtain joint positions.
            q_joint = self.data.qpos.copy()[7:]
            q_joint += dq * DT
            self.data.qpos[7:] = q_joint

            mujoco.mj_fwdPosition(self.model, self.data)
            #mujoco.mj_kinematics(self.model, self.env.mjData)
            #mujoco.mj_step(self.model, self.env.mjData)

        return q_joint

