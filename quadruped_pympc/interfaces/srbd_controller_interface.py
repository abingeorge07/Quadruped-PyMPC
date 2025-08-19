import numpy as np
import config as cfg

from additional_lib.helper_functions import get_gait_phase

class SRBDControllerInterface:
    """This is an interface for a controller that uses the SRBD method to optimize the gait"""

    def __init__(self):
        """Constructor for the SRBD controller interface"""

        self.type = cfg.mpc_params['type']
        self.mpc_dt = cfg.mpc_params['dt']
        self.horizon = cfg.mpc_params['horizon']

        # Contact MPC
        self.previous_contact_mpc = np.array([1, 1, 1, 1])

        # Simulation parameters
        self.sim_param = cfg.sim_param

        # Get the gait phase
        self.gait_phase = get_gait_phase(self.sim_param)

        # Get the number of joints
        self.num_joints = cfg.robot_params['num_joints']

        if self.sim_param["controller"] == "NMPCController":
            from quad_pympc.quadruped_pympc.controllers.gradient.nominal.centroidal_nmpc_nominal import Acados_NMPC_Nominal
            self.controller = Acados_NMPC_Nominal()

        elif self.sim_param["controller"] == "DefaultController":
            if(cfg.robot_params['num_joints'] == 12):   
                from controllers.foothold_planner.defaultController import DefaultController
                self.controller = DefaultController(self.gait_phase)
            elif(cfg.robot_params['num_joints'] == 8):
                from controllers.foothold_planner.dof_8.walk_main import simple_walk
                self.controller = simple_walk(self.gait_phase, morphology=cfg.robot_params['morphology'])

        




    def compute_control(
        self,
        state_current: dict,
        ref_state: dict,
        contact_sequence: np.ndarray = None,
        inertia: np.ndarray = None,
        external_wrenches: np.ndarray = np.zeros((6,))
    ):
        """Compute the control using the SRBD method

        Args:
            state_current (dict): The current state of the robot
            ref_state (dict): The reference state of the robot
            contact_sequence (np.ndarray): The contact sequence of the robot
            inertia (np.ndarray): The inertia of the robot
            optimize_swing (int): The flag to optimize the swing
            external_wrenches (np.ndarray): The external wrench applied to the robot to compensate

        Returns:
            tuple: The GRFs and the feet positions in world frame,
                   and the best sample frequency (only if the controller is sampling)
        """


    
        if(self.sim_param["controller"] == "NMPCController"):
            # Get the the GRFs, foothold, joint positions, velocities, and accelerations
            nmpc_GRFs, nmpc_footholds, nmpc_predicted_state, _ = self.controller.compute_control(
                        state_current, ref_state, contact_sequence, inertia=inertia, external_wrenches=external_wrenches
                    )

            nmpc_joints_pos = None
            nmpc_joints_vel = None
            nmpc_joints_acc = None
    
            # print("NMPC GRFs: ", nmpc_GRFs)
            # print("NMPC Footholds: ", nmpc_footholds)
            # print("NMPC Predicted State: ", nmpc_predicted_state)
            # print("NMPC Joints Position: ", nmpc_joints_pos)
            # print("NMPC Joints Velocity: ", nmpc_joints_vel)
            # print("NMPC Joints Acceleration: ", nmpc_joints_acc)
            
            # input("Press Enter to continue...")

            return (
                nmpc_GRFs,
                nmpc_footholds,
                nmpc_joints_pos,
                nmpc_joints_vel,
                nmpc_joints_acc,
                nmpc_predicted_state,
            )
        
        
        elif(self.sim_param["controller"] == "DefaultController"):
                if(self.num_joints == 8):
                    # Get the the GRFs, foothold, joint positions, velocities, and accelerations
                    torque = self.controller.compute_control(state_current, ref_state)

                    return torque
                
                elif(self.num_joints == 12):
                    print("Default controller for 12 DOF not implemented yet")
                    return None, None, None, None, None, None