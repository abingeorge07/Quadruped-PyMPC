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

        if self.sim_param["controller"] == "NMPCController":
            from quad_pympc.quadruped_pympc.controllers.gradient.nominal.centroidal_nmpc_nominal import Acados_NMPC_Nominal
            self.controller = Acados_NMPC_Nominal()
        elif self.sim_param["controller"] == "DefaultController":
            from controllers.foothold_planner.defaultController import DefaultController
            self.controller = DefaultController(self.gait_phase)

        




    def compute_control(
        self,
        state_current: dict,
        ref_state: dict,
        contact_sequence: np.ndarray,
        inertia: np.ndarray,
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


    

        # Gethe the GRFs, foothold, joint positions, velocities, and accelerations
        nmpc_GRFs, nmpc_footholds, nmpc_predicted_state, _ = self.controller.compute_control(
                    state_current, ref_state, contact_sequence, inertia=inertia, external_wrenches=external_wrenches
                )

        nmpc_joints_pos = None
        nmpc_joints_vel = None
        nmpc_joints_acc = None
        

        # nmpc_footholds = LegsAttr(
        #     FL=nmpc_footholds[0], FR=nmpc_footholds[1], RL=nmpc_footholds[2], RR=nmpc_footholds[3]
        # )





        return (
            nmpc_GRFs,
            nmpc_footholds,
            nmpc_joints_pos,
            nmpc_joints_vel,
            nmpc_joints_acc,
            nmpc_predicted_state,
        )