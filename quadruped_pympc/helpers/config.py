"""This file includes all of the configuration parameters for the MPC controllers
and of the internal simulations that can be launch from the folder /simulation.
"""
import numpy as np

import sys
sys.path.append('../../../additional_lib')
from helper_functions import get_gait_phase, GaitType

robot = 'aliengo' # 'aliengo', 'b2', 'go1', 'hyqreal(not working)', 'mini_cheetah', Origami_8', 'Origami_12', 'anymal_c (LATER)
scene = 'flat' # 'obstacle', 'stair', 'flat', 'stair_simple', 'perlin_0_2', 'perlin_0_4', 'perlin_0_6', 'perlin_0_8'

# If True, the trajectory will be plotted
# only works for terrain-aware controller
plt_traj = True

hardcoded = False 

render = True  # if True, the simulation will be rendered

# Default dimensions for the rendering viewport
WIDTH = 800
HEIGHT = 600

record_params = {
    'RECORD': False,  # if True, the simulation will be recorded
    'FPS': 120,  # if True, the FPS will be recorded
    'filename': 'simulation_recording.mp4',  # name of the recording file
}

# Get latency values
latency_values_acq = True
latency_values_acq_name = "data/latency/latency_flat_values.npz"

mujoco_params = {
    # this is the integration time used in the simulator
    'dt':                          0.001, # in seconds
    "gravity_compensation": -1.0
}

morphologies_dict = {
    'ankles': 0,
    'knees': 1,
    'X': 2,
    'O': 3
}

max_height = 0.35  # maximum height of the robot
phi_stance_hip_front = 0.1
phi_stance_hip_hind = 0.1
phi_stance_abduction_left = 0
phi_stance_abduction_right = -phi_stance_abduction_left


# limb num translation
kFL = 0
kHL = 2
kFR = 1
kHR = 3

leg_names = ['FL', 'FR', 'RL', 'RR']  # Front Left, Front Right, Hind Left, Hind Right


leg_num = {
    'FL': kFL,  # Front Left
    'FR': kFR,  # Front Right
    'RL': kHL,  # Hind Left
    'RR': kHR,  # Hind Right
}

hip_names = {
    'FL': 'FL_hip',
    'FR': 'FR_hip',
    'RL': 'RL_hip',
    'RR': 'RR_hip',
}

toe_names_geom = {
    'FL': 'FL',
    'FR': 'FR',
    'RL': 'RL',
    'RR': 'RR',
}

# Should update start position & goal
# should be based on the scene
goal_param = {
    'x0': 0,
    'y0': 0,
    'final_x': 22,
    'final_y': -1,
    'step_duration': 0.5
}

if (robot == 'Origami_8'):
    high_level_param = {
        'com_controller': "Default", # 'Default', 'TrajectoryPlanner', 'MPCPlanner'
        'dt': 0.2, # for MPC 
        'freq': 100,
        'N': 20,
        'dt_mppi': 0.2,
        'N_mppi': 30,
        'rollout': 400, # number of rollouts for MPPI
        'safety_margin': 0.00001,  #(m),
        'optimizer': 'casadi' # scipy, casadi
    }
else:
    high_level_param = {
        'com_controller': "Default", # 'Default', 'TrajectoryPlanner', 'MPCPlanner'
        'dt': 0.1, # for MPC 
        'freq': 10,
        'N': 10,
        'dt_mppi': 0.2,
        'N_mppi': 30,
        'rollout': 400, # number of rollouts for MPPI
        'safety_margin': 0.1,  #(m),
        'optimizer': 'casadi' # scipy, casadi
    }

    
if (robot == 'aliengo'):
    mass = 24.637
    hip_height = 0.35 # height of the robot (req)
    inertia = np.array([[0.2310941359705289, -0.0014987128245817424, -0.021400468992761768],
                        [-0.0014987128245817424, 1.4485084687476608, 0.0004641447134275615],
                        [-0.021400468992761768, 0.0004641447134275615, 1.503217877350808]])
    
elif (robot == 'Origami_8'):
    mass = 2.5
    hip_height = 0.17 # height of the robot (req)
    inertia =  np.array([[ 3.47733696e-02,  8.36203506e-23, -3.64377544e-03],
                          [ 8.36203506e-23,  4.54184309e-02,  6.73918429e-22],
                          [-3.64377544e-03,  6.73918429e-22,  5.93818280e-02]])
    
elif (robot == 'Origami_12'):
    mass = 24.637
    hip_height = 0.323 # height of the robot (req)
    inertia = np.array([[0.2310941359705289, -0.0014987128245817424, -0.021400468992761768],
                        [-0.0014987128245817424, 1.4485084687476608, 0.0004641447134275615],
                        [-0.021400468992761768, 0.0004641447134275615, 1.503217877350808]])
    
elif (robot == 'go1'):
    mass = 12.019
    hip_height = 0.27
    inertia = np.array([[1.58460467e-01, 1.21660000e-04, -1.55444692e-02],
                        [1.21660000e-04, 4.68645637e-01, -3.12000000e-05],
                        [-1.55444692e-02, -3.12000000e-05, 5.24474661e-01]])

elif (robot == 'b2'):
    mass = 83.49
    hip_height = 0.44  # height of the robot (req)
    inertia = np.array([[0.2310941359705289, -0.0014987128245817424, -0.021400468992761768],
                        [-0.0014987128245817424, 1.4485084687476608, 0.0004641447134275615],
                        [-0.021400468992761768, 0.0004641447134275615, 1.503217877350808]])


elif (robot == 'hyqreal'):
    mass = 108.40
    hip_height = 0.50
    inertia = np.array([[4.55031444e+00, 2.75249434e-03, -5.11957307e-01],
                        [2.75249434e-03, 2.02411774e+01, -7.38560592e-04],
                        [-5.11957307e-01, -7.38560592e-04, 2.14269772e+01]])
    
elif (robot == 'mini_cheetah'):
    mass = 12.5
    hip_height = 0.27  # height of the robot (req)
    inertia = np.array([[1.58460467e-01, 1.21660000e-04, -1.55444692e-02],
                        [1.21660000e-04, 4.68645637e-01, -3.12000000e-05],
                        [-1.55444692e-02, -3.12000000e-05, 5.24474661e-01]])

# elif (robot == 'anymal_c'):
#     mass = 12.5
#     hip_height = 0.27  # height of the robot (req)
#     inertia = np.array([[1.58460467e-01, 1.21660000e-04, -1.55444692e-02],
#                         [1.21660000e-04, 4.68645637e-01, -3.12000000e-05],
#                         [-1.55444692e-02, -3.12000000e-05, 5.24474661e-01]])

# x_center = [r_stance*np.cos(phi_stance_front), r_stance*np.cos(phi_stance_hind), r_stance*np.cos(phi_stance_front), r_stance*np.cos(phi_stance_hind)]
# y_center = [r_stance*np.sin(phi_stance_front), r_stance*np.sin(phi_stance_hind), r_stance*np.sin(phi_stance_front), r_stance*np.sin(phi_stance_hind)]

# Robot parameters for A1
robot_params = {
    "clearance": 0.4 - hip_height,  # height of the robot (max_height - req_height)
    "l_h": 0.085,
    "l1": 0.2,  # hip to knee link length
    "l2": 0.2,  # knee to foot link length
    "length": 0.366,  # length of the robot
    "width": 0.094,  # width of the robot
    "hip_offset": 0.82, #2.4,  # offset of the hip from the body center,
    "knee_offset": -1.57, #-1.65966325,  # offset of the knee from the hip
    "num_joints": 12,  # number of joints in the robot
    # 'delta_width': 0.005,
    'limbs': 4,  # number of limbs in the robot
    'morphology': 'ankles',  # 'X', 'O', 'knees', 'ankles' (default)
    'max_step_length': 0.12,
    'max_vel': 1.0,
    'max_w': 2.0
}

if(robot == 'Origami_8'):

    robot_params['l1'] = 0.1  # hip to knee link length
    robot_params['l2'] = 0.1  # knee to foot link length
    robot_params['length'] = 0.14  # length of the robot
    robot_params['width'] = 0.109  # width of the robot
    robot_params['hip_offset'] = 0.055  # offset of the hip from the body center,
    robot_params['knee_offset'] = 0.0395  # offset of the knee from the hip
    robot_params['num_joints'] = 8  # number of joints in the robot
    robot_params['max_step_length'] = 0.08
    robot_params['max_vel'] = 0.1
    robot_params['max_w'] = 0.5

    # hip_height = 0.17  # height of the robot (req)

    hip_names = {
        'FL': 'hip_0',
        'FR': 'hip_1',
        'RL': 'hip_2',
        'RR': 'hip_3',
    }

    leg_num = {
        'FL': 0,  # Front Left
        'FR': 1,  # Front Right
        'RL': 2,  # Hind Left
        'RR': 3,  # Hind Right
    }

    toe_names_geom = {
        'FL': 'toe_0',
        'FR': 'toe_1',
        'RL': 'toe_2',
        'RR': 'toe_3',
    }



# Number of joints per limb
robot_params['joints_per_limb'] = int(robot_params['num_joints'] / robot_params['limbs'])

height_map_params = {
    "resolution_vfa": 0.04,
    "dimension_vfa": 7
}

# Gait parameters (for A1)
sim_param = {
    "visual_foothold_adaptation": "height",  # 'blind', 'vfa', 'height'
    "resolution_vfa": 0.04,  # Resolution of the visual foothold adaptation
    "dimension_vfa": 7,  # Dimension of the visual foothold adaptation
    'gait':                        'crawl',  # 'trot', 'pace', 'crawl', 'bound', 'full_stance'
    'gait_params':                 {'trot': {'step_freq': 1.4, 'duty_factor': 0.65, 'type': GaitType.TROT.value},
                                    'crawl': {'step_freq': 1.0, 'duty_factor': 0.8, 'type': GaitType.BACKDIAGONALCRAWL.value},
                                    'pace': {'step_freq': 1.4, 'duty_factor': 0.7, 'type': GaitType.PACE.value},
                                    'bound': {'step_freq': 1.8, 'duty_factor': 0.65, 'type': GaitType.BOUNDING.value},
                                    'full_stance': {'step_freq': 2, 'duty_factor': 0.65, 'type': GaitType.FULL_STANCE.value},
                                   },
    'controller': "DefaultController",  # 'DefaultController', 'TerrainAwareController', 'FootEndAvoidanceController', 'NMPCController', 'MPCControllerWithFootholdOptimization', 'RaibertHeuristicController', 'WholeBodyController'
    'vel_control': False,  # use velocity control (default True for RaibertHeuristicController)
    'max_step_height': 0.3 * hip_height,  # maximum step height for the robot
    # 'max_step_length': 0.08,  # maximum step length for the robot
    # maximum step width for the robot
    'min_jerk': False,  # use minimum jerk trajectory for the swing phase
    'obs_clearance': 0.01,  # clearance from the obstacles
    'support_polygon': False,  # use support polygon for the robot
    "retraction": 0.0001,
    "foothold_radius": 0.02,  # radius of the foothold (FootEndAvoidanceController)
    "min_step_distance": 0.05,  # minimum step distance for the robot (FootEndAvoidanceController),
    "foothold_alpha": 1.0,  # alpha for the foothold heuristic score (FootEndAvoidanceController)
    'mpc_frequency': 1000,
    'use_inertia_recomputation': True,
    'stand_still_time': 1.0, # time to stand still at the start of the simulation(s)
    'swing_generator':             'scipy',  # 'scipy', 'explicit'
    'swing_position_gain_fb':      500,
    'swing_velocity_gain_fb':      10,
    'ref_z':                       hip_height,
    # This is used to activate or deactivate the reflexes upon contact detection
    'reflex_trigger_mode':       'geom_contact', # 'tracking', 'geom_contact', False
    'reflex_height_enhancement': True,
    'velocity_modulator': True,
}

if(sim_param['controller'] == 'DefaultController'):
    sim_param['swing_generator'] = 'scipy'
    sim_param['swing_position_gain_fb'] = 10
    sim_param['swing_velocity_gain_fb'] = 0


robot_limb_phase = get_gait_phase(sim_param)

gravity_constant = 9.81  # m/s^2, standard gravity constant
# MPC Parameters
mpc_params = {
    # 'nominal' optimized directly the GRF
    # 'input_rates' optimizes the delta GRF
    # 'sampling' is a gpu-based mpc that samples the GRF
    # 'collaborative' optimized directly the GRF and has a passive arm model inside
    # 'lyapunov' optimized directly the GRF and has a Lyapunov-based stability constraint
    # 'kinodynamic' sbrd with joints - experimental
    'type':                                    'nominal',

    # print the mpc info
    'verbose':                                 False,

    # horizon is the number of timesteps in the future that the mpc will optimize
    # dt is the discretization time used in the mpc
    'horizon':                                 18,
    'dt':                                      0.02,

    # GRF limits for each single leg
    "grf_max":                                 mass * gravity_constant,
    "grf_min":                                 0,
    'mu':                                      0.5,

    # this is used to have a smaller dt near the start of the horizon
    'use_nonuniform_discretization':           False,
    'horizon_fine_grained':                    2,
    'dt_fine_grained':                         0.01,

    # if this is true, we optimize the step frequency as well
    # for the sampling controller, this is done in the rollout
    # for the gradient-based controller, this is done with a batched version of the ocp
    'optimize_step_freq':                      True,
    'step_freq_available':                     [1.4, 1.6, 1.8, 2.0, 2.2, 2.4],

    # ----- START properties only for the gradient-based mpc -----

    # this is used if you want to manually warm start the mpc
    'use_warm_start':                          True,

    # this enables integrators for height, linear velocities, roll and pitch
    'use_integrators':                         True,
    'alpha_integrator':                        0.1,
    'integrator_cap':                          [0.5, 0.2, 0.2, 0.0, 0.0, 1.0],

    # if this is off, the mpc will not optimize the footholds and will
    # use only the ones provided in the reference
    'use_foothold_optimization':               True,

    # this is set to false automatically is use_foothold_optimization is false
    # because in that case we cannot chose the footholds and foothold
    # constraints do not any make sense
    'use_foothold_constraints':                True,

    # works with all the mpc types except 'sampling'. In sim does not do much for now,
    # but in real it minizimes the delay between the mpc control and the state
    'use_RTI':                                 False,
    # If RTI is used, we can set the advance RTI-step! (Standard is the simpler RTI)
    # See https://arxiv.org/pdf/2403.07101.pdf
    'as_rti_type':                             "Standard",  # "AS-RTI-A", "AS-RTI-B", "AS-RTI-C", "AS-RTI-D", "Standard"
    'as_rti_iter':                             1,  # > 0, the higher the better, but slower computation!

    # This will force to use DDP instead of SQP, based on https://arxiv.org/abs/2403.10115.
    # Note that RTI is not compatible with DDP, and no state costraints for now are considered
    'use_DDP':                                 False,

    # this is used only in the case 'use_RTI' is false in a single mpc feedback loop.
    # More is better, but slower computation!
    'num_qp_iterations':                       2,

    # this is used to speeding up or robustify acados' solver (hpipm).
    'solver_mode':                             'balance',  # balance, robust, speed, crazy_speed


    # these is used only for the case 'input_rates', using as GRF not the actual state
    # of the robot of the predicted one. Can be activated to compensate
    # for the delay in the control loop on the real robot
    'use_input_prediction':                    False,

    # ONLY ONE CAN BE TRUE AT A TIME (only gradient)
    'use_static_stability':                    False,
    'use_zmp_stability':                       False,
    'trot_stability_margin':                   0.04,
    'pace_stability_margin':                   0.1,
    'crawl_stability_margin':                  0.04,  # in general, 0.02 is a good value

    # this is used to compensate for the external wrenches
    # you should provide explicitly this value in compute_control
    'external_wrenches_compensation':          True,
    'external_wrenches_compensation_num_step': 15,

    # this is used only in the case of collaborative mpc, to
    # compensate for the external wrench in the prediction (only collaborative)
    'passive_arm_compensation':                True,


    # Gain for Lyapunov-based MPC
    'K_z1': np.array([1, 1, 10]),
    'K_z2': np.array([1, 4, 10]),
    'residual_dynamics_upper_bound': 30,
    'use_residual_dynamics_decay': False,


    }