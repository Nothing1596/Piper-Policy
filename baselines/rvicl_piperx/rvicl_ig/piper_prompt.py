"""Hardware-fact substitutions only; retain the frozen recursive video protocol."""
from .agent import PROMPT


def hardware_parameters(settings):
    """Task-independent facts from the same runtime modules used by execution."""
    from piperx_middleware.kinematics import (
        PIPERX_MDH, POSITION_TOLERANCE_M, ORIENTATION_TOLERANCE_DEG)
    from piperx_middleware.models import JOINT_LIMITS_DEG
    return {
        "model": "PiperX", "arm_count": 1, "arm": "left", "arm_dof": 6,
        "joint_order": [f"joint{i}" for i in range(1, 7)],
        "joint_types": ["revolute"] * 6,
        "joint_limits_deg": [list(pair) for pair in JOINT_LIMITS_DEG],
        "position_frame": "base=world", "position_unit": "metre",
        "quaternion": "wxyz; TCP-local to base rotation; q and -q are identical rotations",
        "joint_state_unit": "radian",
        "tcp_axes": {"forward": "+Z from flange to fingertips", "jaw_open_close": "X", "handedness": "right-handed"},
        "flange_to_tcp": {"translation_m": list(settings.tcp_offset_m),
                          "rpy_extrinsic_xyz_deg": list(settings.tcp_offset_rpy_deg)},
        "gripper": {"configured_width_range_m": [0., settings.gripper_max_m],
                    "width_definition": "API total-opening convention; simulation sums the two finger slide coordinates, not per-finger displacement",
                    "normalized_command": "0=closed, 1=configured maximum; null keeps current command"},
        "kinematics": {"model": "modified_DH", "columns": ["d_m", "a_m", "alpha_rad", "theta_offset_rad"],
            "rows": [list(row) for row in PIPERX_MDH],
            "transform_order": "T_base_tcp=product_i(Rx(alpha_i) Tx(a_i) Rz(q_i+theta_offset_i) Tz(d_i)) T_flange_tcp",
            "source": "piperx_middleware.kinematics.PIPERX_MDH"},
        "ik": {"method": "finite_multistart_least_squares_lm", "orientation_search": False,
               "target": "one requested xyz AND orientation, both fixed during solve",
               "position_tolerance_m": POSITION_TOLERANCE_M,
               "orientation_tolerance_deg": ORIENTATION_TOLERANCE_DEG,
               "tolerance_meaning": "numerical convergence, not calibrated physical accuracy",
               "collision_checking": False, "complete_reachability_test": False},
        "parameter_sources": {"joint_limits": "piperx_middleware.models.JOINT_LIMITS_DEG",
                              "tcp_and_gripper": "active isolated executor Settings"},
    }


def piper_prompt():
    prompt = PROMPT.replace("You control two Panda arms in The Imitator Game.",
        "You control one six-joint PiperX arm in a MuJoCo simulation.")
    prompt = prompt.replace("Panda grippers", "PiperX grippers").replace("WHAT THE THREE VIEWS", "WHAT THE TWO VIEWS")
    start, end = prompt.index("ROBOT INTERFACE\n"), prompt.index("IMAGE EVIDENCE (")
    prompt = prompt[:start] + """ROBOT INTERFACE
There is ONE arm. Keep the upstream arm field and always use arm="left"; there is no right arm or move_both.
head is the fixed camera; left_wrist is the sole arm's wrist camera. Neither name depends on where the arm appears in an image.
move_to, rotate_wrist, set_gripper and release inherit the frozen RPent primitive bodies. xyz is metres in the robot BASE frame (the simulation world frame); quat is wxyz. The TCP is the simulated fingertip centre, using the executor's declared flange-to-TCP transform. quat=null preserves orientation. rotate_wrist is a WORLD-Z yaw rotation at fixed TCP position, not a J6 joint increment.
Piper's existing move_to resolver and numerical IK replace Panda's planner. Motion has the executor's JOINT-SPACE ENDPOINT semantics, not a guaranteed Cartesian straight line or collision-certified trajectory. The adapter linearly samples that six-joint path at up to 2 degrees per joint before upstream waypoint selection; the existing executor supplies the physical joint control and measured feedback. Internal qpos paths are radians; the executor accepts degrees. substeps=0 retains the full path, 25 is the upstream waypoint cap, and 1 selects only the final waypoint. Tool-local +Z runs from flange towards fingertips; the gap opens/closes along tool-local X. The TCP axes rotate with the tool and are not fixed base axes.
set_gripper val=0 closes and val=1 opens to the executor's configured maximum width. Values describe commanded opening, not guaranteed grasp or calibrated force. Check the actual width, scene and a small lift. wait holds for 50 ms per requested step, up to 50 steps. The step budget counts accepted actuator waypoints/holds, not physical simulation ticks. Physics continues while the model reasons.
View fixed and wrist RGBD after actions. Cameras are captured sequentially with their own source timestamps, depth and capture-aligned calibration; they are not synchronized. Stored back_project measurements refer to the named capture, not automatically to the latest physical scene. Refresh after movement or delay. No VLA, hidden object pose or reward is exposed to the planner.
state.robot_parameters contains the active limits, MDH link parameters, flange-to-TCP transform and configured gripper range. state.arms.left.tcp_axes_world gives the current TCP basis vectors in base coordinates, and joints_deg and tcp_rpy_extrinsic_xyz_deg are alternate units/coordinates of the same measured configuration. RPY uses Rz(yaw) Ry(pitch) Rx(roll); quaternion sign reversal does not change orientation.
move_to.xyz is the TCP target, not the flange target; the executor already applies the flange-to-TCP transform.
state.last_ik_feedback records the most recent requested pose, measured joint seed and actual numerical solver outcome. IK fixes BOTH position and orientation; it does not search alternative orientations. An ik_no_solution result is not a proof that a position or all orientations there are unreachable. The inherited generic plan_failed hint may mention collision; Piper's numerical IK does not test collision, so that hint is not a collision finding. No candidate replacement pose is supplied.

""" + prompt[end:]
    start, end = prompt.index("IMAGE EVIDENCE ("), prompt.index("SAM PERCEPTION\n")
    prompt = prompt[:start] + """IMAGE EVIDENCE (READ THE LABEL IMMEDIATELY BEFORE EACH IMAGE)
Current observations are the fixed head and single left_wrist camera batch, each with its own capture timestamp and simulated time. Execution-process images, when present, are measured historical start/intermediate/end captures. They are never current observations. Demonstration images are recorded video evidence. Use current RGBD to decide the next action; retrieve demonstration clips for strategy. Older labels do not imply image bytes are still in the model request.

""" + prompt[end:]
    prompt = prompt.replace("Cameras are head, left_wrist and right_wrist", "Cameras are head and left_wrist")
    prompt = prompt.replace("The independent evaluator holds the last joint target through the remaining official horizon and computes metrics.",
        "The independent simulation evaluator holds briefly for settling and records the scene's own metrics; this adapter is not an official IG or LIBERO score.")
    return prompt
