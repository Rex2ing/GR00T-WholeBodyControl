# ELF3 robot configuration for IsaacLab training.
#
# ELF3 has 29 hinge joints + 1 floating base. The MuJoCo (MJCF) joint/body order
# is DFS, while IsaacLab indexes joints/bodies via BFS over the URDF tree. The
# two orderings differ — `ELF3_ISAACLAB_TO_MUJOCO_MAPPING` carries both index
# arrays so retargeted MuJoCo trajectories can be remapped into IsaacLab.
#
# Note: ELF3 has no `pelvis` link — its base / root body is `torso_link`, and
# the lower-body chain hangs off the waist sub-chain. Any reward / observation
# config that references G1's `pelvis`, `left_*_link`, `right_*_link` etc. must
# be overridden in the experiment yaml (e.g. `sonic_elf3.yaml`).
#
# PD gains (stiffness / damping) are taken directly from the real-robot
# deployment code (bxi_rl_controller_ros2_example/inference/amp.py) to ensure
# training and deployment use identical actuator models — matching the G1 approach
# where gains are embedded in the exported ONNX metadata.

import re as _re

from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

from gear_sonic.envs.manager_env.mdp.actuators import DelayedImplicitActuatorCfg

ASSET_DIR = "gear_sonic/data/assets"


# ---------------------------------------------------------------------------
# PD gains — from real-robot deployment code (bxi_rl_controller_ros2_example).
# Order matches ELF3_JOINT_NAMES (MuJoCo MJCF DFS order).
# These values were tuned on the physical ELF3 robot and should NOT be changed
# without re-tuning on hardware.
# ---------------------------------------------------------------------------
ELF3_DEPLOY_KP_MUJOCO = [
    108.448, 162.672, 176.421,   # waist_y, waist_x, waist_z
    176.421, 176.421,  54.224, 176.421,  33.493,  21.771,  # l: hip_y, hip_x, hip_z, knee, ankle_y, ankle_x
    176.421, 176.421,  54.224, 176.421,  33.493,  21.771,  # r: hip_y, hip_x, hip_z, knee, ankle_y, ankle_x
     54.224,  54.224,  16.747,  54.224,  16.747,  16.747,  16.747,  # l: sh_y, sh_x, sh_z, elbow, wr_x, wr_y, wr_z
     54.224,  54.224,  16.747,  54.224,  16.747,  16.747,  16.747,  # r: sh_y, sh_x, sh_z, elbow, wr_x, wr_y, wr_z
]

ELF3_DEPLOY_KD_MUJOCO = [
     6.904, 10.356, 11.231,     # waist_y, waist_x, waist_z
    11.231, 11.231,  3.452, 11.231,  2.132,  1.386,  # l: hip_y, hip_x, hip_z, knee, ankle_y, ankle_x
    11.231, 11.231,  3.452, 11.231,  2.132,  1.386,  # r: hip_y, hip_x, hip_z, knee, ankle_y, ankle_x
     3.452,  3.452,  1.066,  3.452,  1.066,  1.066,  1.066,  # l: sh_y, sh_x, sh_z, elbow, wr_x, wr_y, wr_z
     3.452,  3.452,  1.066,  3.452,  1.066,  1.066,  1.066,  # r: sh_y, sh_x, sh_z, elbow, wr_x, wr_y, wr_z
]

# Real-robot rated joint torque limits (N*m), from the deployment repository's
# ``elf3_control_constants.py::ELF3_EFFORT_LIMIT_MUJOCO`` and ELF3 URDF. These
# are physical actuator caps, not the synthetic effort values historically used
# to reconstruct SONIC's action scale.
ELF3_DEPLOY_EFFORT_LIMIT_MUJOCO = [
     90.0, 100.0, 150.0,  # waist_y, waist_x, waist_z
    150.0, 150.0,  45.0, 150.0, 50.0, 20.0,  # left leg
    150.0, 150.0,  45.0, 150.0, 50.0, 20.0,  # right leg
     45.0,  45.0,  21.0,  45.0, 21.0, 21.0, 21.0,  # left arm
     45.0,  45.0,  21.0,  45.0, 21.0, 21.0, 21.0,  # right arm
]

# SONIC action scale used by the real deployment controller. Keep this
# independent from the physical torque limits above: deployment intentionally
# uses this established policy ABI, and recomputing it from the rated effort
# caps would change the meaning of every policy action.
ELF3_SONIC_ACTION_SCALE_MUJOCO = [
    0.230525229, 0.153683486, 0.141706486,
    0.141706486, 0.141706486, 0.230525229, 0.212559729, 0.373212313, 0.229663314,
    0.141706486, 0.141706486, 0.230525229, 0.212559729, 0.373212313, 0.229663314,
    0.230525229, 0.230525229, 0.373201170, 0.230525229,
    0.373201170, 0.373201170, 0.373201170,
    0.230525229, 0.230525229, 0.373201170, 0.230525229,
    0.373201170, 0.373201170, 0.373201170,
]


# ---------------------------------------------------------------------------
# Joint / body order mappings.
# IsaacLab order is BFS over the URDF link tree (root = torso_link), computed
# from `gear_sonic/data/assets/robot_description/urdf/elf3/elf3.urdf`. The
# MuJoCo order is the DFS order in `mjcf/elf3.xml`.
# Following bxi.py pattern: define name lists and compute mappings via .index().
# ---------------------------------------------------------------------------

# IsaacLab body order (verified by running robot in IsaacLab container).
# Layer-order traversal: L-R-W (left-right-waist) per level, NOT W-L-R.
ELF3_ISAACLAB_JOINTS = [
    "torso_link",         # IL[ 0] (root)
    "l_shoulder_y_link",  # IL[ 1]
    "r_shoulder_y_link",  # IL[ 2]
    "waist_y_link",       # IL[ 3]
    "l_shoulder_x_link",  # IL[ 4]
    "r_shoulder_x_link",  # IL[ 5]
    "waist_x_link",       # IL[ 6]
    "l_shoulder_z_link",  # IL[ 7]
    "r_shoulder_z_link",  # IL[ 8]
    "waist_z_link",       # IL[ 9]
    "l_elbow_y_link",     # IL[10]
    "r_elbow_y_link",     # IL[11]
    "l_hip_y_link",       # IL[12]
    "r_hip_y_link",       # IL[13]
    "l_wrist_x_link",     # IL[14]
    "r_wrist_x_link",     # IL[15]
    "l_hip_x_link",       # IL[16]
    "r_hip_x_link",       # IL[17]
    "l_wrist_y_link",     # IL[18]
    "r_wrist_y_link",     # IL[19]
    "l_hip_z_link",       # IL[20]
    "r_hip_z_link",       # IL[21]
    "l_wrist_z_link",     # IL[22]
    "r_wrist_z_link",     # IL[23]
    "l_knee_y_link",      # IL[24]
    "r_knee_y_link",      # IL[25]
    "l_ankle_y_link",     # IL[26]
    "r_ankle_y_link",     # IL[27]
    "l_ankle_x_link",     # IL[28]
    "r_ankle_x_link",     # IL[29]
]

# MuJoCo body order is DFS over the MJCF tree (mjcf/elf3.xml).
ELF3_MUJOCO_BODY_NAMES = [
    "torso_link",         # MJ[ 0] (root)
    "waist_y_link",       # MJ[ 1]
    "waist_x_link",       # MJ[ 2]
    "waist_z_link",       # MJ[ 3]
    "l_hip_y_link",       # MJ[ 4]
    "l_hip_x_link",       # MJ[ 5]
    "l_hip_z_link",       # MJ[ 6]
    "l_knee_y_link",      # MJ[ 7]
    "l_ankle_y_link",     # MJ[ 8]
    "l_ankle_x_link",     # MJ[ 9]
    "r_hip_y_link",       # MJ[10]
    "r_hip_x_link",       # MJ[11]
    "r_hip_z_link",       # MJ[12]
    "r_knee_y_link",      # MJ[13]
    "r_ankle_y_link",     # MJ[14]
    "r_ankle_x_link",     # MJ[15]
    "l_shoulder_y_link",  # MJ[16]
    "l_shoulder_x_link",  # MJ[17]
    "l_shoulder_z_link",  # MJ[18]
    "l_elbow_y_link",     # MJ[19]
    "l_wrist_x_link",     # MJ[20]
    "l_wrist_y_link",     # MJ[21]
    "l_wrist_z_link",     # MJ[22]
    "r_shoulder_y_link",  # MJ[23]
    "r_shoulder_x_link",  # MJ[24]
    "r_shoulder_z_link",  # MJ[25]
    "r_elbow_y_link",     # MJ[26]
    "r_wrist_x_link",     # MJ[27]
    "r_wrist_y_link",     # MJ[28]
    "r_wrist_z_link",     # MJ[29]
]

# MuJoCo joint names in MJCF order (waist→l_leg→r_leg→l_arm→r_arm).
ELF3_JOINT_NAMES = [
    "waist_y_joint",      # MJ DOF[ 0]
    "waist_x_joint",      # MJ DOF[ 1]
    "waist_z_joint",      # MJ DOF[ 2]
    "l_hip_y_joint",      # MJ DOF[ 3]
    "l_hip_x_joint",      # MJ DOF[ 4]
    "l_hip_z_joint",      # MJ DOF[ 5]
    "l_knee_y_joint",     # MJ DOF[ 6]
    "l_ankle_y_joint",    # MJ DOF[ 7]
    "l_ankle_x_joint",    # MJ DOF[ 8]
    "r_hip_y_joint",      # MJ DOF[ 9]
    "r_hip_x_joint",      # MJ DOF[10]
    "r_hip_z_joint",      # MJ DOF[11]
    "r_knee_y_joint",     # MJ DOF[12]
    "r_ankle_y_joint",    # MJ DOF[13]
    "r_ankle_x_joint",    # MJ DOF[14]
    "l_shoulder_y_joint", # MJ DOF[15]
    "l_shoulder_x_joint", # MJ DOF[16]
    "l_shoulder_z_joint", # MJ DOF[17]
    "l_elbow_y_joint",    # MJ DOF[18]
    "l_wrist_x_joint",    # MJ DOF[19]
    "l_wrist_y_joint",    # MJ DOF[20]
    "l_wrist_z_joint",    # MJ DOF[21]
    "r_shoulder_y_joint", # MJ DOF[22]
    "r_shoulder_x_joint", # MJ DOF[23]
    "r_shoulder_z_joint", # MJ DOF[24]
    "r_elbow_y_joint",    # MJ DOF[25]
    "r_wrist_x_joint",    # MJ DOF[26]
    "r_wrist_y_joint",    # MJ DOF[27]
    "r_wrist_z_joint",    # MJ DOF[28]
]

# Keep the 12 leg-observation slots in left-leg / right-leg order.
ELF3_LOWER_JOINT_NAMES = [
    f"{side}_{joint}_joint"
    for side in ("l", "r")
    for joint in ("hip_y", "hip_x", "hip_z", "knee_y", "ankle_y", "ankle_x")
]

# The torso is the root, so waist motion also moves the leg chain. Only replace
# the arms during upper-body augmentation; preserve both the waist and legs.
ELF3_UPPER_BODY_AUGMENTATION_JOINT_NAMES = [
    f"{side}_{joint}_joint"
    for side in ("l", "r")
    for joint in ("shoulder_y", "shoulder_x", "shoulder_z", "elbow_y", "wrist_x", "wrist_y", "wrist_z")
]

ELF3_NATIVE_JOINT_NAMES: tuple[str, ...] = (
    "l_shoulder_y_joint",
    "r_shoulder_y_joint",
    "waist_y_joint",
    "l_shoulder_x_joint",
    "r_shoulder_x_joint",
    "waist_x_joint",
    "l_shoulder_z_joint",
    "r_shoulder_z_joint",
    "waist_z_joint",
    "l_elbow_y_joint",
    "r_elbow_y_joint",
    "l_hip_y_joint",
    "r_hip_y_joint",
    "l_wrist_x_joint",
    "r_wrist_x_joint",
    "l_hip_x_joint",
    "r_hip_x_joint",
    "l_wrist_y_joint",
    "r_wrist_y_joint",
    "l_hip_z_joint",
    "r_hip_z_joint",
    "l_wrist_z_joint",
    "r_wrist_z_joint",
    "l_knee_y_joint",
    "r_knee_y_joint",
    "l_ankle_y_joint",
    "r_ankle_y_joint",
    "l_ankle_x_joint",
    "r_ankle_x_joint",
)

# IsaacLab DOF names (verified by running robot in IsaacLab container).
# Layer-order traversal: L-R-W (left-right-waist) per level.
# Matches ELF3_ISAACLAB_JOINTS[1:] link-by-link.
ELF3_ISAACLAB_DOF_NAMES = ELF3_NATIVE_JOINT_NAMES

# Compute DOF mappings from name lists.
# Motion library uses: dof_pos[:, mujoco_to_isaaclab_dof] → reorders MuJoCo DFS to IsaacLab BFS.
# This requires mujoco_to_isaaclab_dof[isaaclab_pos] = mujoco_index.
ELF3_MUJOCO_TO_ISAACLAB_DOF = [
    ELF3_JOINT_NAMES.index(name) for name in ELF3_ISAACLAB_DOF_NAMES
]
# Inverse: isaaclab_to_mujoco_dof[mujoco_pos] = isaaclab_index.
ELF3_ISAACLAB_TO_MUJOCO_DOF = [
    ELF3_ISAACLAB_DOF_NAMES.index(name) for name in ELF3_JOINT_NAMES
]

# Compute body mappings from name lists.
# Motion library uses: body_pos[:, mujoco_to_isaaclab_body] → reorders MuJoCo DFS to IsaacLab BFS.
# This requires mujoco_to_isaaclab_body[isaaclab_pos] = mujoco_index.
ELF3_MUJOCO_TO_ISAACLAB_BODY = [
    ELF3_MUJOCO_BODY_NAMES.index(name) for name in ELF3_ISAACLAB_JOINTS
]
# Inverse: isaaclab_to_mujoco_body[mujoco_pos] = isaaclab_index.
ELF3_ISAACLAB_TO_MUJOCO_BODY = [
    ELF3_ISAACLAB_JOINTS.index(name) for name in ELF3_MUJOCO_BODY_NAMES
]

# Per-joint PD gains in IsaacLab BFS order (reordered from MuJoCo DFS).
ELF3_DEPLOY_KP_ISAACLAB = [ELF3_DEPLOY_KP_MUJOCO[ELF3_JOINT_NAMES.index(n)] for n in ELF3_ISAACLAB_DOF_NAMES]
ELF3_DEPLOY_KD_ISAACLAB = [ELF3_DEPLOY_KD_MUJOCO[ELF3_JOINT_NAMES.index(n)] for n in ELF3_ISAACLAB_DOF_NAMES]
ELF3_DEPLOY_EFFORT_LIMIT_ISAACLAB = [
    ELF3_DEPLOY_EFFORT_LIMIT_MUJOCO[ELF3_JOINT_NAMES.index(n)]
    for n in ELF3_ISAACLAB_DOF_NAMES
]


def _build_joint_value_dict(joint_names_expr, values):
    """Select a per-joint parameter vector using IsaacLab joint regexes."""
    expressions = joint_names_expr if isinstance(joint_names_expr, list) else [joint_names_expr]
    matched = []
    for expr in expressions:
        pattern = _re.compile("^" + expr + "$")
        for name in ELF3_ISAACLAB_DOF_NAMES:
            if pattern.match(name) and name not in matched:
                matched.append(name)
    return {
        name: value
        for name, value in zip(ELF3_ISAACLAB_DOF_NAMES, values)
        if name in matched
    }

# Build {joint_name: value} dicts for each actuator group.
def _build_pd_dict(joint_names_expr):
    """Build stiffness/damping dict for the joints matching the given expressions.
    
    IsaacLab uses regex patterns like '.*_hip_y_joint' to match joint names.
    This function converts those patterns to proper regex and builds per-joint dicts.
    """
    return (
        _build_joint_value_dict(joint_names_expr, ELF3_DEPLOY_KP_ISAACLAB),
        _build_joint_value_dict(joint_names_expr, ELF3_DEPLOY_KD_ISAACLAB),
    )


# Pre-build PD dicts for each actuator group.
_LEGS_JOINTS = [".*_hip_y_joint", ".*_hip_x_joint", ".*_hip_z_joint", ".*_knee_y_joint"]
_FEET_JOINTS = [".*_ankle_y_joint", ".*_ankle_x_joint"]
_WAIST_JOINTS = ["waist_y_joint", "waist_x_joint", "waist_z_joint"]
_ARMS_JOINTS = [".*_shoulder_y_joint", ".*_shoulder_x_joint", ".*_shoulder_z_joint", ".*_elbow_y_joint"]
_WRISTS_JOINTS = [".*_wrist_x_joint", ".*_wrist_y_joint", ".*_wrist_z_joint"]

_LEGS_KP, _LEGS_KD = _build_pd_dict(_LEGS_JOINTS)
_FEET_KP, _FEET_KD = _build_pd_dict(_FEET_JOINTS)
_WAIST_KP, _WAIST_KD = _build_pd_dict(_WAIST_JOINTS)
_ARMS_KP, _ARMS_KD = _build_pd_dict(_ARMS_JOINTS)
_WRISTS_KP, _WRISTS_KD = _build_pd_dict(_WRISTS_JOINTS)
_LEGS_EFFORT = _build_joint_value_dict(
    _LEGS_JOINTS, ELF3_DEPLOY_EFFORT_LIMIT_ISAACLAB
)
_FEET_EFFORT = _build_joint_value_dict(
    _FEET_JOINTS, ELF3_DEPLOY_EFFORT_LIMIT_ISAACLAB
)
_WAIST_EFFORT = _build_joint_value_dict(
    _WAIST_JOINTS, ELF3_DEPLOY_EFFORT_LIMIT_ISAACLAB
)
_ARMS_EFFORT = _build_joint_value_dict(
    _ARMS_JOINTS, ELF3_DEPLOY_EFFORT_LIMIT_ISAACLAB
)
_WRISTS_EFFORT = _build_joint_value_dict(
    _WRISTS_JOINTS, ELF3_DEPLOY_EFFORT_LIMIT_ISAACLAB
)

# Armature = 转子等效惯量。G1 recipe 的不变量是 kp = armature·ω²、kd = 2ζ·armature·ω
# （ω = 10 Hz、ζ = 2，见 robots/g1.py:10-27），G1 那边是先定电机型号的 armature 再算 kp/kd。
# ELF3 反过来：kp/kd 取自真机部署（部署 PD 必须等于训练 PD），所以 armature 只能反推。
# 2026-08-13 换成 kp/ω²，原先是 G1 的 Unitree 电机常数
_NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535  # 10 Hz，与 G1 一致
_DAMPING_RATIO = 2.0                        # ζ，与 G1 一致

# Provisional control-path delay for ELF3 sim-to-real A/B.  sim_dt=0.005 s, so
# 0-2 physics steps corresponds to 0-10 ms.  Replace these bounds with measured
# command-to-torque latency; they are not asserted as hardware ground truth.
ELF3_ACTUATOR_MIN_DELAY_STEPS = 0
ELF3_ACTUATOR_MAX_DELAY_STEPS = 2


def _armature_from_kp(kp_dict):
    """armature = kp / ω²，逐关节。返回 {关节名: armature}。"""
    return {name: kp / (_NATURAL_FREQ**2) for name, kp in kp_dict.items()}


_LEGS_ARMATURE = _armature_from_kp(_LEGS_KP)      # hip_y/x/knee 0.0446880, hip_z 0.0137351
_FEET_ARMATURE = _armature_from_kp(_FEET_KP)      # ankle_y 0.0084839, ankle_x 0.0055147
_WAIST_ARMATURE = _armature_from_kp(_WAIST_KP)    # waist_y 0.0274702, x 0.0412053, z 0.0446880
_ARMS_ARMATURE = _armature_from_kp(_ARMS_KP)      # shoulder_y/x+elbow 0.0137351, shoulder_z 0.0042421
_WRISTS_ARMATURE = _armature_from_kp(_WRISTS_KP)  # 0.0042421

ELF3_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": ELF3_ISAACLAB_JOINTS,
    # IsaacLab DOF names, aligned with the *_dof index arrays below. Carried in the
    # mapping so runtime guards can resolve an index back to a physical joint without
    # a spawned articulation (see commands.py _resolve_lower_body_indices).
    "isaaclab_dof_names": list(ELF3_ISAACLAB_DOF_NAMES),
    "isaaclab_to_mujoco_dof": ELF3_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": ELF3_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": ELF3_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": ELF3_MUJOCO_TO_ISAACLAB_BODY,
    # Leg observations use 12 DOFs (3..14), excluding the waist (0..2).
    "lower_joint_names_mujoco": ELF3_LOWER_JOINT_NAMES,
    "lower_joint_indices_mujoco": [
        ELF3_JOINT_NAMES.index(name) for name in ELF3_LOWER_JOINT_NAMES
    ],
    # Augmentation replaces only the 14 arm DOFs (15..28).
    "upper_body_augmentation_indices_mujoco": [
        ELF3_JOINT_NAMES.index(name) for name in ELF3_UPPER_BODY_AUGMENTATION_JOINT_NAMES
    ],
}


# ---------------------------------------------------------------------------
# Articulation config.
# ---------------------------------------------------------------------------

ELF3_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/robot_description/urdf/elf3/elf3.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0, damping=0.0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # torso_link is the root; spawn 高度取「脚底刚好落地」。
        # FK 实测默认屈膝姿下脚底 = torso − 1.0349 m（MJCF 足底胶囊 −1.0339），故取 1.035；
        # 原值 1.0 会让生成时脚底穿地 3.5 cm。对比 G1：robots/g1.py:224 pos=(0,0,0.76)
        # 同样是它自己的站立 anchor 高度（其 MJCF 实测 0.7570，0.76 是取整 —— 拿这两个数
        # 做跨机器人比例折算时要用实测值，见 sonic_elf3.yaml 的 root_height_threshold）。
        # 注：旧注释把 1.0 说成「= 部署 kStandHeight」是混淆 —— remote_controller 的
        # kStandHeight 是遥控器目标基座高度指令，与脚底几何无关，两件事。
        # 默认姿态对齐真机部署 bxi_rl_controller_ros2_example/.../inference/amp.py
        # 的 default_dof_pos(膝34°, 脚平躯干竖直)。原值(膝26°)与真机不一致, 已改正。
        pos=(0.0, 0.0, 1.035),
        joint_pos={
            ".*_hip_y_joint": -0.30,
            ".*_knee_y_joint": 0.60,
            ".*_ankle_y_joint": -0.30,
            ".*_shoulder_y_joint": 0.20,
            ".*_elbow_y_joint": 0.60,
            "l_shoulder_x_joint": 0.20,
            "r_shoulder_x_joint": -0.20,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_y_joint",
                ".*_hip_x_joint",
                ".*_hip_z_joint",
                ".*_knee_y_joint",
            ],
            effort_limit_sim=_LEGS_EFFORT,
            velocity_limit_sim={
                ".*_hip_y_joint": 20.0,
                ".*_hip_x_joint": 20.0,
                ".*_hip_z_joint": 20.0,
                ".*_knee_y_joint": 20.0,
            },
            stiffness=_LEGS_KP,
            damping=_LEGS_KD,
            armature=_LEGS_ARMATURE,
            min_delay=ELF3_ACTUATOR_MIN_DELAY_STEPS,
            max_delay=ELF3_ACTUATOR_MAX_DELAY_STEPS,
        ),
        "feet": DelayedImplicitActuatorCfg(
            effort_limit_sim=_FEET_EFFORT,
            velocity_limit_sim=20.0,
            joint_names_expr=[".*_ankle_y_joint", ".*_ankle_x_joint"],
            stiffness=_FEET_KP,
            damping=_FEET_KD,
            armature=_FEET_ARMATURE,
            min_delay=ELF3_ACTUATOR_MIN_DELAY_STEPS,
            max_delay=ELF3_ACTUATOR_MAX_DELAY_STEPS,
        ),
        "waist": DelayedImplicitActuatorCfg(
            effort_limit_sim=_WAIST_EFFORT,
            velocity_limit_sim=20.0,
            joint_names_expr=["waist_y_joint", "waist_x_joint", "waist_z_joint"],
            stiffness=_WAIST_KP,
            damping=_WAIST_KD,
            armature=_WAIST_ARMATURE,
            min_delay=ELF3_ACTUATOR_MIN_DELAY_STEPS,
            max_delay=ELF3_ACTUATOR_MAX_DELAY_STEPS,
        ),
        "arms": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_y_joint",
                ".*_shoulder_x_joint",
                ".*_shoulder_z_joint",
                ".*_elbow_y_joint",
            ],
            effort_limit_sim=_ARMS_EFFORT,
            velocity_limit_sim={
                ".*_shoulder_y_joint": 20.0,
                ".*_shoulder_x_joint": 20.0,
                ".*_shoulder_z_joint": 20.0,
                ".*_elbow_y_joint": 20.0,
            },
            stiffness=_ARMS_KP,
            damping=_ARMS_KD,
            armature=_ARMS_ARMATURE,
            min_delay=ELF3_ACTUATOR_MIN_DELAY_STEPS,
            max_delay=ELF3_ACTUATOR_MAX_DELAY_STEPS,
        ),
        "wrists": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_x_joint",
                ".*_wrist_y_joint",
                ".*_wrist_z_joint",
            ],
            effort_limit_sim=_WRISTS_EFFORT,
            velocity_limit_sim=20.0,
            stiffness=_WRISTS_KP,
            damping=_WRISTS_KD,
            armature=_WRISTS_ARMATURE,
            min_delay=ELF3_ACTUATOR_MIN_DELAY_STEPS,
            max_delay=ELF3_ACTUATOR_MAX_DELAY_STEPS,
        ),
    },
)


# ---------------------------------------------------------------------------
# Action scale is a policy/deployment ABI, not the physical torque limit.
#
# The deployment controller sends ELF3_SONIC_ACTION_SCALE_MUJOCO verbatim. It
# intentionally differs from ``0.25 * rated_effort / kp`` on 22/29 joints. The
# actuator ``effort_limit_sim`` values above therefore use the real ELF3 caps,
# while this mapping preserves the action semantics expected by SONIC weights.
# ---------------------------------------------------------------------------
if len(ELF3_SONIC_ACTION_SCALE_MUJOCO) != len(ELF3_JOINT_NAMES):
    raise RuntimeError("ELF3 action-scale table must match the joint-name table")

ELF3_ACTION_SCALE = dict(zip(ELF3_JOINT_NAMES, ELF3_SONIC_ACTION_SCALE_MUJOCO))
