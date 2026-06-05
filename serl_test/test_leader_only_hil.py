# test_leader_only_hil_with_act_policy.py

import time
from enum import Enum, auto
import sys
import termios
import tty
import select
from typing import Any

from lerobot.envs.configs import HILSerlProcessorConfig, HILSerlRobotEnvConfig
from lerobot.rl.gym_manipulator import make_robot_env

from lerobot.robots.so101_follower import SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101LeaderConfig
from lerobot.utils.robot_utils import busy_wait

# Camera config
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

# Policy / dataset utilities used by official record_loop
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device


class ControlMode(Enum):
    POLICY = auto()
    PAUSED_FOR_ALIGNMENT = auto()
    TELEOP_INTERVENTION = auto()
    PAUSED_AFTER_INTERVENTION = auto()


class NonBlockingKeyboard:
    """Non-blocking single-key reader for Linux terminal."""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = None

    def __enter__(self):
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def get_key(self) -> str | None:
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            return sys.stdin.read(1)
        return None


class RelativeLeaderActionMapper:
    """Convert absolute leader joint positions into follower-relative joint targets."""

    def __init__(self, gripper_mode: str = "absolute"):
        self.leader_ref = None
        self.follower_ref = None
        self.gripper_mode = gripper_mode

    def reset(
        self,
        leader_action: dict[str, float],
        follower_pos: dict[str, float],
    ) -> None:
        self.leader_ref = {key: float(value) for key, value in leader_action.items()}
        self.follower_ref = {key: float(value) for key, value in follower_pos.items()}

    def __call__(
        self,
        leader_action: dict[str, float],
        motor_names: list[str],
    ) -> dict[str, float]:
        if self.leader_ref is None or self.follower_ref is None:
            raise RuntimeError("RelativeLeaderActionMapper must be reset before use.")

        target = {}

        for name in motor_names:
            key = f"{name}.pos"

            if name == "gripper" and self.gripper_mode == "absolute":
                target[key] = float(leader_action[key])
                continue

            leader_delta = float(leader_action[key]) - float(self.leader_ref[key])
            target[key] = float(self.follower_ref[key]) + leader_delta

        return target


def get_hold_action(env, motor_names: list[str]) -> dict[str, float]:
    """Return current follower joint positions as a hold action."""
    current_pos = env.get_raw_joint_positions()

    return {
        f"{name}.pos": float(current_pos[f"{name}.pos"])
        for name in motor_names
    }


def handle_keyboard_transition(
    key: str | None,
    mode: ControlMode,
    relative_mapper: RelativeLeaderActionMapper,
    teleop_device,
    env,
) -> tuple[ControlMode, bool]:
    """Update control mode based on keyboard input.

    Returns:
        new_mode, should_quit
    """
    if key is None:
        return mode, False

    if key == "q":
        print("Quit requested.")
        return mode, True

    if mode == ControlMode.POLICY:
        if key == "p":
            print("POLICY -> PAUSED_FOR_ALIGNMENT")
            print("Manually align leader near follower, then press 's'.")
            return ControlMode.PAUSED_FOR_ALIGNMENT, False

    elif mode == ControlMode.PAUSED_FOR_ALIGNMENT:
        if key == "s":
            leader_ref = teleop_device.get_action()
            follower_ref = env.get_raw_joint_positions()

            relative_mapper.reset(
                leader_action=leader_ref,
                follower_pos=follower_ref,
            )

            print("PAUSED_FOR_ALIGNMENT -> TELEOP_INTERVENTION")
            print("Relative mapper reset.")
            return ControlMode.TELEOP_INTERVENTION, False

    elif mode == ControlMode.TELEOP_INTERVENTION:
        if key == "e":
            print("TELEOP_INTERVENTION -> PAUSED_AFTER_INTERVENTION")
            print("Press 'r' to resume policy.")
            return ControlMode.PAUSED_AFTER_INTERVENTION, False

    elif mode == ControlMode.PAUSED_AFTER_INTERVENTION:
        if key == "r":
            print("PAUSED_AFTER_INTERVENTION -> POLICY")
            return ControlMode.POLICY, False

    return mode, False

def convert_hil_obs_to_dataset_values(obs: dict[str, Any]) -> dict[str, Any]:
    """Convert HIL env observation to the format expected by build_dataset_frame.

    HIL env obs:
        obs["agent_pos"]
        obs["pixels"]["front"]
        obs["pixels"]["side"]

    build_dataset_frame expects:
        values["agent_pos"]
        values["front"]
        values["side"]
    """
    values = {}

    # State
    values["agent_pos"] = obs["agent_pos"]

    # Images
    values["front"] = obs["pixels"]["front"]
    values["side"] = obs["pixels"]["side"]

    # Optional: keep raw joint keys too, if present
    for key, value in obs.items():
        if key not in ["pixels"]:
            values[key] = value

    return values


def predict_policy_robot_action(
    obs: dict[str, Any],
    *,
    dataset: LeRobotDataset,
    policy,
    preprocessor,
    postprocessor,
    single_task: str,
    robot_type: str,
):
    """Run ACT policy using the same path as official record_loop."""

    obs_for_frame = convert_hil_obs_to_dataset_values(obs)

    observation_frame = build_dataset_frame(
        dataset.features,
        obs_for_frame,
        prefix=OBS_STR,
    )


    action_values = predict_action(
        observation=observation_frame,
        policy=policy,
        device=get_safe_torch_device(policy.config.device),
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=policy.config.use_amp,
        task=single_task,
        robot_type=robot_type,
    )

    robot_action = make_robot_action(
        action_values,
        dataset.features,
    )

    return robot_action

# ====== user settings ======

follower_port = "/dev/ttyACM1"
leader_port = "/dev/ttyACM0"

follower_id = "so101_follower"
leader_id = "so101_leader"

fps = 30

# Cameras used during recording/training
front_camera_index = 2
side_camera_index = 0

# Recorded dataset used for ACT training
dataset_repo_id = "local/pick_bin_and_place_01"

# Trained ACT policy path
policy_path = "outputs/train/act_pick_bin_and_place_01/checkpoints/last/pretrained_model"

# Must match dataset.single_task used during recording/training
single_task = "pick a bin and place it in circle"

# ===========================


robot_cfg = SO101FollowerConfig(
    port=follower_port,
    id=follower_id,
    cameras={
        "front": OpenCVCameraConfig(
            index_or_path=front_camera_index,
            width=640,
            height=480,
            fps=30,
        ),
        "side": OpenCVCameraConfig(
            index_or_path=side_camera_index,
            width=640,
            height=480,
            fps=30,
        ),
    },
)

teleop_cfg = SO101LeaderConfig(
    port=leader_port,
    id=leader_id,
)

processor_cfg = HILSerlProcessorConfig(
    control_mode="leader",
)

env_cfg = HILSerlRobotEnvConfig(
    robot=robot_cfg,
    teleop=teleop_cfg,
    processor=processor_cfg,
    fps=fps,
)

env, teleop_device = make_robot_env(env_cfg)

obs, info = env.reset()
motor_names = list(env.robot.bus.motors.keys())

print("motor_names:", motor_names)
print("initial obs keys:", obs.keys())
print("initial pixels keys:", obs.get("pixels", {}).keys())

# ====== Load dataset / policy / processors ======

print(f"Loading dataset: {dataset_repo_id}")
dataset = LeRobotDataset(dataset_repo_id)

print(f"Loading policy config: {policy_path}")
policy_cfg = PreTrainedConfig.from_pretrained(policy_path)

print("Creating policy...")
policy = make_policy(
    policy_cfg,
    ds_meta=dataset.meta,
)

print("Creating pre/post processors...")
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy_cfg,
    pretrained_path=policy_path,
    dataset_stats=dataset.meta.stats,
)

policy.to(policy.config.device)
policy.eval()

policy.reset()
preprocessor.reset()
postprocessor.reset()

print(f"Loaded ACT policy from: {policy_path}")
print("policy device:", policy.config.device)
print("policy input features:", policy.config.input_features)
print("policy output features:", policy.config.output_features)

# ====== HIL state ======

mode = ControlMode.POLICY
relative_mapper = RelativeLeaderActionMapper(gripper_mode="absolute")

# Important:
# hold_action is initialized from current position only once.
# After that, pause transitions inherit last_commanded_action instead of reading current position.
hold_action = get_hold_action(env, motor_names)
last_commanded_action = hold_action.copy()

print("keyboard-controlled HIL + ACT policy test started")
print("Controls:")
print("  p: pause policy and enter leader-alignment mode")
print("  s: start teleop intervention")
print("  e: end teleop intervention")
print("  r: resume policy")
print("  q: quit")
print("Current POLICY mode runs ACT policy.")
print("Pause modes hold last commanded action.")

try:
    with NonBlockingKeyboard() as keyboard:
        i = 0

        while True:
            start = time.perf_counter()

            key = keyboard.get_key()
            prev_mode = mode

            mode, should_quit = handle_keyboard_transition(
                key=key,
                mode=mode,
                relative_mapper=relative_mapper,
                teleop_device=teleop_device,
                env=env,
            )

            # Handle mode transition side effects
            if mode != prev_mode:
                if mode == ControlMode.POLICY:
                    # Reset ACT chunk/cache when returning to policy
                    policy.reset()
                    preprocessor.reset()
                    postprocessor.reset()

                    # Do not read current joint positions here.
                    # Inherit last command to avoid gravity-induced target drift.
                    hold_action = last_commanded_action.copy()

                    print("Policy and processors reset.")
                    print("Hold action inherited from last command at POLICY entry.")

                elif mode in [
                    ControlMode.PAUSED_FOR_ALIGNMENT,
                    ControlMode.PAUSED_AFTER_INTERVENTION,
                ]:
                    # Do not read current joint positions here.
                    # Inherit last commanded target for seamless hold.
                    hold_action = last_commanded_action.copy()
                    print(f"Hold action inherited from last command: {mode.name}")

            if should_quit:
                break

            # ---- action selection by mode ----
            if mode == ControlMode.POLICY:
                action = predict_policy_robot_action(
                    obs,
                    dataset=dataset,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    single_task=single_task,
                    robot_type=env.robot.robot_type,
                )

            elif mode == ControlMode.PAUSED_FOR_ALIGNMENT:
                # Hold last commanded target while manually aligning leader.
                action = hold_action

            elif mode == ControlMode.TELEOP_INTERVENTION:
                # Control follower by relative leader displacement.
                leader_action = teleop_device.get_action()
                action = relative_mapper(
                    leader_action=leader_action,
                    motor_names=motor_names,
                )

            elif mode == ControlMode.PAUSED_AFTER_INTERVENTION:
                # Hold final intervention target before resuming policy.
                action = hold_action

            else:
                raise RuntimeError(f"Unknown mode: {mode}")

            # Save last command before sending it.
            # This is important for seamless pause/resume.
            last_commanded_action = action.copy()

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print("episode ended")
                break

            if i % fps == 0:
                print(f"step={i}, mode={mode.name}")
                if mode == ControlMode.POLICY:
                    print("policy_action:", action)

            i += 1

            busy_wait(1.0 / fps - (time.perf_counter() - start))

finally:
    env.close()

    if teleop_device is not None and getattr(teleop_device, "is_connected", False):
        teleop_device.disconnect()

    print("closed")