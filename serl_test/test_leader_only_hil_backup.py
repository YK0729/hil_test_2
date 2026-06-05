# test_leader_only_hil.py

import time
from enum import Enum, auto
import sys
import termios
import tty
import select

from lerobot.envs.configs import HILSerlProcessorConfig, HILSerlRobotEnvConfig
from lerobot.rl.gym_manipulator import make_robot_env
from lerobot.robots.so101_follower import SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101LeaderConfig
from lerobot.utils.robot_utils import busy_wait


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

# ====== user settings ======

follower_port = "/dev/ttyACM1"  # 実際の follower port に変更
leader_port = "/dev/ttyACM0"    # 実際の leader port に変更

follower_id = "so101_follower"
leader_id = "so101_leader"

fps = 10
num_steps = 250

# ===========================


robot_cfg = SO101FollowerConfig(
    port=follower_port,
    id=follower_id,
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

relative_mapper = RelativeLeaderActionMapper(
    gripper_mode="absolute",
)

mode = ControlMode.POLICY
relative_mapper = RelativeLeaderActionMapper(gripper_mode="absolute")

print("keyboard-controlled leader-only HIL test started")
print("Controls:")
print("  p: pause policy and enter leader-alignment mode")
print("  s: start teleop intervention")
print("  e: end teleop intervention")
print("  r: resume policy")
print("  q: quit")
print("Current POLICY mode uses hold action for now.")
print("motor_names:", motor_names)

try:
    with NonBlockingKeyboard() as keyboard:
        i = 0

        while True:
            start = time.perf_counter()

            key = keyboard.get_key()

            mode, should_quit = handle_keyboard_transition(
                key=key,
                mode=mode,
                relative_mapper=relative_mapper,
                teleop_device=teleop_device,
                env=env,
            )

            if should_quit:
                break

            # ---- action selection by mode ----
            if mode == ControlMode.POLICY:
                # まだ本物の policy は使わず hold
                action = get_hold_action(env, motor_names)

            elif mode == ControlMode.PAUSED_FOR_ALIGNMENT:
                # leader を手動で follower に近づける時間
                # follower はその場で保持
                action = get_hold_action(env, motor_names)

            elif mode == ControlMode.TELEOP_INTERVENTION:
                # leader の相対変化量で follower を操作
                leader_action = teleop_device.get_action()
                action = relative_mapper(
                    leader_action=leader_action,
                    motor_names=motor_names,
                )

            elif mode == ControlMode.PAUSED_AFTER_INTERVENTION:
                # policy 再開前の待機
                action = get_hold_action(env, motor_names)

            else:
                raise RuntimeError(f"Unknown mode: {mode}")

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print("episode ended")
                break

            if i % fps == 0:
                print(f"step={i}, mode={mode.name}")

            i += 1
            busy_wait(1.0 / fps - (time.perf_counter() - start))

finally:
    env.close()

    if teleop_device is not None and getattr(teleop_device, "is_connected", False):
        teleop_device.disconnect()

    print("closed")