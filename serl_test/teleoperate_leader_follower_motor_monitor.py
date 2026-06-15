# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple script to control a robot from teleoperation.

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
  }' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import csv
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    reachy2,
    so_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    reachy2_teleoperator,
    so_leader,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


MOTOR_STATUS_NAMES = [
    "Present_Position",
    "Present_Velocity",
    "Present_Load",
    "Present_Current",
]


def try_sync_read(bus, data_name: str) -> dict[str, float]:
    """Read one motor status register from a Feetech/Dynamixel bus."""
    try:
        values = bus.sync_read(data_name)
    except TypeError:
        values = bus.sync_read(data_name=data_name)

    if values is None:
        return {}

    if isinstance(values, dict):
        return {str(k): float(v) for k, v in values.items()}

    motor_names = list(bus.motors.keys()) if hasattr(bus, "motors") else []

    if motor_names and hasattr(values, "__len__") and len(values) == len(motor_names):
        return {name: float(v) for name, v in zip(motor_names, values)}

    if hasattr(values, "__len__"):
        return {f"motor_{i}": float(v) for i, v in enumerate(values)}

    return {"value": float(values)}


def read_all_motor_status(bus) -> dict[str, dict[str, float]]:
    status = {}

    for data_name in MOTOR_STATUS_NAMES:
        try:
            status[data_name] = try_sync_read(bus, data_name)
        except Exception:
            status[data_name] = {}

    return status


def flatten_motor_status(
    prefix: str,
    status: dict[str, dict[str, float]],
) -> dict[str, float]:
    row = {}

    for data_name, values in status.items():
        for motor_name, value in values.items():
            row[f"{prefix}.{data_name}.{motor_name}"] = float(value)

    return row

@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig

    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None

    # Display all cameras on screen
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False

    # Motor status logging
    motor_log_path: str | None = None
    motor_log_every_steps: int = 3
    motor_print_every_steps: int = 30

def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
    motor_log_path: str | None = None,
    motor_log_every_steps: int = 3,
    motor_print_every_steps: int = 30,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        display_compressed_images: If True, compresses images before sending them to Rerun for display.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
    """

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    step = 0

    motor_log_file = None
    motor_log_writer = None

    if motor_log_path:
        motor_log_path = Path(motor_log_path)
        motor_log_path.parent.mkdir(parents=True, exist_ok=True)
        motor_log_file = motor_log_path.open("w", newline="")
        print(f"[INFO] Motor log path: {motor_log_path}")

    try:
        while True:
            loop_start = time.perf_counter()

            # Get robot observation
            obs = robot.get_observation()

            # Get teleop action
            raw_action = teleop.get_action()

            # Process teleop action through pipeline
            teleop_action = teleop_action_processor((raw_action, obs))

            # Process action for robot through pipeline
            robot_action_to_send = robot_action_processor((teleop_action, obs))

            # Send processed action to robot
            _ = robot.send_action(robot_action_to_send)

            # Motor status logging
            if motor_log_file is not None and step % motor_log_every_steps == 0:
                timestamp = time.perf_counter() - start

                leader_status = read_all_motor_status(teleop.bus)
                follower_status = read_all_motor_status(robot.bus)

                row = {
                    "timestamp": float(timestamp),
                    "step": int(step),
                }

                row.update(flatten_motor_status("leader", leader_status))
                row.update(flatten_motor_status("follower", follower_status))

                if motor_log_writer is None:
                    motor_log_writer = csv.DictWriter(
                        motor_log_file,
                        fieldnames=list(row.keys()),
                    )
                    motor_log_writer.writeheader()

                motor_log_writer.writerow(row)
                motor_log_file.flush()

                if step % motor_print_every_steps == 0:
                    leader_load = leader_status.get("Present_Load", {})
                    follower_load = follower_status.get("Present_Load", {})

                    print("")
                    print(f"[Motor load step={step}]")
                    print("leader load :", leader_load)
                    print("follower load:", follower_load)

            if display_data:
                # Process robot observation through pipeline
                obs_transition = robot_observation_processor(obs)

                log_rerun_data(
                    observation=obs_transition,
                    action=teleop_action,
                    compress_images=display_compressed_images,
                )

                print("\n" + "-" * (display_len + 10))
                print(f"{'NAME':<{display_len}} | {'NORM':>7}")

                for motor, value in robot_action_to_send.items():
                    print(f"{motor:<{display_len}} | {value:>7.2f}")

                move_cursor_up(len(robot_action_to_send) + 3)

            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(1 / fps - dt_s, 0.0))
            loop_s = time.perf_counter() - loop_start

            print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            move_cursor_up(1)

            step += 1

            if duration is not None and time.perf_counter() - start >= duration:
                return

    finally:
        if motor_log_file is not None:
            motor_log_file.close()
            print("[INFO] Motor log saved.")

@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    teleop.connect()
    robot.connect()

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
            motor_log_path=cfg.motor_log_path,
            motor_log_every_steps=cfg.motor_log_every_steps,
            motor_print_every_steps=cfg.motor_print_every_steps,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
