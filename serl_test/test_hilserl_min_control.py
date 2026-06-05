# test_hilserl_min_control.py

import time
import torch

from lerobot.envs.configs import HILSerlProcessorConfig, HILSerlRobotEnvConfig
from lerobot.rl.gym_manipulator import (
    make_robot_env,
    make_processors,
    step_env_and_process_transition,
)
from lerobot.processor import create_transition, TransitionKey
from lerobot.robots.so101_follower import SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101LeaderConfig
from lerobot.teleoperators.keyboard import KeyboardEndEffectorTeleopConfig
from lerobot.utils.robot_utils import busy_wait

follower_port = "/dev/ttyACM0"  # 実際の値に変更
leader_port = "/dev/ttyACM1"    # 実際の値に変更

follower_id = "so101_follower"
leader_id = "so101_leader"

fps = 10
device = "cpu"

robot_cfg = SO101FollowerConfig(port=follower_port, id=follower_id)
teleop_cfg = SO101LeaderConfig(port=leader_port, id=leader_id)

processor_cfg = HILSerlProcessorConfig(control_mode="leader")

env_cfg = HILSerlRobotEnvConfig(
    robot=robot_cfg,
    teleop=teleop_cfg,
    processor=processor_cfg,
    fps=fps,
)

env, teleop_device = make_robot_env(env_cfg)
env_processor, action_processor = make_processors(
    env=env,
    teleop_device=teleop_device,
    cfg=env_cfg,
    device=device,
)

obs, info = env.reset()
env_processor.reset()
action_processor.reset()

transition = create_transition(observation=obs, info=info)
transition = env_processor(transition)

use_gripper = (
    env_cfg.processor.gripper.use_gripper
    if env_cfg.processor.gripper is not None
    else True
)

print("start minimal HIL control")
print("Move leader / press intervention key according to HIL-SERL settings.")
print("Ctrl+C to stop.")

try:
    for i in range(100):
        start = time.perf_counter()

        neutral_action = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        if use_gripper:
            neutral_action = torch.cat([neutral_action, torch.tensor([1.0])])

        transition = step_env_and_process_transition(
            env=env,
            transition=transition,
            action=neutral_action,
            env_processor=env_processor,
            action_processor=action_processor,
        )

        reward = transition.get(TransitionKey.REWARD, None)
        done = transition.get(TransitionKey.DONE, False)
        truncated = transition.get(TransitionKey.TRUNCATED, False)
        info = transition.get(TransitionKey.INFO, {})

        print(i, "reward=", reward, "done=", done, "truncated=", truncated, "info=", info)

        if done or truncated:
            print("episode ended")
            break

        busy_wait(1.0 / fps - (time.perf_counter() - start))

finally:
    env.close()
    if teleop_device is not None and getattr(teleop_device, "is_connected", False):
        teleop_device.disconnect()
    print("closed")