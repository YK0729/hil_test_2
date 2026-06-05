# test_hilserl_reset.py

from lerobot.envs.configs import HILSerlProcessorConfig, HILSerlRobotEnvConfig
from lerobot.rl.gym_manipulator import make_robot_env
from lerobot.robots.so101_follower import SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101LeaderConfig

follower_port = "/dev/ttyACM0"  # 実際の値
leader_port = "/dev/ttyACM1"    # 実際の値

follower_id = "so101_follower"
leader_id = "so101_leader"

robot_cfg = SO101FollowerConfig(port=follower_port, id=follower_id)
teleop_cfg = SO101LeaderConfig(port=leader_port, id=leader_id)
processor_cfg = HILSerlProcessorConfig(control_mode="leader")

env_cfg = HILSerlRobotEnvConfig(
    robot=robot_cfg,
    teleop=teleop_cfg,
    processor=processor_cfg,
)

env, teleop_device = make_robot_env(env_cfg)

obs, info = env.reset()

print("reset OK")
print(type(obs))
print(obs.keys() if hasattr(obs, "keys") else obs)

env.close()