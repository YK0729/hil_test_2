##portの検索
lerobot-find-port

##カメラを探す
lerobot-find-cameras

##モーターの登録
#follower
lerobot-setup-motors \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1

#leader
lerobot-setup-motors \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1


##calibrationの実行
#follower
lerobot-calibrate \
--robot.type=so101_follower \
--robot.port=/dev/ttyACM0 \
--robot.id=my_follower

#leader
lerobot-calibrate \
--teleop.type=so101_leader \
--teleop.port=/dev/ttyACM1 \
--teleop.id=my_leader


##テレオペ（録画なし）
lerobot-teleoperate \
--robot.type=so101_follower \
--robot.port=/dev/ttyACM0 \
--robot.id=my_follower \
--robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}, gripper: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
--teleop.type=so101_leader \
--teleop.port=/dev/ttyACM1 \
--teleop.id=my_leader \
--display_data=true


##テレオペ（録画あり）
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=so101_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30},  side: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30},}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=so101_leader \
  --dataset.repo_id=local/pick_bin_and_place_01 \
  --dataset.single_task="pick a bin and place it in circle" \
  --dataset.num_episodes=20 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=15 \
  --dataset.push_to_hub=False

#結果の再生
lerobot-replay \
--robot.type=so101_follower \
--robot.port=/dev/ttyACM1 \
--robot.id=so101_follower \
--dataset.repo_id=local/pick_bin_and_place_01 \
--dataset.episode=0

#学習
lerobot-train \
  --dataset.repo_id=local/pick_bin_and_place_01 \
  --policy.type=act \
  --policy.repo_id=local/act_pick_bin_and_place_01 \
  --output_dir=outputs/train/act_pick_bin_and_place_01 \
  --job_name=act_pick_bin_and_place_01 \
  --policy.device=cuda \
  --wandb.enable=false \
  --training.steps=10000 \
  --dataset.push_to_hub=false


#推論
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=so101_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30} }" \
  --policy.path=outputs/train/act_pick_bin_and_place_01/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --dataset.repo_id=local/eval_act_pick_bin_and_place_01 \
  --dataset.single_task="eval pick bin and place" \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=15 \
  --dataset.push_to_hub=false


##human-in-the-loop型の推論
python serl_test/hil_record_policy_pause_0.4.3.py \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30} }" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_leader \
  --policy.path=outputs/train/act_pick_bin_and_place_01/checkpoints/last/pretrained_model \
  --dataset.repo_id=local/eval_hil_043_test_01 \
  --dataset.single_task="pick a bin and place it in circle" \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --display_data=false \
  --dataset.vcodec=h264
