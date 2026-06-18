#!/usr/bin/env python

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rerun as rr
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_MOTOR_STATUS = [
    "Present_Load",
    "Present_Current",
    "Present_Velocity",
]

DEFAULT_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def image_to_numpy(x: Any) -> np.ndarray:
    img = to_numpy(x)

    # torch image is often CHW
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))

    # float image [0, 1] -> uint8
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)

    return img


def scalar_value(x: Any) -> float:
    arr = to_numpy(x)
    if arr.shape == ():
        return float(arr)
    return float(arr.reshape(-1)[0])


def vector_values(x: Any) -> np.ndarray:
    arr = to_numpy(x)
    return arr.reshape(-1).astype(float)


def find_image_keys(item: dict[str, Any]) -> list[str]:
    image_keys = []

    for key, value in item.items():
        if not key.startswith("observation.images."):
            continue

        try:
            arr = to_numpy(value)
        except Exception:
            continue

        if arr.ndim == 3:
            image_keys.append(key)

    return image_keys


def get_feature_names(ds: LeRobotDataset, key: str, n: int) -> list[str]:
    try:
        feature = ds.features[key]
        names = feature.get("names", None)
        if names is not None and len(names) == n:
            return [str(x) for x in names]
    except Exception:
        pass

    return [f"{key.replace('.', '_')}_{i:02d}" for i in range(n)]


def get_episode_indices(ds: LeRobotDataset, episode_index: int) -> list[int]:
    indices = []

    for i in range(len(ds)):
        item = ds[i]

        if "episode_index" not in item:
            raise KeyError("Dataset item has no 'episode_index'. Cannot select episode.")

        ep = int(scalar_value(item["episode_index"]))
        if ep == episode_index:
            indices.append(i)

    if not indices:
        raise ValueError(f"No frames found for episode_index={episode_index}")

    return indices


def load_motor_csv(
    dataset_root: Path,
    episode_index: int,
    motor_csv: str | None,
) -> pd.DataFrame | None:
    if motor_csv is not None:
        path = Path(motor_csv).expanduser()
    else:
        path = dataset_root / "motor_logs" / f"episode_{episode_index:06d}_motor.csv"

    if not path.exists():
        print(f"[WARN] Motor CSV not found: {path}")
        return None

    df = pd.read_csv(path)
    print(f"[INFO] Loaded motor CSV: {path}")
    print(f"[INFO] Motor CSV rows: {len(df)}")
    return df


def get_motor_row(
    motor_df: pd.DataFrame | None,
    episode_frame_index: int,
    timestamp: float,
) -> pd.Series | None:
    if motor_df is None or len(motor_df) == 0:
        return None

    if "episode_frame_index" in motor_df.columns:
        hit = motor_df[motor_df["episode_frame_index"] == episode_frame_index]
        if len(hit) > 0:
            return hit.iloc[0]

    if "timestamp" in motor_df.columns:
        idx = (motor_df["timestamp"] - timestamp).abs().idxmin()
        return motor_df.loc[idx]

    if episode_frame_index < len(motor_df):
        return motor_df.iloc[episode_frame_index]

    return None


def short_status_name(status_name: str) -> str:
    if status_name == "Present_Load":
        return "load"
    if status_name == "Present_Current":
        return "current"
    if status_name == "Present_Velocity":
        return "velocity"
    return status_name.replace("Present_", "").lower()


def log_images(item: dict[str, Any], image_keys: list[str]) -> None:
    for key in image_keys:
        img = image_to_numpy(item[key])
        short_key = key.replace("observation.images.", "")
        rr.log(f"camera/{short_key}", rr.Image(img))


def log_action(ds: LeRobotDataset, item: dict[str, Any]) -> None:
    if "action" not in item:
        return

    action = vector_values(item["action"])
    names = get_feature_names(ds, "action", len(action))

    for name, value in zip(names, action):
        rr.log(f"action/{name}", rr.Scalars(float(value)))


def log_observation_state(ds: LeRobotDataset, item: dict[str, Any]) -> None:
    if "observation.state" not in item:
        return

    state = vector_values(item["observation.state"])
    names = get_feature_names(ds, "observation.state", len(state))

    for name, value in zip(names, state):
        rr.log(f"observation/state/{name}", rr.Scalars(float(value)))


def log_motor_row(
    row: pd.Series | None,
    enabled_status: set[str],
    enabled_motors: set[str],
) -> None:
    if row is None:
        return

    skip_cols = {
        "timestamp",
        "episode_index",
        "episode_frame_index",
        "frame_index",
        "loop_step",
        "loop_time_s",
    }

    for col, value in row.items():
        if col in skip_cols:
            continue

        # Expected:
        # follower.Present_Load.gripper
        # follower.Present_Current.gripper
        # follower.Present_Velocity.gripper
        parts = col.split(".")
        if len(parts) != 3:
            continue

        prefix, status_name, motor_name = parts

        if prefix != "follower":
            continue

        # Positionはここで完全に除外される
        if status_name not in enabled_status:
            continue

        if motor_name not in enabled_motors:
            continue

        try:
            v = float(value)
        except Exception:
            continue

        status_short = short_status_name(status_name)

        # Rerun上で load/current/velocity が別々の階層になる
        rr.log(f"motor/{status_short}/{motor_name}", rr.Scalars(v))


def log_motor_derived(
    row: pd.Series | None,
    prev_row: pd.Series | None,
    enabled_motors: set[str],
) -> None:
    if row is None or prev_row is None:
        return

    for motor_name in enabled_motors:
        col = f"follower.Present_Load.{motor_name}"

        if col not in row.index or col not in prev_row.index:
            continue

        try:
            dload = float(row[col]) - float(prev_row[col])
        except Exception:
            continue

        rr.log(f"motor/dload/{motor_name}", rr.Scalars(dload))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo_id",
        required=True,
        help="LeRobot dataset repo id, e.g. local/motor_log_save_test",
    )
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--motor_csv", default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--real_time", action="store_true")
    parser.add_argument("--session_name", default="motor_log_viewer")

    parser.add_argument(
        "--motor_status",
        nargs="+",
        default=DEFAULT_MOTOR_STATUS,
        choices=["Present_Load", "Present_Current", "Present_Velocity"],
        help="Motor status to visualize. Position is intentionally excluded.",
    )

    parser.add_argument(
        "--motor_names",
        nargs="+",
        default=DEFAULT_MOTOR_NAMES,
        help="Motor names to visualize.",
    )

    parser.add_argument(
        "--no_state",
        action="store_true",
        help="Do not show observation.state.",
    )

    parser.add_argument(
        "--no_action",
        action="store_true",
        help="Do not show action.",
    )

    parser.add_argument(
        "--show_dload",
        action="store_true",
        help="Show frame-to-frame difference of load.",
    )

    args = parser.parse_args()

    enabled_status = set(args.motor_status)
    enabled_motors = set(args.motor_names)

    ds = LeRobotDataset(args.repo_id)
    dataset_root = Path(ds.root)

    fps = args.fps if args.fps is not None else float(ds.fps)
    dt = 1.0 / fps

    episode_indices = get_episode_indices(ds, args.episode_index)
    motor_df = load_motor_csv(dataset_root, args.episode_index, args.motor_csv)

    first_item = ds[episode_indices[0]]
    image_keys = find_image_keys(first_item)

    print(f"[INFO] Dataset root: {dataset_root}")
    print(f"[INFO] Episode: {args.episode_index}")
    print(f"[INFO] Frames: {len(episode_indices)}")
    print(f"[INFO] Playback fps: {fps}")
    print(f"[INFO] Image keys: {image_keys}")
    print(f"[INFO] Motor status: {sorted(enabled_status)}")
    print(f"[INFO] Motor names: {sorted(enabled_motors)}")

    rr.init(args.session_name, spawn=True)

    prev_motor_row = None

    for local_frame_idx, dataset_idx in enumerate(episode_indices):
        loop_start = time.perf_counter()
        item = ds[dataset_idx]

        if "timestamp" in item:
            timestamp = scalar_value(item["timestamp"])
        else:
            timestamp = local_frame_idx * dt

        rr.set_time("time", timestamp=float(timestamp))
        rr.set_time("frame", sequence=int(local_frame_idx))

        log_images(item, image_keys)

        if not args.no_action:
            log_action(ds, item)

        if not args.no_state:
            log_observation_state(ds, item)

        motor_row = get_motor_row(
            motor_df=motor_df,
            episode_frame_index=local_frame_idx,
            timestamp=timestamp,
        )

        log_motor_row(
            motor_row,
            enabled_status=enabled_status,
            enabled_motors=enabled_motors,
        )

        if args.show_dload:
            log_motor_derived(
                motor_row,
                prev_motor_row,
                enabled_motors=enabled_motors,
            )

        prev_motor_row = motor_row

        loop_elapsed = time.perf_counter() - loop_start

        if args.real_time:
            time.sleep(max(dt - loop_elapsed, 0.0))


if __name__ == "__main__":
    main()