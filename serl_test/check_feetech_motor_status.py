#!/usr/bin/env python

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

from lerobot.configs import parser
from lerobot.robots import RobotConfig, make_robot_from_config, so_follower  # noqa: F401
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


READ_NAMES = [
    "Present_Position",
    "Present_Velocity",
    "Present_Load",
    "Present_Current",
]


@dataclass
class MotorStatusConfig:
    robot: RobotConfig

    # 読み取り周期
    hz: float = 10.0

    # ログ時間
    duration_s: float = 30.0

    # CSV保存先
    csv_path: str = "logs/motor_status_log.csv"

    # 何stepごとに標準出力するか
    print_every: int = 10


def try_sync_read(bus: Any, data_name: str) -> dict[str, float]:
    """Read one motor register safely."""
    last_error = None

    call_patterns = [
        lambda: bus.sync_read(data_name),
        lambda: bus.sync_read(data_name=data_name),
    ]

    for call in call_patterns:
        try:
            values = call()

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

        except Exception as e:
            last_error = e

    raise RuntimeError(f"Failed to sync_read({data_name}): {last_error}")


def read_all_motor_status(bus: Any) -> dict[str, dict[str, float]]:
    readings = {}

    for data_name in READ_NAMES:
        try:
            readings[data_name] = try_sync_read(bus, data_name)
        except Exception as e:
            readings[data_name] = {}
            print(f"[WARN] {data_name} read failed: {e}")

    return readings


def flatten_readings(
    timestamp: float,
    step: int,
    readings: dict[str, dict[str, float]],
) -> dict[str, float | int]:
    row: dict[str, float | int] = {
        "timestamp": float(timestamp),
        "step": int(step),
    }

    for data_name, values in readings.items():
        for motor_name, value in values.items():
            row[f"{data_name}.{motor_name}"] = float(value)

    return row


def format_motor_table(readings: dict[str, dict[str, float]]) -> str:
    motor_names = sorted(
        {
            motor_name
            for values in readings.values()
            for motor_name in values.keys()
        }
    )

    lines = []
    lines.append(
        f"{'motor':18s} "
        f"{'pos':>10s} "
        f"{'vel':>10s} "
        f"{'load':>10s} "
        f"{'current':>10s}"
    )
    lines.append("-" * 64)

    for motor in motor_names:
        pos = readings.get("Present_Position", {}).get(motor)
        vel = readings.get("Present_Velocity", {}).get(motor)
        load = readings.get("Present_Load", {}).get(motor)
        cur = readings.get("Present_Current", {}).get(motor)

        def fmt(x):
            return "NA" if x is None else f"{x:10.2f}"

        lines.append(
            f"{motor:18s} "
            f"{fmt(pos):>10s} "
            f"{fmt(vel):>10s} "
            f"{fmt(load):>10s} "
            f"{fmt(cur):>10s}"
        )

    return "\n".join(lines)


@parser.wrap()
def main(cfg: MotorStatusConfig) -> None:
    init_logging()
    print(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)

    csv_path = Path(cfg.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    csv_file = None
    csv_writer = None
    fieldnames = None

    try:
        robot.connect()
        print("[INFO] Robot connected.")
        print("[INFO] Motor names:", list(robot.bus.motors.keys()))
        print(f"[INFO] Logging to: {csv_path}")

        csv_file = csv_path.open("w", newline="")

        dt_target = 1.0 / cfg.hz
        start_t = time.perf_counter()
        step = 0

        while time.perf_counter() - start_t < cfg.duration_s:
            loop_t = time.perf_counter()
            timestamp = loop_t - start_t

            readings = read_all_motor_status(robot.bus)
            row = flatten_readings(
                timestamp=timestamp,
                step=step,
                readings=readings,
            )

            if csv_writer is None:
                fieldnames = list(row.keys())
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                csv_writer.writeheader()

            csv_writer.writerow(row)
            csv_file.flush()

            if step % cfg.print_every == 0:
                print("")
                print(f"[t={timestamp:8.3f}s step={step}]")
                print(format_motor_table(readings))

            step += 1

            elapsed = time.perf_counter() - loop_t
            time.sleep(max(dt_target - elapsed, 0.0))

    finally:
        if csv_file is not None:
            csv_file.close()
            print(f"[INFO] CSV saved: {csv_path}")

        if robot.is_connected:
            robot.disconnect()
            print("[INFO] Robot disconnected.")


if __name__ == "__main__":
    register_third_party_plugins()
    main()