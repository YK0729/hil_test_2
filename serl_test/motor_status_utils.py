#モーターの各種数値の読み取り保存用関数群
MOTOR_STATUS_NAMES = [
    "Present_Position",
    "Present_Velocity",
    "Present_Load",
    "Present_Current",
]


def try_sync_read(bus, data_name: str) -> dict[str, float]:
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
    out = {}

    for data_name in MOTOR_STATUS_NAMES:
        try:
            out[data_name] = try_sync_read(bus, data_name)
        except Exception:
            out[data_name] = {}

    return out


def flatten_motor_status(
    prefix: str,
    status: dict[str, dict[str, float]],
) -> dict[str, float]:
    row = {}

    for data_name, values in status.items():
        for motor_name, value in values.items():
            row[f"{prefix}.{data_name}.{motor_name}"] = float(value)

    return row