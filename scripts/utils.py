import os
import subprocess
from dataclasses import dataclass

from loguru import logger


@dataclass
class Model:
    name: str
    temp: float = 0.0
    samples_per_task: int = 1


def cmd(cmd: str) -> bool:
    process = subprocess.run(
        cmd,
        shell=True,
        text=True,
        capture_output=True,
        timeout=5,
    )

    if process.stderr:
        logger.warning(f"Err: {process.stderr}")
        logger.warning(f"Return code: {process.returncode}")
    return process.returncode == 0


def cleanup_dylib(name: str):
    try:
        os.remove(f"{name}.dylib")
    except FileNotFoundError:
        logger.warning(f"File {name}.dylib not found")


def cleanup_file(name: str):
    try:
        os.remove(f"{name}")
    except FileNotFoundError:
        logger.warning(f"Exe {name} not found")
