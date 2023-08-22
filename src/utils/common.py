import csv
import datetime
import os
import random
from typing import Any

import loguru
import numpy as np
from numpy import typing as npt

WINE_TYPE_COLUMN_NAME = "wine type"
CSV_SEPARATOR = ";"

Features = npt.NDArray[npt.NDArray[np.float32]]
Target = npt.NDArray[np.int8]


def _split_array_into_3_parts(
    arr: npt.NDArray, first_part_size: int, second_part_size: int
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    return (
        arr[:first_part_size],
        arr[first_part_size:second_part_size],
        arr[second_part_size:],
    )


def split_into_train_val_test(
    x: Features,
    y: Target,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, tuple[Features, Target]]:
    full_size = len(y)

    indices = np.random.permutation(full_size)
    x = x[indices]
    y = y[indices]

    val_size = int(full_size * val_ratio)
    test_size = int(full_size * test_ratio)

    val_x, test_x, train_x = _split_array_into_3_parts(
        x, val_size, val_size + test_size
    )
    val_y, test_y, train_y = _split_array_into_3_parts(
        y, val_size, val_size + test_size
    )

    return {
        "train": (train_x, train_y),
        "val": (val_x, val_y),
        "test": (test_x, test_y),
    }


def load_data_from_csv(
    filename: str, sep: str = " "
) -> tuple[Features, Target, list[str]]:
    header = []
    x = []
    y = []
    with open(filename) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=sep)
        header = next(iter(csvreader))

        for row in csvreader:
            x.append(row[:-1])
            y.append(row[-1])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.int8), header


def save_csv(header: list[str], x: Features, y: Target, filename: str, sep: str):
    with open(filename, "w") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=sep, lineterminator="\n")
        csvwriter.writerow(header)
        for i in range(len(y)):
            row = x[i].astype(str).tolist() + [str(y[i])]
            csvwriter.writerow(row)


def set_deterministic_mode(seed: int):
    """Fixes the seed for all random processes
    (for pure python, numpy).
    """
    loguru.logger.info("Set random seed: {}", seed)

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_current_time() -> str:
    return datetime.datetime.now().strftime("%d_%m_%y_%H:%M:%S")


def get_condition_str(condition: dict[str, Any]):
    return " AND ".join([f"[{key}]={value}" for key, value in condition.items()])
