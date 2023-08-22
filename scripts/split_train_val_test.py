import argparse
import os.path

import loguru
import numpy as np

from src.utils import common


def _configure_argparser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data",
        nargs="+",
        type=str,
        default=["data/winequality-red.csv", "data/winequality-white.csv"],
        help="the csv files to split into train val test.",
    )
    argparser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="The proportion of data that will be used for validation.",
    )
    argparser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="The proportion of data that will be used for test.",
    )
    argparser.add_argument(
        "--seed",
        type=int,
        default=21,
        help="Random seed for deterministic partition.",
    )
    return argparser


def main(data: list[str], val_ratio: float, test_ratio: float, seed: int):
    common.set_deterministic_mode(seed)

    _x, _y, header = [], [], []

    for ind, filename in enumerate(data):
        loguru.logger.info("Reading file {} | wine type: {}", filename, ind)

        x, y, header = common.load_data_from_csv(filename, sep=common.CSV_SEPARATOR)
        type_column = np.repeat(np.float32(ind), len(y))
        _x.append(np.column_stack((type_column, x)))
        _y.append(y)

    header = ["wine type"] + header
    x, y = np.concatenate(_x), np.concatenate(_y)

    partition = common.split_into_train_val_test(x, y, val_ratio, test_ratio)

    dirname = os.path.dirname(data[-1])
    for group_name, (x, y) in partition.items():
        filename = os.path.join(dirname, f"{group_name}.csv")
        loguru.logger.info(
            "Saving {} into {} | dataset size: {}",
            group_name,
            filename,
            len(y),
        )
        common.save_csv(header, x, y, filename, common.CSV_SEPARATOR)


if __name__ == "__main__":
    args = _configure_argparser().parse_args()
    main(**vars(args))
