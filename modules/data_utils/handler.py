import os
import pandas as pd


def get_path( DIR_PATH: str, FILE: str ):
    return os.path.join(DIR_PATH, FILE)


def load_train_data( PATH: str ):
    data = pd.read_csv(
        PATH,
        sep=",",
        index_col=0,
        header=0,
    )
    return data


def load_test_data( PATH: str ):
    data = pd.read_csv(
        PATH,
        sep=",",
        index_col=0,
        header=0,
        parse_dates=["dates"]
    )
    return data


if __name__ == "__main__":
    DATA_DIR = "./data"

    TRAIN_FILE = "train.csv"
    TEST_FILE = "test.csv"

    TRAIN_PATH = get_path(DATA_DIR, TRAIN_FILE)
    TEST_PATH = get_path(DATA_DIR, TEST_FILE)

    train_df = load_train_data(TRAIN_PATH)
    test_df = load_test_data(TEST_FILE)