#! /usr/bin/env python3

"""
Split images into separate train and test folders
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", "--d", type=Path, default="./datasets")
    args = parser.parse_args()

    dataset_dir = args.dataset_root / "CUB_200_2011"  # type: Path

    im_df = pd.read_csv(dataset_dir / "images.txt", sep=" ",
                        header=None, names=["image_id", "image_path"])
    split_df = pd.read_csv(dataset_dir/ "train_test_split.txt",
                           sep=" ", header=None, names=["image_id", "is_train"])
    merged_df = im_df.merge(split_df, on="image_id")
    merged_df["image_id"] -= 1

    dst_train_dir, dst_test_dir = dataset_dir / "train", dataset_dir / "test"

    for row in tqdm(merged_df.itertuples(index=False), total=len(merged_df)):
        image_path = Path(row.image_path)

        src_path = dataset_dir / "images" / image_path
        dst_path = (dst_train_dir if row.is_train else dst_test_dir) / image_path.parent

        if not dst_path.exists():
            dst_path.mkdir(parents=True, exist_ok=True)

        shutil.move(src_path.as_posix(), dst_path.as_posix())

    shutil.rmtree(dataset_dir / "images")
