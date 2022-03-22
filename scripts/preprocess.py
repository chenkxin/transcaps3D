"""
Compute mean and std for 3d datasets, and normalize them.
Finally results will be saved at each directory with filetype "npy"
Usage:
    python scripts/preprocess.py [--check] --bws --types --datasets
"""
import os, sys
import glob
import logging

sys.path.append(os.getcwd())
from lib.utils import dotdict
from lib.dataset.utils import compute_mean_and_std_for
from lib.dataset.transforms import get_transform
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from scripts.minus_mean_std import minus_mean_std


def save_npy(
    dataset_name, train, bws, repeat=1, rot=False, normalize=False, type="rotate"
):
    """Compute the npy file and save them in disk

    Args:
        dataset_name: str
        train: bool
        bws: list
        normalize: bool (useless now)

    Returns:

    """
    partition = "train" if train else "test"
    print(f"Convert off file in <{dataset_name},{partition},{type}>")
    files = glob.glob(f"data/{dataset_name}/{dataset_name}_{partition}/*.off")
    for b in bws:
        transform = get_transform(
            dataset_name, train, b, normalize, repeat=repeat, rot=rot, type=type
        )
        executor = ProcessPoolExecutor(num_workers)
        futures = [executor.submit(transform, f) for f in files]
        wait(futures)


def preprocess_dataset_by_roration(dataset_names, bandwidths, type="rotate"):

    for dataset_name in dataset_names:
        for train in [True, False]:
            if type == "rotate":
                rot = True
                #if train:
                    #repeat = 4  # training data argument
                    #repeat = 1
                #else:
                repeat = 1
            else:
                rot = False
                #repeat = 1
            try:
                save_npy(
                    dataset_name,
                    train,
                    bandwidths,
                    repeat=repeat,
                    normalize=False,
                    rot=rot,
                    type=type,
                )
                compute_mean_and_std_for(
                    dataset_name=dataset_name,
                    train=train,
                    bws=bandwidths,
                    save=True,
                    config=dotdict({"type": type, "pick_randomly": False}),
                )
                minus_mean_std(dataset_name, train, type, bandwidths)
            except Exception as e:
                print(e)


def check(dataset_names, bandwidths, type="rotate"):
    for dataset_name in dataset_names:
        for train in [True, False]:
            compute_mean_and_std_for(
                dataset_name,
                train,
                bandwidths,
                save=False,
                config=dotdict({"type": type, "pick_randomly": False}),
            )


if __name__ == "__main__":
    types = ["rotate", "no_rotate"]

    datasets = ["shrec15_0.{}".format(i) for i in [2, 3, 4, 6, 7]] + [
        "shrec17",
        "modelnet10",
        "modelnet40",
    ]
    bws = [32, 16, 8]
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        default=False,
        action="store_true",
        help="Check whether every dataset is normalized",
    )
    parser.add_argument(
        "--bws",
        nargs="+",
        type=int,
        default=bws,
        help="Bandwidth of the spherical signal, list(int)",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        type=str,
        default=types,
        help="rotate or no_rotate, list(str)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        default=datasets,
        help="the dataset you want to process, list(str)",
    )
    args = parser.parse_args()

    if args.check:
        for type in args.types:
            check(args.datasets, args.bws, type=type)
    else:
        num_workers = 48
        for type in args.types:
            preprocess_dataset_by_roration(args.datasets, args.bws, type)
