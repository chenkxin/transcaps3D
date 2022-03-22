import pickle as pkl
import os
import glob
import numpy as np
from tqdm import tqdm


def minus_mean_std(name, train, type, bandwidths):
    for b in bandwidths:
        minus(name, train, type, b)


def _load_info(file):
    try:
        info = pkl.load(open(file, "rb"))
    except:
        print("file non-existent")
        info = {}
    return info


def minus(name, train, type, b=32):
    """minus mean and std from npy files, and re-save them as npy files"""
    partition = "train" if train else "test"
    files = glob.glob(f"data/{name}/{name}_{partition}/{type}/b{b}*.npy")
    info = _load_info("data/info.pkl")
    for f in files:
        head, tail = os.path.split(f)
        bandwidth = int(tail.split("_")[0][1:])
        _specific_info = info[name][partition][type][bandwidth]
        img = np.load(f)
        img = img - _specific_info["mean"]
        img = img / _specific_info["std"]
        np.save(f, img)


def add(name, train, type):
    """minus mean and std from npy files, and re-save them as npy files"""
    files = glob.glob(f"data/{name}/*/{type}/*.npy")
    info = _load_info("data/info.pkl")
    for f in tqdm(files):
        head, tail = os.path.split(f)
        partition = "train" if train else "test"
        bandwidth = int(tail.split("_")[0][1:])
        _specific_info = info[name][partition][type][bandwidth]
        img = np.load(f)
        img = img * _specific_info["std"]
        img = img + _specific_info["mean"]

        np.save(f, img)


if __name__ == "__main__":
    for name in ["modelnet10", "modelnet40", "shrec15", "shrec17"]:
        for train in [True, False]:
            for type in ["no_rotate", "rotate"]:
                minus(name, train, type)
