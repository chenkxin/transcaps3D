import argparse
import json
import os

import pandas as pd


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def read_param(expr_dir):
    """read a experiment's param to json"""
    path = f"{expr_dir}/params.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    data = dotdict(data)
    return data


def save_to_excel(param, filename="logs/exprs.csv"):
    """save the param json to excel"""
    df = pd.DataFrame(param, columns=param.keys())
    # df = df.reindex(sorted(df.columns), axis=1)

    cols = df.columns.tolist()
    front_cols = [
        "expr_name",
        "best_acc",
        "eval_acc",
        "routing",
        "use_residual_block",
        "loss",
        "type",
    ]
    new_cols = front_cols + [i for i in cols if i not in front_cols]
    df = df.reindex(columns=new_cols)
    df = df.to_csv(filename)


def save(files, filename="logs/exprs.csv"):
    if not files:
        return
    try:
        keys = read_param(files[0]).keys()
    except:
        print(files[0])
        return
    json_list = dotdict({i: [] for i in keys})
    for expr_dir in files:
        param = read_param(expr_dir)
        if param:
            for i in keys:
                elem = param[i] if i in param.keys() else None
                json_list[i].append(elem)

        save_to_excel(json_list, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A experiments collector",
    )

    parser.add_argument(
        "files",
        metavar="FILE",
        nargs="*",
        help="Directories of experiments",
    )
    args = parser.parse_args()
    save(args.files)
