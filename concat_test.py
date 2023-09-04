import argparse
import os

import numpy
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='Get best pretrained models')

parser.add_argument('--save', default = "", type=str,
                    help='results directory')
parser.add_argument('-f', nargs='+', type=list, default=["videos"],
                    help='starting features')

def main():
    """Performs calculation of best models to concatenate with starting features"""
    args = parser.parse_args()
    args.f = ["".join(feature) for feature in args.f]
    file_dir = os.path.join(args.save, f"model_val_labels.csv")
    if not os.path.exists(file_dir):
        return

    data = {}
    with open(file_dir, "r") as readfile:
        df = pd.read_csv(readfile, index_col=0)
        for idx, row in df.iterrows():
            cat = None
            for col in row:
                col = col[1:-1]
                arr = np.fromstring(col, dtype=int, sep=',')
                if cat is not None:
                    cat = np.concatenate((cat, arr))
                else:
                    cat = arr
            data[idx] = cat.astype(bool)

    base_models = data[args.f[0]]
    for f in args.f:
        prev_data = data[f]
        data.pop(f)

        base_models = np.logical_or(prev_data, base_models)

    print(f"From {args.f} features, taking single answer as correct, they cover {(np.count_nonzero(base_models) / len(base_models))*100}% answers")

    data2 = {}
    for key, value in data.items():
        extra_acc = np.logical_or(value, base_models)
        data2[key] = (np.count_nonzero(extra_acc) / len(base_models))*100

    data2 = sorted(data2.items(), key=lambda x:x[1])
    for tup in data2:
        print(f"  Adding {tup[0]} feature, will cover {tup[1]}% answers")

if __name__ == "__main__":
    main()