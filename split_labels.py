import argparse
import os
import pandas as pd
from constants import dirsdict
from constants import valuesdict

parser = argparse.ArgumentParser(description='Split labels to spontaneus and deliberate')

parser.add_argument('--d', default = "", type=str,
                    help='subdir')

def main():
    args = parser.parse_args()

    dir = dirsdict["trained_dir"]
    if args.d != "":
        dir = os.path.join(dir, args.d)


    for features in os.listdir(dir):
        subdir = os.path.join(dir, features)
        print()
        for fold in os.listdir(subdir):
            subdirfold = os.path.join(subdir, fold)

            for file in ["train", "val", "test"]:
                file_dir = os.path.join(subdirfold, f"model_{file}_labels.csv")
                if not os.path.exists(file_dir):
                    continue

                with open(file_dir, "r") as readfile:
                    df = pd.read_csv(readfile)
                    df_deliberate = df[df['true'] == 0]
                    df_spontaneus = df[df['true'] == 1]

                    df_deliberate = df_deliberate.rename(columns={df_deliberate.columns[0] : ""})
                    df_spontaneus = df_spontaneus.rename(columns={df_spontaneus.columns[0] : ""})

                    new_file_dir_deliberate = os.path.join(subdirfold, f"model_{file}_labels_del.csv")
                    new_file_dir_spontaneus = os.path.join(subdirfold, f"model_{file}_labels_spo.csv")
                    df_deliberate.to_csv(new_file_dir_deliberate, sep=",", index=False)
                    df_spontaneus.to_csv(new_file_dir_spontaneus, sep=",", index=False)

main()