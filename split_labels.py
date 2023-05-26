import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser(description='Split Labels to differen csvs')

parser.add_argument('--d', type=str, help='directory')

def main():
    args = parser.parse_args()

    for file in ["train", "val", "test"]:
        file_dir = os.path.join(args.d, f"model_{file}_labels.csv")
        with open(file_dir, "r") as readfile:
            df = pd.read_csv(readfile)
            df_deliberate = df[df['true'] == 0]
            df_spontaneus = df[df['true'] == 1]

            df_deliberate = df_deliberate.rename(columns={df_deliberate.columns[0] : ""})
            df_spontaneus = df_spontaneus.rename(columns={df_spontaneus.columns[0] : ""})

            new_file_dir_deliberate = os.path.join(args.d, f"model_{file}_labels_del.csv")
            new_file_dir_spontaneus = os.path.join(args.d, f"model_{file}_labels_spo.csv")
            df_deliberate.to_csv(new_file_dir_deliberate, sep=",", index=False)
            df_spontaneus.to_csv(new_file_dir_spontaneus, sep=",", index=False)


main()