import argparse
import os
import pandas as pd
from constants import dirsdict
import shutil

parser = argparse.ArgumentParser(description='Get best pretrained models')

parser.add_argument('--d', default = "", type=str,
                    help='subdir')
parser.add_argument('--save', default = "", type=str,
                    help='results directory')

def main():
    args = parser.parse_args()

    val_accs = {}
    test_accs = {}

    dir = dirsdict["train_dir"]
    if args.d != "":
        dir = os.path.join(dir, args.d)

    try:
        shutil.rmtree(args.save)
        os.makedirs(args.save)
    except FileNotFoundError:
        os.makedirs(args.save)

    for features in os.listdir(dir):
        subdir = os.path.join(dir, features)

        val_accs[features] = dict()
        test_accs[features] = dict()

        for fold in os.listdir(subdir):
            best_fold_dir = os.path.join(args.save, str(fold))
            if not os.path.exists(best_fold_dir):
                os.makedirs(best_fold_dir)

            subdirfold = os.path.join(subdir, fold)

            file_dir = os.path.join(subdirfold, f"model_acc.csv")
            if not os.path.exists(file_dir):
                continue

            with open(file_dir, "r") as readfile:
                df = pd.read_csv(readfile)

                df_accval = df['Accuracy Validate']
                df_acctest = df['Accuracy Test']

                maxidx = df_accval.idxmax()

                max_accval = df_accval[maxidx]
                acctest_accval = df_acctest[maxidx]

                val_accs[features][f"F{fold}"] = max_accval
                test_accs[features][f"F{fold}"] = acctest_accval

                models_dir = os.path.join(subdirfold, "models")
                models_dir_list = os.listdir(models_dir)
                best_model = [model_name for model_name in models_dir_list if f"E-{maxidx}" in model_name]

                shutil.copy2(os.path.join(models_dir, best_model[0]), os.path.join(best_fold_dir, f"{features}.pt"))

    df = pd.DataFrame.from_dict(val_accs).transpose()
    df['average'] = df.mean(numeric_only=True, axis=1)
    df['std +-'] = df.std(numeric_only=True, axis=1)
    df.to_csv(os.path.join(args.save, "val_accs.csv"), sep=",")
    df.to_excel(os.path.join(args.save, "val_accs.xlsx"))

    df = pd.DataFrame.from_dict(test_accs).transpose()
    df['average'] = df.mean(numeric_only=True, axis=1)
    df['std +-'] = df.std(numeric_only=True, axis=1)
    df.to_csv(os.path.join(args.save, "test_accs.csv"), sep=",")
    df.to_excel(os.path.join(args.save, "test_accs.xlsx"))

main()