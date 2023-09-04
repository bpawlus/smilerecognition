import argparse
import os
import pandas as pd
from constants import dirsdict
import shutil
import numpy as np

parser = argparse.ArgumentParser(description='Get best pretrained models')

parser.add_argument('--d', default = "", type=str,
                    help='subdir')
parser.add_argument('--save', default = "", type=str,
                    help='results directory')

def main():
    """Summarizes trained single models and outputs saved parameters for best single models"""
    args = parser.parse_args()

    val_accs = {}
    test_accs = {}

    train_accs_spo = {}
    train_accs_del = {}

    val_accs_spo = {}
    val_accs_del = {}

    test_accs_spo = {}
    test_accs_del = {}

    dir = dirsdict["single_dir"]
    if args.d != "":
        dir = os.path.join(dir, args.d)

    try:
        shutil.rmtree(args.save)
        os.makedirs(args.save)
    except FileNotFoundError:
        os.makedirs(args.save)

    for features in os.listdir(dir):
        subdir = os.path.join(dir, features)

        #Update
        val_accs[features] = dict()
        test_accs[features] = dict()

        train_accs_spo[features] = dict()
        train_accs_del[features] = dict()

        val_accs_spo[features] = dict()
        val_accs_del[features] = dict()

        test_accs_spo[features] = dict()
        test_accs_del[features] = dict()

        for fold in os.listdir(subdir):
            best_fold_dir = os.path.join(args.save, str(fold))
            if not os.path.exists(best_fold_dir):
                os.makedirs(best_fold_dir)

            subdirfold = os.path.join(subdir, fold)

            model_acc_dir = os.path.join(subdirfold, f"model_acc.csv")
            model_loss_dir = os.path.join(subdirfold, f"model_loss.csv")
            if not os.path.exists(model_acc_dir) or not os.path.exists(model_loss_dir):
                continue

            with open(model_acc_dir, "r") as readfile, open(model_loss_dir, "r") as readfile2:
                df = pd.read_csv(readfile)
                df2 = pd.read_csv(readfile2)

                df_accval = df['Accuracy Validate']
                df_acctest = df['Accuracy Test']

                df_lossval = df2.filter(regex='Loss Validate')
                loss_validate_name = df_lossval.columns[0]
                df_lossval = df_lossval[df_lossval.columns[0]]
                df_losstest = df2.filter(regex='Loss Test')

                df_losstest = df_losstest[df_losstest.columns[0]]

                df_val = pd.concat([df_accval,df_lossval],axis=1)
                df_test = pd.concat([df_acctest, df_losstest], axis=1)
                df_val = df_val.sort_values(by="Accuracy Validate")
                df_test = df_test.sort_values(by="Accuracy Test")

                max_value_acc = df_val['Accuracy Validate'][df_val['Accuracy Validate'].idxmax()]

                df_accval_highest = df_val.where(df_val['Accuracy Validate'] == max_value_acc).dropna()
                min_value_loss = df_accval_highest[loss_validate_name][df_accval_highest[loss_validate_name].idxmin()]

                df_accval_highest = df_accval_highest.where(df_accval_highest[loss_validate_name] == min_value_loss).dropna()

                max_idx = df_accval_highest.index[0]
                test_acc_for_max_value_acc = df_test["Accuracy Test"][max_idx]

                val_accs[features][f"F{fold}"] = max_value_acc
                test_accs[features][f"F{fold}"] = test_acc_for_max_value_acc

                for accs_spo_del_file in [("train", train_accs_spo, train_accs_del), ("val", val_accs_spo, val_accs_del), ("test", test_accs_spo, test_accs_del)]:
                    accs_spo_del_file_dir = os.path.join(subdirfold, f"model_{accs_spo_del_file[0]}_labels.csv")
                    if not os.path.exists(accs_spo_del_file_dir):
                         continue

                    with open(accs_spo_del_file_dir, "r") as readfile_labels:
                        df_labels = pd.read_csv(readfile_labels)
                        np_labels_best = df_labels[f"E{max_idx}"].to_numpy().astype(bool)
                        np_labels_true = df_labels["true"].to_numpy().astype(bool)

                        np_labels_true_spo = np.asarray(np_labels_true == True).nonzero()
                        np_labels_true_del = np.asarray(np_labels_true == False).nonzero()

                        np_labels_best_spo = np_labels_best[np_labels_true_spo]
                        np_labels_best_del = np_labels_best[np_labels_true_del]

                        accs_spo_del_file[1][features][f"F{fold}"] = np.mean(np_labels_best_spo)
                        accs_spo_del_file[2][features][f"F{fold}"] = 1 - np.mean(np_labels_best_del)



                #Copy best model
                models_dir = os.path.join(subdirfold, "models")
                models_dir_list = os.listdir(models_dir)
                best_model = [model_name for model_name in models_dir_list if f"E-{max_idx}" in model_name]
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


    for accs_spo_del_file in [("train", train_accs_spo, train_accs_del), ("val", val_accs_spo, val_accs_del), ("test", test_accs_spo, test_accs_del)]:
        df = pd.DataFrame.from_dict(accs_spo_del_file[1]).transpose()
        df['average'] = df.mean(numeric_only=True, axis=1)
        df['std +-'] = df.std(numeric_only=True, axis=1)
        df.to_csv(os.path.join(args.save, f"{accs_spo_del_file[0]}_accs_spo.csv"), sep=",")
        df.to_excel(os.path.join(args.save, f"{accs_spo_del_file[0]}_accs_spo.xlsx"))

        df = pd.DataFrame.from_dict(accs_spo_del_file[2]).transpose()
        df['average'] = df.mean(numeric_only=True, axis=1)
        df['std +-'] = df.std(numeric_only=True, axis=1)
        df.to_csv(os.path.join(args.save, f"{accs_spo_del_file[0]}_accs_del.csv"), sep=",")
        df.to_excel(os.path.join(args.save, f"{accs_spo_del_file[0]}_accs_del.xlsx"))

if __name__ == "__main__":
    main()