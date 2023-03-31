import argparse
import shutil
import os

import net
from constants import dirsdict
from constants import valuesdict

parser = argparse.ArgumentParser(description='RealSmileNet training')

parser.add_argument('--batch_size', default = valuesdict["batch_size"], type=int,
                    help='the mini batch size used for training')
parser.add_argument('--label_path', default = dirsdict["labels_dir"], type=str,
                    help='the path contains training labels')
parser.add_argument('--frame_path', default = dirsdict["frames_zip_dir"]+".zip", type=str,
                    help='the path contains processed data')
parser.add_argument('--frequency', default = valuesdict["frequency"], type=int,
                    help='the frequency used to sample the data')
parser.add_argument('--epochs', default=valuesdict["epochs"], type=int,
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=valuesdict["learning_rate"], type=float,
                    help='learning rate')
parser.add_argument('--out', default=dirsdict["trained_models_dir"], type=str,
                    help='output directory for trained models')
parser.add_argument('--verbose', default=dirsdict["verbose_path"], type=str,
                    help='verbose file for models training')
parser.add_argument('--out_plot', default=dirsdict["evaluation_plot_dir"], type=str,
                    help='learning plot for trained model')

def main():
    args = parser.parse_args()

    try :
        shutil.rmtree(f"{args.out}")
        os.makedirs(f"{args.out}")
    except FileNotFoundError:
        os.makedirs(f"{args.out}")

    uvanemo = net.UVANEMO(
        args.epochs,
        args.lr,
        args.label_path,
        args.frame_path,
        args.frequency,
        args.batch_size,
        args.out,
        args.verbose,
        args.out_plot
    )

    uvanemo.prepareData(str(1))
    uvanemo.train(str(1))
    uvanemo.evaluate_last()

main()