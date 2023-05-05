import argparse
import shutil
import os
import datetime
import net
from constants import dirsdict
from constants import valuesdict

parser = argparse.ArgumentParser(description='RealSmileNet training')

parser.add_argument('--batch_size', default = valuesdict["batch_size"], type=int,
                    help='the mini batch size used for training')
parser.add_argument('--label_path', default = dirsdict["labels_dir"], type=str,
                    help='the path contains training labels')
parser.add_argument('--frame_path', default = dirsdict["frames_zip_dir"], type=str,
                    help='the path contains processed video frames data')
parser.add_argument('--features_path', default = dirsdict["features_zip_dir"], type=str,
                    help='the path contains processed features data')
parser.add_argument('--vgg_path', default = dirsdict["vgg_zip_dir"], type=str,
                    help='the path contains processed vgg features')
parser.add_argument('--frequency', default = valuesdict["frequency"], type=int,
                    help='the frequency used to sample the data')
parser.add_argument('--epochs', default=valuesdict["epochs"], type=int,
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=valuesdict["learning_rate"], type=float,
                    help='learning rate')
parser.add_argument('--models_dir', default=dirsdict["trained_dir"], type=str,
                    help='directory for all trained models')
parser.add_argument('--out_model', default=dirsdict["models_name"], type=str,
                    help='subdirectory name for trained models')
parser.add_argument('--out_verbose', default=dirsdict["verbose_name"], type=str,
                    help='verbose file name for models training')
parser.add_argument('--out_plot', default=dirsdict["evaluation_name"], type=str,
                    help='learning plot image name for trained model')
parser.add_argument('--calcminmax_features', default=False, type=str,
                    help='calculate min, max and avg for extracted features')
parser.add_argument('-f', nargs='+', required=True, type=list, default=["videos"],
                    help='list of features for model creation.')

def main():
    args = parser.parse_args()

    date = datetime.datetime.now()
    date = date.strftime("%Y-%m-%d_%H-%M")
    try :
        shutil.rmtree(f"{args.models_dir}/{date}")
        os.makedirs(f"{args.models_dir}/{date}")
        os.makedirs(f"{args.models_dir}/{date}/{args.out_model}")
    except FileNotFoundError:
        os.makedirs(f"{args.models_dir}/{date}")
        os.makedirs(f"{args.models_dir}/{date}/{args.out_model}")

    args.out_plot = f"{args.models_dir}/{date}/{args.out_plot}"
    args.out_verbose = f"{args.models_dir}/{date}/{args.out_verbose}"
    args.out_model = f"{args.models_dir}/{date}/{args.out_model}"
    f = ["".join(feature) for feature in args.f]

    uvanemo = net.UVANEMO(
        args.epochs,
        args.lr,
        args.label_path,
        args.frame_path,
        args.frequency,
        args.features_path,
        args.vgg_path,
        args.batch_size,
        args.out_plot,
        args.out_verbose,
        args.out_model,
        args.calcminmax_features,
        f
    )

    uvanemo.prepareData(str(1))
    uvanemo.train(str(1))

main()