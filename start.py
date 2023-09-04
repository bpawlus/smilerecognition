import argparse
import shutil
import os
import datetime
import net
from constants import dirsdict
from constants import valuesdict

parser = argparse.ArgumentParser(description='RealSmileNet and AUDA training')

parser.add_argument('--usage', default='Single', choices=('Split','Single','Multi'),
                    help='Single - train single models, Multi - train concatenation models, Split - split folds for k-cross validation')

parser.add_argument('--batch_size_train', default = valuesdict["batch_size_train"], type=int,
                    help='the mini batch size used for training')
parser.add_argument('--batch_size_valtest', default = valuesdict["batch_size_valtest"], type=int,
                    help='the mini batch size used for validation and testing')

parser.add_argument('--folds_path', default = dirsdict["folds_dir"], type=str,
                    help='the path contains k-fold cross validation groups')
parser.add_argument('--folds_orig_path', default = dirsdict["folds_orig_dir"], type=str,
                    help='the path contains original realsmilenet folds')

parser.add_argument('--label_path', default = dirsdict["labels_dir"], type=str,
                    help='the path that contains training labels')
parser.add_argument('--frame_path', default = dirsdict["frames_zip_dir"], type=str,
                    help='the path that contains processed video frames')
parser.add_argument('--features_path', default = dirsdict["features_zip_dir"], type=str,
                    help='the path that contains processed AUDA features')

parser.add_argument('-f', nargs='+', type=list, default=["videos"],
                    help='list of features for model creation.')
parser.add_argument('--frequency', default = valuesdict["frequency"], type=int,
                    help='the frequency used to sample the data')
parser.add_argument('--epochs', default=valuesdict["epochs"], type=int,
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=valuesdict["learning_rate"], type=float,
                    help='learning rate')

parser.add_argument('--single_dir', default=dirsdict["single_dir"], type=str,
                    help='directory for single models that will be trained')
parser.add_argument('--multi_dir', default=dirsdict["multi_dir"], type=str,
                    help='directory with model parameters that will be concatenated')

parser.add_argument('-ignore', nargs='+', type=list, default=["videos"],
                    help='ignore list of submodel features for concatenation model creation.')
parser.add_argument('--v', '--variants', default=valuesdict["learning_rate"], type=int,
                    help='concatenation variant')
parser.add_argument('--calcminmax_features', default=False, type=str,
                    help='calculate min, max and avg for extracted features')



def main():
    """Starts process of single and concatenation models training and validation"""
    k = 10
    for fold in range(1):
            args = parser.parse_args()
            date = datetime.datetime.now()
            date = date.strftime("%Y-%m-%d_%H-%M")

            if args.usage == "Split":
                uvanemo = net.UVANEMO(
                    args.epochs,
                    args.lr,
                    args.folds_path,
                    args.frame_path,
                    args.frequency,
                    args.features_path,
                    args.batch_size_train,
                    args.batch_size_valtest,
                    args.calcminmax_features
                )

                uvanemo.split_orig(args.folds_orig_path, args.folds_path)

            args.single_dir = os.path.join(args.single_dir, date)

            try:
                shutil.rmtree(args.single_dir)
                os.makedirs(args.single_dir)
            except FileNotFoundError:
                os.makedirs(args.single_dir)

            if args.usage == "Single":
                #args.epochs = epochs[i]
                #args.f = fs[i]
                #args.lr = lrs[i]

                f = ["".join(feature) for feature in args.f]

                uvanemo = net.UVANEMO(
                    args.epochs,
                    args.lr,
                    args.folds_path,
                    args.frame_path,
                    args.frequency,
                    args.features_path,
                    args.batch_size_train,
                    args.batch_size_valtest,
                    args.calcminmax_features
                )

                k = 10
                for fold in range(k):
                    uvanemo.prepare_data_single(str(fold + 1), args.single_dir, f)
                    uvanemo.single_train(str(fold + 1))
            elif args.usage == "Multi":
                #args.lr = lrs[i]
                #args.epochs = epochs[i]
                #args.v = variants[i]

                ignore = ["".join(i) for i in args.ignore]
                uvanemo = net.UVANEMO(
                    args.epochs,
                    args.lr,
                    args.folds_path,
                    args.frame_path,
                    args.frequency,
                    args.features_path,
                    args.batch_size_train,
                    args.batch_size_valtest,
                    args.calcminmax_features
                )

                k = 10
                for fold in range(k):
                    uvanemo.prepare_data_concat(str(fold + 1), args.single_dir, args.multi_dir, args.v, ignore)
                    uvanemo.multi_train(str(fold + 1))
if __name__ == "__main__":
    main()