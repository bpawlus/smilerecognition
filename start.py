import argparse
import shutil
import os
import datetime
import net
from constants import dirsdict
from constants import valuesdict

parser = argparse.ArgumentParser(description='RealSmileNet training')

parser.add_argument('--batch_size_train', default = valuesdict["batch_size_train"], type=int,
                    help='the mini batch size used for training')
parser.add_argument('--batch_size_valtest', default = valuesdict["batch_size_valtest"], type=int,
                    help='the mini batch size used for validation and testing')

parser.add_argument('--split_first', default='True', choices=('Orig','True','False'),
                    help='should k-fold cross validation be performed first')
parser.add_argument('--folds_path', default = dirsdict["folds_dir"], type=str,
                    help='the path contains k-fold cross validation groups')
parser.add_argument('--folds_orig_path', default = dirsdict["folds_orig_dir"], type=str,
                    help='the path contains original realsmilenet folds')

parser.add_argument('--label_path', default = dirsdict["labels_dir"], type=str,
                    help='the path contains training labels')
parser.add_argument('--frame_path', default = dirsdict["frames_zip_dir"], type=str,
                    help='the path contains processed video frames data')
parser.add_argument('--features_path', default = dirsdict["features_zip_dir"], type=str,
                    help='the path contains processed features data')
parser.add_argument('--vgg_path', default = dirsdict["vgg_zip_dir"], type=str,
                    help='the path contains processed vgg features')

parser.add_argument('-f', nargs='+', required=True, type=list, default=["videos"],
                    help='list of features for model creation.')
parser.add_argument('--frequency', default = valuesdict["frequency"], type=int,
                    help='the frequency used to sample the data')
parser.add_argument('--epochs', default=valuesdict["epochs"], type=int,
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=valuesdict["learning_rate"], type=float,
                    help='learning rate')

parser.add_argument('--models_dir', default=dirsdict["trained_dir"], type=str,
                    help='directory for all trained models')

parser.add_argument('--calcminmax_features', default=False, type=str,
                    help='calculate min, max and avg for extracted features')



def main():
    fs = [["aus"], ["videos"], ["si"], ["d1da27"], ["d2da27"], ['videos', 'aus', 'si', 'd1da27']]

    for f in fs:
        args = parser.parse_args()

        date = datetime.datetime.now()
        date = date.strftime("%Y-%m-%d_%H-%M")
        args.models_dir = os.path.join(args.models_dir, date)
        try:
            shutil.rmtree(args.models_dir)
            os.makedirs(args.models_dir)
        except FileNotFoundError:
            os.makedirs(args.models_dir)

        #f = ["".join(feature) for feature in args.f]
        uvanemo = net.UVANEMO(
            args.epochs,
            args.lr,
            args.label_path,
            args.folds_path,
            args.folds_orig_path,
            args.frame_path,
            args.frequency,
            args.features_path,
            args.vgg_path,
            args.batch_size_train,
            args.batch_size_valtest,
            args.models_dir,
            args.calcminmax_features,
            f
        )

        k = 9
        if args.split_first == 'True':
            k = 10
            uvanemo.split(k)
        elif args.split_first == "Orig":
            k = 9
            uvanemo.split_orig(10)

        for i in range(k):
            uvanemo.prepareData(str(i+1))
            uvanemo.train(str(i+1))

main()