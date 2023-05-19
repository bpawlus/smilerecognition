import numpy
import torch
import os
import json
import numpy as np
import zipfile
import pandas as pd
from PIL import Image
import io

def read_image(zip,name):
    return Image.open(io.BytesIO(zip.read(name)))

def read_au_txt(zip,name):
    return pd.read_csv(io.BytesIO(zip.read(name)), sep=",")

def read_si_txt(zip,name):
    return pd.read_csv(io.BytesIO(zip.read(name)), sep="\n", header=None)



def load_frames(videos_data_names, videos_data, name, videos_frequency, video_max_len):
    files = [f for f in videos_data_names[name] if f.endswith(".jpg") and f.replace(".jpg","").isnumeric()]
    files = sorted(files, key = lambda x : int(x.replace(".jpg","")))

    last = [files[-1]] if len(files) % videos_frequency >= 2/3 * videos_frequency else []
    start = 0

    files = np.concatenate([np.array(files)[start::videos_frequency],last])

    frame_count = 0

    pad_x = np.zeros((video_max_len,3,48,48))
    values = []

    for frame in files:
        try:
            cur = read_image(videos_data, os.path.join(name,frame))
        except:
            cur = read_image(videos_data, name+"/"+frame)
        resized_asarray = np.asarray(cur.resize((48,48)))
        cur = np.swapaxes(resized_asarray, 2, 0)

        values.append(cur)
        frame_count+=1

    x = np.array(values)
    frame_count = video_max_len - frame_count
    pad_x[frame_count:] = x

    return pad_x, frame_count

def load_features_si(features_data_names, features_data, name, videos_frequency, video_max_len, video_len):
    files = features_data_names[name]
    pd = read_si_txt(features_data, files["smile_intensities"])

    pad_x = np.zeros((video_max_len, 1))

    last = [files[-1]] if len(files) % videos_frequency >= 2 / 3 * videos_frequency else []
    start = 0
    values = [np.concatenate([pd[i].values[start::videos_frequency], last]) for i in range(1)]
    frame_count = len(values[0])

    x = np.array(values).transpose()
    x = 1/(1+np.exp(-(x*100))) # Translacja elementów (natężenia uśmiechu) - zawierają niskie wartości ujemne i niskie dodatnie - do tensorów warto zastosować moim zdaniem sigmoid
    #  *20 - min max z smile_intensities
    pad_x[0:frame_count] = x

    return pad_x, frame_count

def load_features_aus(features_data_names, features_data, name, videos_frequency, video_max_len, video_len):
    files = features_data_names[name]
    pd = read_au_txt(features_data, files["action_units"])
    columns = sorted(pd.columns.values.tolist())

    pad_x = np.zeros((video_max_len, len(columns)))

    last = [files[-1]] if len(files) % videos_frequency >= 2 / 3 * videos_frequency else []
    start = 0
    values = [np.concatenate([pd[i].values[start::videos_frequency], last]) for i in columns]
    frame_count = len(values[0])

    x = np.array(values).transpose()
    x = 1/(1+np.exp(-(x-0.75))) # Translacja elementów (natężenia action unitów) - zawierają wysokie wartości ujemne i wysokie dodatnie - do tensorów warto zastosować moim zdaniem sigmoid
    # -0.75 - średnia z action_units
    pad_x[0:frame_count] = x

    return pad_x, frame_count

def load_features_dynamics(features_data_names, features_data, name, dynamics_subfeature_name, dynamics_subfeature_mul, videos_frequency, video_max_len, video_len):
    files = features_data_names[name]
    pd = read_au_txt(features_data, files[dynamics_subfeature_name])
    columns = sorted(pd.columns.values.tolist())

    pad_x = np.zeros((video_max_len, len(columns)))

    last = [files[-1]] if len(files) % videos_frequency >= 2 / 3 * videos_frequency else []
    start = 0
    values = [np.concatenate([pd[i].values[start::videos_frequency], last]) for i in columns]
    frame_count = len(values[0])

    x = np.array(values).transpose()
    x = 1/(1 + np.exp(-(x * dynamics_subfeature_mul))) # Translacja elementów (dynamiki natężenia action unitów) - zawierają niskie wartości ujemne i wysokie dodatnie - do tensorów warto zastosować moim zdaniem sigmoid
    pad_x[0:frame_count] = x

    return pad_x, frame_count

def videos_zip_to_dict(data):
    data_name = dict()

    for file in data.namelist():
        if len(file.split("/")) != 2:
            continue
        name, f = file.split("/")
        if name in data_name:
            data_name[name].append(f)
        else:
            data_name[name] = [f]

    return data_name, max([len(i) for i in data_name.values()])

def features_zip_to_dict(data, calcminmax_features):
    data_name = dict()
    minmax_name = dict()

    for file in data.namelist():
        if not file.startswith("features/"):
            feature_name, name_to_split = file.split("/")

            if not name_to_split:
                continue

            splits = name_to_split.split("_")
            res = ["".join(splits[i]+"_" for i in range(4))]

            name = res[0][:-1] # czysta nazwa

            if feature_name == "dynamics": # lub file.startswith("dynamics/"):
                feature_name += "".join("_"+splits[i] for i in range(4, len(splits)))
                feature_name = feature_name.replace(".txt", "")

            if name not in data_name:
                data_name[name] = dict()

            if(calcminmax_features):
                pddata = pd.read_csv(io.BytesIO(data.read(file)))
                pddata_values = pddata.values
                (minval, meanval, maxval) = (pddata_values.min(), pddata_values.mean(), pddata_values.max())

                if feature_name in minmax_name:
                    (prev_minval, prev_meanval, prev_maxval, meancnt) = minmax_name[feature_name]
                    minmax_name[feature_name] = (min(prev_minval, minval), (prev_meanval*meancnt+meanval)/(meancnt+1), max(prev_maxval, maxval), meancnt+1)
                else:
                    minmax_name[feature_name] = (minval, meanval, maxval, 1)

            data_name[name][feature_name] = file
        else:
            if len(file.split("/")) == 2:
                continue

            _, subfeature, name = file.split("/")

            if not name:
                continue

            subfeature = subfeature.replace(" ","_")
            name = name.replace(".txt", "")

            if name not in data_name:
                data_name[name] = dict()

            if(calcminmax_features):
                pddata = pd.read_csv(io.BytesIO(data.read(file)), sep=",", header=None)
                pddata_values = pddata.values
                (minval, meanval, maxval) = (pddata_values.min(), pddata_values.mean(), pddata_values.max())

                if subfeature in minmax_name:
                    (prev_minval, prev_meanval, prev_maxval, meancnt) = minmax_name[subfeature]
                    minmax_name[subfeature] = (min(prev_minval, minval), (prev_meanval*meancnt+meanval)/(meancnt+1), max(prev_maxval, maxval), meancnt+1)
                else:
                    minmax_name[subfeature] = (minval, meanval, maxval, 1)

            data_name[name][subfeature] = file

    return data_name, minmax_name

class UVANEMODataGenerator(torch.utils.data.Dataset):
    def __init__(self, label_path, videos_path, videos_frequency, features_path, vgg_path, feature_list, test = False, calcminmax_features = False):
        self.label_path = label_path
        self.test = test
        self.calcminmax_features = calcminmax_features
        self.feature_list = feature_list
        self.videos_frequency = videos_frequency

        self.videos_path = videos_path
        self.__init_video_data()

        self.features_path = features_path
        self.__init_features_data()

        self.vgg_path = vgg_path
        self.__init_vgg_data()

        self.__init_whole_dataset()

    def __init_video_data(self):
        self.videos_data = zipfile.ZipFile(self.videos_path)
        self.videos_data_names = dict()
        self.frame_scale = 255

        self.videos_data_names, self.video_max_len = videos_zip_to_dict(self.videos_data)
        self.video_max_len = self.video_max_len//self.videos_frequency + 2

    def __init_features_data(self):
        self.features_data = zipfile.ZipFile(self.features_path)
        self.features_data_names, _ = features_zip_to_dict(self.features_data, self.calcminmax_features)

        return

    def __init_vgg_data(self):
        return

    def __init_whole_dataset(self):
        self.data_total = 0
        self.index_name_dic = dict()

        with open(self.label_path) as f:
            labels = json.load(f)

        for index, (k, v) in enumerate(labels.items()):
            self.index_name_dic[index] = [k, v]
        self.data_total = index + 1

    def __len__(self):
        return self.data_total

    def __getitem__(self,idx):
        ids = self.index_name_dic[idx]
        name, label = ids

        # Label
        y = np.zeros(1)
        y[0] = label

        # Frames
        frames_tensors_scaled = numpy.empty(1)
        video_len = 0
        if "videos" in self.feature_list:
            frames_tensors, video_len = load_frames(self.videos_data_names, self.videos_data, name, self.videos_frequency, self.video_max_len)
            frames_tensors_scaled = frames_tensors / self.frame_scale

        # Dynamic Features
        dynamics_tensors = dict()
        for key in self.features_data_names[list(self.features_data_names)[0]]:
            dynamics_tensors.update({key: numpy.empty(1)})
        frames_len = 0

        if "aus" in self.feature_list:
            tensors, frames_len = load_features_aus(self.features_data_names, self.features_data, name, self.videos_frequency, self.video_max_len, video_len)
            dynamics_tensors.update({"action_units": tensors})
        if "si" in self.feature_list:
            tensors, frames_len = load_features_si(self.features_data_names, self.features_data, name, self.videos_frequency, self.video_max_len, video_len)
            dynamics_tensors.update({"smile_intensities": tensors})
        if "d1da27" in self.feature_list:
            key = "dynamics_delta_adjusted_27"
            tensors, frames_len = load_features_dynamics(self.features_data_names, self.features_data, name, key, 150, self.videos_frequency, self.video_max_len, video_len)
            dynamics_tensors.update({key: tensors})
        if "d2da27" in self.feature_list:
            key = "dynamics_2nd_delta_adjusted_27"
            tensors, frames_len = load_features_dynamics(self.features_data_names, self.features_data, name, key, 100, self.videos_frequency, self.video_max_len, video_len)
            dynamics_tensors.update({key: tensors})
        return frames_tensors_scaled, y, video_len, dynamics_tensors, frames_len