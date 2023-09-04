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
    """Reads video frame from zip file

    :param zip: Zip file to read from
    :param name: Name of file to read

    :returns: Video frame data
    """
    return Image.open(io.BytesIO(zip.read(name)))

def read_au_txt(zip,name):
    """Reads AUDA sequential features (AU based) from zip file

    :param zip: Zip file to read from
    :param name: Name of file to read

    :returns: AU based features data
    """
    return pd.read_csv(io.BytesIO(zip.read(name)), sep=",")

def read_si_txt(zip,name):
    """Reads AUDA sequential features (smile intensities) from zip file

    :param zip: Zip file to read from
    :param name: Name of file to read

    :returns: Smile intensities features data
    """
    return pd.read_csv(io.BytesIO(zip.read(name)), sep="\n", header=None)

def read_auwise_txt(zip,name):
    """Reads AUDA features (AU-wise and cross-AU) from zip file

    :param zip: Zip file to read from
    :param name: Name of file to read

    :returns: AUDA features data
    """
    return pd.read_csv(io.BytesIO(zip.read(name)), sep=",", header=None)



def load_frames(videos_data_names, videos_data, name, videos_frequency, video_max_len):
    """Loads video frames to be used in model

    :param name: Requested name
    :param videos_data: Zip file to read from
    :param videos_data_names: Dictionary containing names of all video frames for a video sequence
    :param videos_frequency: FPS used in videos
    :param video_max_len: Length of longest video

    :returns: Video frames data and their length
    """
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

def load_features_si(files, features_data, key, videos_frequency, video_max_len):
    """Loads smile intensities features to be used in model

    :param key: Requested name
    :param features_data: Zip file to read from
    :param files: Dictionary containing names of all AUDA package features for a video sequence
    :param videos_frequency: FPS used in videos
    :param video_max_len: Length of longest video

    :returns: Smile intensities features and their length
    """
    pd = read_si_txt(features_data, files[key])

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

def load_features_aus(files, features_data, aus_feature_name, videos_frequency, sigmoid_mul, sigmoid_add, video_max_len):
    """Loads AU based features to be used in model

    :param key: Requested name
    :param aus_feature_name: Requested AU based feature name
    :param features_data: Zip file to read from
    :param files: Dictionary containing names of all AUDA package features for a video sequence
    :param videos_frequency: FPS used in videos
    :param video_max_len: Length of longest video
    :param sigmoid_mul: Sigmoid data normalization modifier -> narrows or expands the curve
    :param sigmoid_add: Sigmoid data normalization modifier -> offsets the curve
    :returns: AU based features and their length
    """
    pd = read_au_txt(features_data, files[aus_feature_name])
    columns = sorted(pd.columns.values.tolist())

    pad_x = np.zeros((video_max_len, len(columns)))

    last = [files[-1]] if len(files) % videos_frequency >= 2 / 3 * videos_frequency else []
    start = 0
    values = [np.concatenate([pd[i].values[start::videos_frequency], last]) for i in columns]
    frame_count = len(values[0])

    x = np.array(values).transpose()
    x = 1/(1 + np.exp(-(x * sigmoid_mul + sigmoid_add))) # Translacja elementów (dynamiki natężenia action unitów) - zawierają niskie wartości ujemne i wysokie dodatnie - do tensorów warto zastosować moim zdaniem sigmoid
    pad_x[0:frame_count] = x

    return pad_x, frame_count

def load_features_auwise(files, features_data, sigmoid_mul, sigmoid_add):
    """Loads AU-wise features to be used in model

    :param features_data: Zip file to read from
    :param files: Dictionary containing names of all AUDA package features for a video sequence
    :param sigmoid_mul: Sigmoid data normalization modifier -> narrows or expands the curve
    :param sigmoid_add: Sigmoid data normalization modifier -> offsets the curve

    :returns: AU-wise features
    """

    x = []
    for auwise_feature in ["AU-wise_apex", "AU-wise_offset", "AU-wise_onset", "AU-wise_whole_sequence", "AU-wise_whole_smile"]:
        pd = read_auwise_txt(features_data, files[auwise_feature])

        values = pd.values[0]
        x.append(values)

    x = np.array(x)
    x = 1/(1 + np.exp(-(x * sigmoid_mul + sigmoid_add)))
    return x

def load_features_crossau(files, features_data):
    """Loads cross-AU features to be used in model

    :param features_data: Zip file to read from
    :param files: Dictionary containing names of all AUDA package features for a video sequence

    :returns: Cross-AU features
    """
    x = []
    for auwise_feature in ["cross-AU_apex", "cross-AU_offset", "cross-AU_onset", "cross-AU_whole_sequence", "cross-AU_whole_smile"]:
        pd = read_auwise_txt(features_data, files[auwise_feature])

        values = pd.values[0]
        x.append(values)

    x = np.array(x)
    x = np.tanh(x)
    return x

def videos_zip_to_dict(data):
    """Prepares dictionary containing names of all video frames for a video sequence

    :param data: Zip file to read from

    :returns: Dictionary containing names of all video frames for a video sequence and length of longest video
    """
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
    """Prepares dictionary containing names of all AUDA package features for a video sequence

    :param data: Zip file to read from

    :returns: Dictionary containing names of all AUDA package features for a video sequence and length of longest video
    """
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
                (minval, meanval, maxval, varval, totalval) = (pddata_values.min(), pddata_values.mean(), pddata_values.max(), pddata_values.var(), pddata_values.size)

                if feature_name in minmax_name:
                    (prev_minval, prev_meanval, prev_maxval, prev_varval, meancnt, prev_totalval) = minmax_name[feature_name]

                    new_minval = min(prev_minval, minval)
                    new_maxval = max(prev_maxval, maxval)
                    new_total = prev_totalval + totalval

                    new_meanval = (prev_totalval * prev_meanval + totalval * meanval) / new_total

                    mean_of_var = (prev_totalval * prev_varval + totalval * varval) / new_total
                    var_of_means = (prev_totalval * (prev_meanval - new_meanval) ** 2 + totalval * (meanval - new_meanval) ** 2) / new_total
                    new_varval = mean_of_var + var_of_means #https://math.stackexchange.com/a/4567292

                    minmax_name[feature_name] = (new_minval, new_meanval, new_maxval, new_varval, meancnt+1, new_total)
                else:
                    minmax_name[feature_name] = (minval, meanval, maxval, varval, 1, totalval)

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
                pddata = pd.read_csv(io.BytesIO(data.read(file)), sep=",|\\n", header=None)
                pddata_values = pddata.values
                (minval, meanval, maxval, varval, totalval) = (pddata_values.min(), pddata_values.mean(), pddata_values.max(), pddata_values.var(), pddata_values.size)

                if subfeature in minmax_name:
                    (prev_minval, prev_meanval, prev_maxval, prev_varval, meancnt, prev_totalval) = minmax_name[subfeature]

                    new_minval = min(prev_minval, minval)
                    new_maxval = max(prev_maxval, maxval)
                    new_total = prev_totalval + totalval

                    new_meanval = (prev_totalval * prev_meanval + totalval * meanval) / new_total

                    mean_of_var = (prev_totalval * prev_varval + totalval * varval) / new_total
                    var_of_means = (prev_totalval * (prev_meanval - new_meanval) ** 2 + totalval * (meanval - new_meanval) ** 2) / new_total
                    new_varval = mean_of_var + var_of_means  # https://math.stackexchange.com/a/4567292

                    minmax_name[subfeature] = (new_minval, new_meanval, new_maxval, new_varval, meancnt + 1, new_total)
                else:
                    minmax_name[subfeature] = (minval, meanval, maxval, varval, 1, totalval)

            data_name[name][subfeature] = file

    return data_name, minmax_name

class UVANEMODataGenerator(torch.utils.data.Dataset):
    """Dataset of UVA-Nemo and AUDA features"""
    def __init__(self, label_path, videos_path, videos_frequency, features_path, feature_list, test = False, calcminmax_features = False):
        self.label_path = label_path
        self.test = test
        self.calcminmax_features = calcminmax_features
        self.feature_list = feature_list
        self.videos_frequency = videos_frequency

        self.videos_path = videos_path
        self.__init_video_data()

        self.features_path = features_path
        self.__init_features_data()

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

        # AUDA Features
        dynamics_tensors = dict()
        for key in self.features_data_names[list(self.features_data_names)[0]]:
            dynamics_tensors.update({key: numpy.empty(1)})
        frames_len = 0

        if "aus" in self.feature_list:
            key = "action_units"
            tensors, frames_len = load_features_aus(self.features_data_names[name], self.features_data, key, self.videos_frequency, 1, -0.75, self.video_max_len)
            dynamics_tensors.update({key: tensors})
        if "si" in self.feature_list:
            key = "smile_intensities"
            tensors, frames_len = load_features_si(self.features_data_names[name], self.features_data, key, self.videos_frequency, self.video_max_len)
            dynamics_tensors.update({key: tensors})

        if "d2d27" in self.feature_list:
            key = "dynamics_2nd_delta_27"
            tensors, frames_len = load_features_aus(self.features_data_names[name], self.features_data, key, self.videos_frequency, 100, 0, self.video_max_len)
            dynamics_tensors.update({key: tensors})
        if "d2d9" in self.feature_list:
            key = "dynamics_2nd_delta_9"
            tensors, frames_len = load_features_aus(self.features_data_names[name], self.features_data, key, self.videos_frequency, 10, 0, self.video_max_len)
            dynamics_tensors.update({key: tensors})
        if "d2da27" in self.feature_list:
            key = "dynamics_2nd_delta_adjusted_27" #fixed
            tensors, frames_len = load_features_aus(self.features_data_names[name], self.features_data, key, self.videos_frequency, 100, 0, self.video_max_len)
            dynamics_tensors.update({key: tensors})
        if "d2da9" in self.feature_list:
            key = "dynamics_2nd_delta_adjusted_9"
            tensors, frames_len = load_features_aus(self.features_data_names[name], self.features_data, key, self.videos_frequency, 10, 0, self.video_max_len)
            dynamics_tensors.update({key: tensors})

        if "d1d27" in self.feature_list:
            key = "dynamics_delta_27"
            tensors, frames_len = load_features_aus(self.features_data_names[name], self.features_data, key, self.videos_frequency, 5, 0, self.video_max_len)
            dynamics_tensors.update({key: tensors})
        if "d1d9" in self.feature_list:
            key = "dynamics_delta_9"
            tensors, frames_len = load_features_aus(self.features_data_names[name], self.features_data, key, self.videos_frequency, 3, 0, self.video_max_len)
            dynamics_tensors.update({key: tensors})
        if "d1da27" in self.feature_list:
            key = "dynamics_delta_adjusted_27" #fixed
            tensors, frames_len = load_features_aus(self.features_data_names[name], self.features_data, key, self.videos_frequency, 5, 0, self.video_max_len)
            dynamics_tensors.update({key: tensors})
        if "d1da9" in self.feature_list:
            key = "dynamics_delta_adjusted_9"
            tensors, frames_len = load_features_aus(self.features_data_names[name], self.features_data, key, self.videos_frequency, 3, 0, self.video_max_len)
            dynamics_tensors.update({key: tensors})


        if "auwise" in self.feature_list:
            key = "auwise"
            tensors = load_features_auwise(self.features_data_names[name], self.features_data, 1, -0.5)
            dynamics_tensors.update({key: tensors})
        if "crossau" in self.feature_list:
            key = "crossau"
            tensors = load_features_crossau(self.features_data_names[name], self.features_data)
            dynamics_tensors.update({key: tensors})


        return idx, name, frames_tensors_scaled, y, video_len, dynamics_tensors, frames_len