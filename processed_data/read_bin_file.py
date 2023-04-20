"""
File shows how to read binary file with features for uva-nemo datasets
"""
import struct
import numpy as np

path = "PATH TO BINARY FILE WITH FEATURES"


def frame_features_generator(path):
    with open(path, "rb") as binary_file:
        n_frames = struct.unpack("i", binary_file.read(4))[0]
        n_features = struct.unpack("i", binary_file.read(4))[0]
        h = struct.unpack("i", binary_file.read(4))[0]
        w = struct.unpack("i", binary_file.read(4))[0]
        for _ in range(n_frames):
            feature_array = np.zeros((n_features, h, w))
            for x in range(h):
                for y in range(w):
                    features = np.array([struct.unpack_from("f", binary_file.read(4))[0] for _ in range(n_features)])
                    feature_array[:, x, y] = features
            yield feature_array


i = 0
for frame_feature in frame_features_generator(path):
    print(frame_feature.shape)
    i += 1
print(i)
