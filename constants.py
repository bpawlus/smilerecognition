dirsdict = {
    "videos_dir": "data/videos",
    "frames_zip_dir": "processed_data/uva.zip",
    "features_zip_dir": "processed_data/features.zip",
    "vgg_zip_dir": "processed_data/vgg.zip",
    "train_json_dir": "processed_data/train.json",
    "landmarks_dir": "landmarks/shape_predictor_68_face_landmarks.dat",
    "labels_dir": "processed_data/label",
    "folds_dir": "processed_data/folds",
    "folds_orig_dir": "processed_data/folds_original",
    "train_dir": "models",
    "load_dir": "modelstocat"
}

valuesdict = {
    "batch_size_train": 32,
    "batch_size_valtest": 16,
    "frequency": 5,
    "epochs": 70, #70
    "learning_rate": 1e-3
}

#1e-4 - auwise
#4e-5 - aus
#2e-5 - delta adj
#2e-4 - si
#7e-4 - videos,
#5e-3 - concat type 1
#7e-5 - concat type 2
#5e-5 - crossau