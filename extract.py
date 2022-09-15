from data_preprocessing import extract_frames

extract_frames(path="videos", outputZipName="processed_data_post_extract/uva", outputTrainTest="processed_data_post_extract/train.json", pretrained_modelpath="landmarks/shape_predictor_68_face_landmarks.dat", database="UVA-NEMO", process=2)