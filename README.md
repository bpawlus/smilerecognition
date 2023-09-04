# smilerecognition
Project was carried out as part of the Master's Thesis - Deep convolutional neural networks for facial expression recognition in video sequences.

The solution contains several script files. The following can be run using python3 with the appropriate parameters:

- extract_movie_frames.py:  builds zip with face-focused video frames in form of png files, from UVA-NEMO database. Additionaly builds json file with all read videos.
- start.py: starts process of single and concatenation models training and validation
- split_labels.py: splits predicted labels between spontaneus and deliberate ones in single trained single models splits predicted labels between spontaneus and deliberate ones
- get_best_models.py: summarizes trained models and outputs saved parameters for best models
- concat_test.py: performs calculation of best models to concatenate with starting model

For details of parameter usages, it is advised to use: `python FILE -h`