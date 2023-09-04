import os
import zipfile
import cv2
import dlib
from PIL import Image
from io import BytesIO
import json
from datetime import datetime
from constants import dirsdict
        
def get_face_from_image(img, alignment = False):
    """Gets face focus from an image

    :param img: Original image

    :returns: Faced focused image
    """
    rect = None

    try:
        rect = dlib_detector(img, 0)[0]
    except IndexError:
        return None
    
    if alignment:
        return img[rect.top():rect.bottom(), rect.left():rect.right()]
    
    faces = dlib.full_object_detections()
    faces.append(predictor(img, rect))
    img = dlib.get_face_chip(img, faces[0], 224)

    return img



def write_movie_frames(
        flag,
        path,
        video_name,
        output_zip_name
        ):
    """Writes frame images from video

    :param path: Path to video inside zip file
    :param video_name: Name of the video
    :param output_zip_name: Output zip with face-focused video frames
    """
    output_zip = zipfile.ZipFile(f"{output_zip_name}.zip", mode = flag and "w" or "a")

    cap = cv2.VideoCapture(path)
    rec,frame = cap.read()

    frame_id = 0
    while rec:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img = get_face_from_image(frame, True)
        
        if img is not None:
            assert frame.shape[-1] == 3
            img_file = BytesIO()
            Image.fromarray(img).save(img_file, "JPEG")
            name = os.path.join(f"{video_name}", f"{frame_id}.jpg")
            output_zip.writestr(name,img_file.getvalue())
        
        rec,frame = cap.read()
        frame_id+=1
        
    output_zip.close()  


def extract_frames_from_database(
        input_videos_dir,
        output_zip_dir,
        output_json_dir,
        input_landmarks_dir = None,
        ):
    """Builds zip with face-focused video frames in form of png files, from UVA-NEMO database that contains zip with mp4 files. Additionaly builds json file with all read videos

    :param input_videos_dir: Original UVA-NEMO database
    :param output_zip_dir: Output zip directory
    :param output_json_dir: Output json file with all read videos
    :param input_landmarks_dir: DLIB landmark detector
    """
    
    global dlib_detector
    global predictor

    assert input_landmarks_dir != None, "Please provide the pretrained models for shape predictors"

    count = 0
    flag = True
    dlib_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(f"{input_landmarks_dir}")

    items = dict()

    for location, file in enumerate(sorted(os.listdir(input_videos_dir))):
        time_start = datetime.timestamp(datetime.now())

        if not file.endswith(".mp4"):
            continue

        write_movie_frames(flag, os.path.join(input_videos_dir, file), file.replace(".mp4", ""), output_zip_dir)

        if "deliberate" in file:
            items[os.path.splitext(file)[0]] = 0
        elif "spontaneous" in file:
            items[os.path.splitext(file)[0]] = 1
        else:
            items[os.path.splitext(file)[0]] = "?"

        flag = False

        time_end = datetime.timestamp(datetime.now())
        count += 1

        print("Processed video no.", count, file, "in", round(time_end - time_start, 3), "s")

    json_object = json.dumps(items, indent=4)
    with open(output_json_dir, "w") as outfile:
        outfile.write(json_object) #plik wymaga ręcznej separacji na train,test,validate

if __name__ == "__main__":
    extract_frames_from_database(
        input_videos_dir=dirsdict["videos_dir"],
        output_zip_dir=dirsdict["frames_zip_dir"],
        output_json_dir=dirsdict["train_json_dir"],
        input_landmarks_dir=dirsdict["landmarks_dir"],
    )

        