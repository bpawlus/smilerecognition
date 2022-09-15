import os
import zipfile
import cv2
import dlib
from util  import out_put
from PIL import Image
from io import BytesIO
import json
from datetime import datetime

        
def face(img, alignment = False):
    rect = None

    try:
        rect = dlib_detector(img, 0)[0]
    except IndexError:
        return None
    
    if alignment:
        
        return img[rect.top():rect.bottom(),rect.left():rect.right()]
    
    faces = dlib.full_object_detections()
    faces.append(predictor(img, rect))
    img = dlib.get_face_chip(img, faces[0],224)
    return img



def write(flag,path,process,file_name,outputZipName):
    output_zip = zipfile.ZipFile(f"{outputZipName}.zip",mode = flag and "w" or "a")
    cap = cv2.VideoCapture(path)
    rec,frame = cap.read()
    
    c = 0
    while rec:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        if process == 0:
            img = frame
        elif process == 1:
            img = face(frame)
        elif process == 2:
            img = face(frame,True)
        
        if img is not None:
            assert frame.shape[-1] == 3
            img_file = BytesIO()
            Image.fromarray(img).save(img_file,"JPEG")
            name = os.path.join(f"{file_name}",f"{c}.jpg")
            output_zip.writestr(name,img_file.getvalue())
        
        rec,frame = cap.read()
        
        c+=1
        
    output_zip.close()  


def extract_frames(
        path,
        outputZipName,
        outputTrainTest,
        pretrained_modelpath = None,
        verbose  = "processes",
        process  = 0,
        database = None
        ):
    
    global dlib_detector
    global predictor
    
    #santity check, if process = 0, then pretrained_modelpath != None.
    assert  process in [0,1,2], "Please provide valid process number"
    assert  process == 0 or pretrained_modelpath != None, "Please provide the pretrained models, if data preprocessing required."
    assert  database in ["UVA-NEMO"], "Invalid database name"
    
    cs = 0
    flag = True
    dlib_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(f"{pretrained_modelpath}")
    
    if database == "UVA-NEMO":
        items = dict()

        for location,file in enumerate(sorted(os.listdir(path))):
            dt = datetime.timestamp(datetime.now())
            if not file.endswith(".mp4"):
                continue

            write(flag,os.path.join(path,file),process,file.replace(".mp4",""),outputZipName)

            if "deliberate" in file:
                items[os.path.splitext(file)[0]] = 0
            elif "spontaneous" in file:
                items[os.path.splitext(file)[0]] = 1
            else:
                items[os.path.splitext(file)[0]] = "?"

            flag = False

            dt2 = datetime.timestamp(datetime.now())
            print("Processed", file, "in", round(dt2-dt, 3), "s")

            cs+= 1

        json_object = json.dumps(items, indent=4)
        with open(outputTrainTest, "w") as outfile:
            outfile.write(json_object)
        
        
        
        
        
        