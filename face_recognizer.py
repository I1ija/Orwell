import csv
import pandas as pd
import os
from glob import glob
import numpy as np
import cv2
import mediapipe as mp
from datetime import datetime, timedelta, date
import time

det = mp.solutions.face_detection
drawing = mp.solutions.drawing_utils

my_list = []

def get_bbox_from_detection(detection, frame_shape):
    box = detection.location_data.relative_bounding_box
    box = np.array([box.xmin, box.ymin, box.width, box.height])

    imH, imW, _ = frame_shape
    box = box * np.array([imW, imH, imW, imH])
    box = box.astype(np.int32)

    box[:2] = [max(box[i], 0) for i in range(2)]

    return box

def draw_bbox(frame, box):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

def get_feature_vector_from_bb(model, frame, box):
    x, y, w, h = box
    face_image = frame[y:y+h, x:x+w, :]
    feature = model.feature(face_image)

    return feature

def get_features_from_images(model, faceDetector): # , imgFolder
    imgFiles = glob('./faces/*.jpg')

    features, names = [], []
    for imgFile in imgFiles:
        img = cv2.imread(imgFile)

        img = img[:, :, ::-1]  #bgr2rgb
        results = faceDetector.process(img)
        img = img[:, :, ::-1] #rgb2bgr
        
        box = get_bbox_from_detection(results.detections[0], img.shape) # samo 1 lice uzimamo; iz baze

        imgFeature = get_feature_vector_from_bb(model, img, box)
        features.append(imgFeature)

        _, name = os.path.split(imgFile) # _ = imgPath; u 'name' zadnje iza '/'
        names.append(name)

    return features, names

def find_best_match(model, features, names, query_feature):
    max_score = 0
    match_name = ''
    for feature, name in zip(features, names): # zip stvara tupleove, parove; generator/iterator
        score = model.match(feature, query_feature, 0) # 0-cos distance, 1-euklid
        if score > max_score:
            max_score = score
            match_name = name

    return match_name, max_score


face_det = det.FaceDetection(model_selection=0, min_detection_confidence=0.9999)
model = cv2.FaceRecognizerSF.create(
    model = "./models/resnet18.onnx",
    config = '',
    backend_id = 0, 
    target_id = 0 
)
features, names = get_features_from_images(model, face_det)

cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    frame.flags.writeable = False
    frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = face_det.process(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.detections == None: continue

    for detection in results.detections:
        box = get_bbox_from_detection(detection, frame.shape)
        draw_bbox(frame, box)

        query_feature = get_feature_vector_from_bb(model, frame, box)
        match_name, max_score = find_best_match(model, features, names, query_feature)

        x, y, _, _ = box
        cv2.putText(frame, f"Matched: {match_name}", org=(x, y - 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)

         
    cv2.imshow('Orwellq', frame)
   
    if cv2.waitKey(1) & 0xFF==ord('q'):
        dict = {'name': [], 'time':[] }
        time1 = time.asctime()
         
        dict['name'].append(match_name)
        dict["time"].append(time1)
        
        # open file for writing, "w" is writing
        csv_file = "output.csv"
        csv_columns = ['Name','Time']

        
        dict_copy = dict.copy()
        my_list.append(dict_copy)
        
        print(my_list)

        keys =  my_list[0].keys()

        with open('output.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(my_list)
                    

        


        
    cv2.waitKey(5)

cv2.destroyAllWindows()
cap.release()

########### faceDet iz mediapipea detektira, a 'nas' transferlearningModel recognizea