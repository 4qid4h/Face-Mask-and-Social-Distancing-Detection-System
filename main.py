from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np #for numerical operations
import imutils #for image processing functions
import time #for timing purposes
import cv2 
import os
import math #for mathematical operations

# Load face mask detector model and face detector
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath) #faceNet yg digunekan utk detect face dalam frame capture
maskNet = load_model("mask_detector.model") #Loaded with a pre-trained Keras model for mask detection

# Load YOLO model for social distancing
labelsPath = "./coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = "./yolov3.weights"
configPath = "./yolov3.cfg"
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath) #Load YOLOv3 model

#function untuk Face mask detection 
def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0)) # (Preprocess the Image)Convert the image to a blob 
    faceNet.setInput(blob) #Set the Blob as Input
    detections = faceNet.forward() #Forward Pass to Get Detections

    faces = []
    locs = []
    preds = []

    #Process Detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # (Mask Classification) 
            # Extract and Preprocess Each Face
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

#Euclidean distance calculation
def calculate_distance(box1, box2):
    x_dist = abs(box2[0] - box1[0])
    y_dist = abs(box2[1] - box1[1])
    distance = math.sqrt(x_dist**2 + y_dist**2) # (*)darab...**2 ialah kuasa 2
    return distance

#Start real-time(open camera)
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

#Main loop untuk real-time processing
while True:
    frame = vs.read() #read frames from video stream
    frame = imutils.resize(frame, width=1300) # resize frame kpd 1300 pixel
    (H, W) = frame.shape[:2]  #extracts frame dimensions

    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet) #call function detect

    #detecting person with YOLO
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob) #set blob sbagai input to YOLO
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.1 and classID == 0: #check confidence kene lebih dari 0.1 (class id 0 for person)
                box = detection[0:4] * np.array([W, H, W, H]) #calculate bounding box
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3) #utk identify close pair of people
    if len(idxs) > 0:
        idxs = idxs.flatten()

    close_pairs = []
    for i in range(len(idxs) - 1):
        for j in range(i + 1, len(idxs)):
            box1 = boxes[idxs[i]]
            box2 = boxes[idxs[j]]
            distance = calculate_distance(box1, box2)
            if distance < 700:  #tukar kat sini utk pixel jarak
                close_pairs.append((idxs[i], idxs[j]))

    #untuk face mask detection rectangle box colour
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255) #red / green
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    #untuk social distancing rectangle box colour
    color_alert = (0, 0, 255) #red is used for "No Mask"
    color_normal = (0, 255, 0) #Green is used for "Mask"

    for (i, j) in close_pairs:   #kalau kurang 1 meter / dekat
        (x, y, w, h) = boxes[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_alert, 2)
        cv2.putText(frame, "Red Alert", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_alert, 2)
        (x, y, w, h) = boxes[j]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_alert, 2)
        cv2.putText(frame, "Red Alert", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_alert, 2)

    for i in idxs: #lebih 1 meter / jauh
        if all((i != pair[0] and i != pair[1]) for pair in close_pairs):
            (x, y, w, h) = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_normal, 2)
            cv2.putText(frame, "Normal", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_normal, 2)

    cv2.imshow("Face Mask & Social Distancing Detection for Isolation Ward", frame)
    key = cv2.waitKey(1) & 0xFF

    #Tutup detection
    if key == ord("a"):
        break

cv2.destroyAllWindows()
vs.stop()
