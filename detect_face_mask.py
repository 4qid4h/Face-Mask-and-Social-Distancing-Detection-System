from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import time
import cv2
import os

# Untuk detect face dalam video 
def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]   # height and width of the input frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0)) # creates a blob (binary large object) from the input frame

    # Pass the blob through faceNet utk detect face
    faceNet.setInput(blob)
    detections = faceNet.forward()
    # print(detections.shape)

    faces = [] # untuk store face image
    locs = []  # untuk store face positions
    preds = [] # Untuk store mask predictions

    # Loop (face mask ni detect secara loop 'berterusan')
    for i in range(0, detections.shape[2]):
        # Untuk extract the confidence (probability) associated with the detection.
        # confidence level utk tunjuk tahap keyakinan yang dibuat oleh detection model
        confidence = detections[0, 0, i, 2] 

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence (min 0.5)
        if confidence > 0.5:

            # Extract the bounding box coordinates and scale them with the frame dimensions.
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int") # Convert bounding box coordinates to integers

            # utk pastikan the bounding box coordinates fall within the frame dimensions
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX] # Extract the face region of interest (ROI) from the frame
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) # Convert the color format of the face from BGR to RGB.
            face = cv2.resize(face, (224, 224)) # Resize the face to a fixed size of 224x224 pixels
            face = img_to_array(face) # Convert the face to a NumPy array
            face = preprocess_input(face) # Preprocess the face for the model

            faces.append(face) # Add the preprocessed face to the faces list
            locs.append((startX, startY, endX, endY)) # Add the bounding box coordinates to the locs list

    # Akan buat prediction if at least one face are detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32") # Convert the faces list (containing detected face images) into a NumPy array with data type float32
        preds = maskNet.predict(faces, batch_size=32) # Use the mask detection model (maskNet) to predict whether each face is wearing a mask.

    return (locs, preds) # Return 2-tuple 

# Utk load pre-trained face detection model
prototxtPath = r"face_detector\deploy.prototxt"   # deploy.prototxt is text-based file used to define the architecture or network structure of a deep learning model
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"   # this file contains the pre-trained weights or parameters of a deep neural network model used for face detection
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath) # function from the OpenCV library used to load deep neural network models.

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# Start kamera
print("[INFO] starting video stream...")
vs = cv2.VideoCapture("rtsp://192.168.0.1")

# loop over the frames from the video stream
while True:
    ret, frame = vs.read() # Reads a frame from the video stream
    if not ret:
        break
    
    frame = imutils.resize(frame, width=400) # Resizes the frame to have a maximum width of 400 pixels using the imutils library.

    # detect faces in the frame and determine if they are wearing face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        label = "Mask" if mask > withoutMask else "No Mask"   # Label kan class "Mask" and "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)  # Set colour bounding box

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100) # Probability in label dekat output box

        # display the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Untuk close output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("a"):
        break

cv2.destroyAllWindows() # Closes all OpenCV windows
vs.release() # Releases the video stream

