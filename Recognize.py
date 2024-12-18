"""import datetime
import os
import time
import cv2 # type: ignore
import dlib # type: ignore
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_blink(shape, predictor):
    (lStart, lEnd) = (42, 48)  # Right eye landmarks
    (rStart, rEnd) = (36, 42)  # Left eye landmarks
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return ear < 0.2  # You might need to adjust this threshold

def recognize_attendence():
    print("Inside the recognize function")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("./TrainingImageLabel/Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("StudentDetails"+os.sep+"StudentDetails.csv")
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'IN-Time', 'OUT-Time']
    attendance = pd.DataFrame(columns=col_names)
    print("Dataframe Reached")

    # Initialize dlib's face detector and predictor
    detector = dlib.get_frontal_face_detector()
    print("detector: ", detector)
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print(predictor)
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height
    print("Video Capturing Started")
    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    # Dictionary to map detected faces to IDs
    face_id_map = {}
    i=0

    while True:
        _, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(int(minW), int(minH)), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (10, 159, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]

            # Recognize face
            Id, conf = recognizer.predict(roi_gray)

            if conf < 100:
                aa = df.loc[df['Id'] == Id]['Name'].values
                confstr = "  {0}%".format(round(100 - conf))
                tt = str(Id) + "-" + aa
                # Add face to dictionary
                face_id_map[(x, y, w, h)] = Id
            else:
                Id = '  Unknown  '
                tt = str(Id)
                confstr = "  {0}%".format(round(100 - conf))

            # Check for blink detection using dlib
            dlib_rects = detector(gray, 0)
            blink_detected = False
            for rect in dlib_rects:
                shape = predictor(gray, rect)
                shape = np.array([[p.x, p.y] for p in shape.parts()])
                if detect_blink(shape, predictor):
                    blink_detected = True
                    break

            # Retrieve ID from dictionary
            if (x, y, w, h) in face_id_map:
                Id = face_id_map[(x, y, w, h)]

            if (100 - conf) > 10 and blink_detected:
                print("condition satisfied")
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = str(aa)[2:-2]
                attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]

            tt = str(tt)[2:-2]
            if (100 - conf) > 15 and blink_detected:
                tt = tt + "[Pass]"
                cv2.putText(im, str(tt), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                i=1
            else:
                cv2.putText(im, str(tt), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

            if (100 - conf) > 15:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 255, 0), 1)
            elif (100 - conf) > 7:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 255, 255), 1)
            else:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 0, 255), 1)

        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('Attendance', im)
        # if (cv2.waitKey(1) == ord('q')):
        #     break
        if i==1:
            break
    ts = time.time()
    print("About to END")
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = "Attendance" + os.sep + "Attendance_" + date + ".csv"
    attendance.to_csv(fileName, index=False, mode='a', header=not os.path.exists(fileName))  # Append mode, header only if file doesn't exist
    print("Attendance Successful")
    cam.release()
    cv2.destroyAllWindows()
"""
# recognize_attendence()

import datetime
import os
import time
import cv2  # type: ignore
import dlib  # type: ignore
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def detect_blink(shape, predictor):
    (lStart, lEnd) = (42, 48)  # Right eye landmarks
    (rStart, rEnd) = (36, 42)  # Left eye landmarks
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return ear < 0.2  # You might need to adjust this threshold


def recognize_attendence():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("./TrainingImageLabel/Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("StudentDetails" + os.sep + "StudentDetails.csv")
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'IN-Time', 'OUT-Time']
    attendance = pd.DataFrame(columns=col_names)

    # Initialize dlib's face detector and predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height
    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    # Dictionary to map detected faces to IDs
    face_id_map = {}
    i = 0

    while True:
        _, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray, 1.2, 5, minSize=(int(minW), int(minH)), flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (10, 159, 255), 2)
            roi_gray = gray[y : y + h, x : x + w]

            # Recognize face
            Id, conf = recognizer.predict(roi_gray)

            if conf < 100:
                aa = df.loc[df['Id'] == Id]['Name'].values[0]
                confstr = "  {0}%".format(round(100 - conf))
                tt = str(Id) + "-" + aa
                # Add face to dictionary
                face_id_map[(x, y, w, h)] = Id
            else:
                Id = '  Unknown  '
                tt = str(Id)
                confstr = "  {0}%".format(round(100 - conf))

            # Check for blink detection using dlib
            dlib_rects = detector(gray, 0)
            blink_detected = False
            for rect in dlib_rects:
                shape = predictor(gray, rect)
                shape = np.array([[p.x, p.y] for p in shape.parts()])
                if detect_blink(shape, predictor):
                    blink_detected = True
                    break

            # Retrieve ID from dictionary
            if (x, y, w, h) in face_id_map:
                Id = face_id_map[(x, y, w, h)]

            if (100 - conf) > 10 and blink_detected:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                # Check if the entry for this ID and date already exists
                existing_row = attendance[(attendance['Id'] == Id) & (attendance['Date'] == date)]
                if not existing_row.empty:
                    # Update the 'OUT-Time' of the existing entry
                    attendance.loc[(attendance['Id'] == Id) & (attendance['Date'] == date), 'OUT-Time'] = timeStamp
                else:
                    # Add a new entry with 'IN-Time' and 'OUT-Time' set to the current time
                    new_entry = pd.DataFrame([{'Id': Id, 'Name': aa, 'Date': date, 'IN-Time': timeStamp, 'OUT-Time': timeStamp}])
                    attendance = pd.concat([attendance, new_entry], ignore_index=True)

            tt = str(tt)[2:-2]
            if (100 - conf) > 15 and blink_detected:
                tt = tt + "[Pass]"
                cv2.putText(im, str(tt), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                i = 1
            else:
                cv2.putText(im, str(tt), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

            if (100 - conf) > 15:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 255, 0), 1)
            elif (100 - conf) > 7:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 255, 255), 1)
            else:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 0, 255), 1)

        cv2.imshow('Attendance', im)
        if i == 1:
            break

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = "Attendance" + os.sep + "Attendance_" + date + ".csv"
    attendance.to_csv(fileName, index=False, mode='a', header=not os.path.exists(fileName))  # Append mode, header only if file doesn't exist
    print("Attendance Successful: ", aa)
    cam.release()
    cv2.destroyAllWindows()


recognize_attendence()