'''import csv
import cv2 # type: ignore
import os

import os.path

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def takeImages():
    Id = input("Enter Your Id: ")
    name = input("Enter Your Name: ")

    if is_number(Id) and name.isalpha():
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(int(min(w)),int(min(h))), flags=cv2.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (10, 159, 255), 2)
                sampleNum += 1

                # Resize the detected face region to a fixed size before saving
                resized_face = cv2.resize(gray[y:y + h, x:x + w], (190, 190))
                
                cv2.imwrite("TrainingImage" + os.sep + name + "." + Id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                
                cv2.imshow('frame', img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum > 100:
                break

        cam.release()
        cv2.destroyAllWindows()
        res = "Images Saved for ID : " + Id + " Name : " + name
        header = ["Id", "Name"]
        row = [Id, name]
        if(os.path.isfile("StudentDetails"+os.sep+"StudentDetails.csv")):
            with open("StudentDetails"+os.sep+"StudentDetails.csv", 'a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(j for j in row)
            csvFile.close()
        else:
            with open("StudentDetails"+os.sep+"StudentDetails.csv", 'a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(i for i in header)
                writer.writerow(j for j in row)
            csvFile.close()
    else:
        if is_number(Id):
            print("Enter Alphabetical Name")
        if name.isalpha():
            print("Enter Numeric ID")'''


# import csv
# import cv2  # type: ignore
# import os

# def is_number(s):
#     try:
#         float(s)
#         return True
#     except ValueError:
#         pass

#     try:
#         import unicodedata
#         unicodedata.numeric(s)
#         return True
#     except (TypeError, ValueError):
#         pass

#     return False

# def takeImages():
#     Id = input("Enter Your Id: ")
#     name = input("Enter Your Name: ")

#     if is_number(Id) and name.isalpha():
#         cam = cv2.VideoCapture(0)
#         harcascadePath = "haarcascade_frontalface_default.xml"
#         detector = cv2.CascadeClassifier(harcascadePath)
#         sampleNum = 0

#         while True:
#             ret, img = cam.read()
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32,42), flags=cv2.CASCADE_SCALE_IMAGE)
#             for (x, y, w, h) in faces:
#                 # Adjust rectangle thickness based on face size of the individual, it is done using w and h where w and h is width and height respectively 
#                 thickness = max(2, ((w * h)  // 1000))
#                 cv2.rectangle(img, (x, y), (x + w, y + h), (10, 159, 255), thickness)
#                 sampleNum += 1
            
#                 # Resize the detected face region to a fixed size before saving
#                 resized_face = cv2.resize(gray[y:y + h, x:x + w], (190, 190))
#                 cv2.imwrite("TrainingImage" + os.sep + name + "." + Id + '.' + str(sampleNum) + ".jpg", resized_face)
                
#                 cv2.imshow('frame', img)
#             if cv2.waitKey(100) & 0xFF == ord('q'):
#                 break
#             elif sampleNum > 100:
#                 break

#         cam.release()
#         cv2.destroyAllWindows()
#         res = "Images Saved for ID : " + Id + " Name : " + name
#         header = ["Id", "Name"]
#         row = [Id, name]
#         if os.path.isfile("StudentDetails" + os.sep + "StudentDetails.csv"):
#             with open("StudentDetails" + os.sep + "StudentDetails.csv", 'a+', newline='') as csvFile:
#                 writer = csv.writer(csvFile)
#                 writer.writerow(row)
#         else:
#             with open("StudentDetails" + os.sep + "StudentDetails.csv", 'w', newline='') as csvFile:
#                 writer = csv.writer(csvFile)
#                 writer.writerow(header)
#                 writer.writerow(row)
#     else:
#         if not is_number(Id):
#             print("Enter Numeric ID")
#         if not name.isalpha():
#             print("Enter Alphabetical Name")


# takeImages()


import csv
import cv2  # type: ignore
import os
import sys

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def takeImages(Id, name):
    print(f"Received ID: {Id}, Name: {name}")  # Debug print

    if is_number(Id) and name.isalpha():
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 42), flags=cv2.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in faces:
                thickness = max(2, ((w * h) // 1000))
                cv2.rectangle(img, (x, y), (x + w, y + h), (10, 159, 255), thickness)
                sampleNum += 1
            
                resized_face = cv2.resize(gray[y:y + h, x:x + w], (190, 190))
                cv2.imwrite("TrainingImage" + os.sep + name + "." + Id + '.' + str(sampleNum) + ".jpg", resized_face)
                
                cv2.imshow('frame', img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum > 100:
                break

        cam.release()
        cv2.destroyAllWindows()
        res = "Images Saved for ID : " + Id + " Name : " + name
        header = ["Id", "Name"]
        row = [Id, name]
        if os.path.isfile("StudentDetails" + os.sep + "StudentDetails.csv"):
            with open("StudentDetails" + os.sep + "StudentDetails.csv", 'a+', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
        else:
            with open("StudentDetails" + os.sep + "StudentDetails.csv", 'w', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(header)
                writer.writerow(row)
    else:
        if not is_number(Id):
            print("Enter Numeric ID")
        if not name.isalpha():
            print("Enter Alphabetical Name")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        Id = sys.argv[1]
        name = sys.argv[2]
        takeImages(Id, name)
    else:
        print("Please provide ID and name as arguments.")






























"""import csv
import cv2 # type: ignore
import dlib # type: ignore
import os

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def takeImages():
    Id = input("Enter Your Id: ")
    name = input("Enter Your Name: ")

    if is_number(Id) and name.isalpha():
        cam = cv2.VideoCapture(0)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        sampleNum = 0

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
                thickness = max(2, ((w * h) // 1000))
                cv2.rectangle(img, (x, y), (x + w, y + h), (10, 159, 255), thickness)
                sampleNum += 1

                # Resize the detected face region to a fixed size before saving
                resized_face = cv2.resize(gray[y:y + h, x:x + w], (190, 190))
                cv2.imwrite("TrainingImage" + os.sep + name + "." + Id + '.' + str(sampleNum) + ".jpg", resized_face)
                
                cv2.imshow('frame', img)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum > 100:
                break

        cam.release()
        cv2.destroyAllWindows()
        res = "Images Saved for ID : " + Id + " Name : " + name
        header = ["Id", "Name"]
        row = [Id, name]
        if os.path.isfile("StudentDetails" + os.sep + "StudentDetails.csv"):
            with open("StudentDetails" + os.sep + "StudentDetails.csv", 'a+', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
        else:
            with open("StudentDetails" + os.sep + "StudentDetails.csv", 'w', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(header)
                writer.writerow(row)
    else:
        if not is_number(Id):
            print("Enter Numeric ID")
        if not name.isalpha():
            print("Enter Alphabetical Name")

if __name__ == "__main__":
    takeImages()
"""