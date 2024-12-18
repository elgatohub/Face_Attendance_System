# import os
# import time
# import cv2 # type: ignore
# import numpy as np
# from PIL import Image
# from threading import Thread




# # -------------- image labesl ------------------------

# def getImagesAndLabels(path):
#     # get the path of all the files in the folder
#     imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
#     # print(imagePaths)

#     # create empth face list
#     faces = []
#     # create empty ID list
#     Ids = []
#     # now looping through all the image paths and loading the Ids and the images
#     for imagePath in imagePaths:
#         # loading the image and converting it to gray scale
#         pilImage = Image.open(imagePath).convert('L')
#         # Now we are converting the PIL image into numpy array
#         imageNp = np.array(pilImage, 'uint8')
#         # getting the Id from the image
#         Id = int(os.path.split(imagePath)[-1].split(".")[1])
#         # extract the face from the training image sample
#         faces.append(imageNp)
#         Ids.append(Id)
#     return faces, Ids


# # ----------- train images function ---------------
# def TrainImages():

#     recognizer = cv2.face.LBPHFaceRecognizer()

#     faces = []  # Replace with actual face images
#     Id = []  # Replace with corresponding labels

#     harcascadePath = "haarcascade_frontalface_default.xml"
#     detector = cv2.CascadeClassifier(harcascadePath)

#     faces, Id = getImagesAndLabels("TrainingImage")
#     Thread(target = recognizer.train(faces, np.array(Id))).start()
    
#     # # Below line is optional for a visual counter effect
#     Thread(target = counter_img("TrainingImage")).start()
#     recognizer.save("TrainingImageLabel"+os.sep+"Trainner.yml")
#     print("All Images")

# # Optional, adds a counter for images trained (You can remove it)
# def counter_img(path):
#     imgcounter = 1
#     imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
#     for imagePath in imagePaths:
#         print(str(imgcounter) + " Images Trained", end="\r")
#         time.sleep(0.008)
#         imgcounter += 1






import os
import cv2
import numpy as np
from PIL import Image
import time
from threading import Thread

# Function to get images and corresponding labels (Ids)
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')  # Convert to grayscale
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids

# Function to train the face recognizer
def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Create LBPH Face Recognizer
    faces, Ids = getImagesAndLabels("TrainingImage")   # Load images and labels

    # Start training in a separate thread
    train_thread = Thread(target=recognizer.train, args=(faces, np.array(Ids)))
    train_thread.start()

    # Optionally, add a visual counter for images trained
    counter_thread = Thread(target=counter_img, args=("TrainingImage",))
    counter_thread.start()

    train_thread.join()  # Wait for training to complete
    counter_thread.join()  # Wait for counter to complete

    recognizer.save("TrainingImageLabel" + os.sep + "Trainner.yml")
    print("Training completed and saved.")

# Optional function: Adds a counter for images trained
def counter_img(path):
    img_counter = 0
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        img_counter += 1
        print(f"{img_counter} Images Trained", end="\r")
        time.sleep(0.1)  # Adjust sleep time as needed










#################################################
#################################################
#  Using dlib Frontal As well as dlib predictor #
#################################################
#################################################

"""import cv2 # type: ignore
import dlib # type: ignore
import numpy as np
import os
from imutils import face_utils

def get_images_and_labels(path, detector, predictor):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]
    face_samples = []
    ids = []

    for image_path in image_paths:
        img = cv2.imread(image_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_img)

        for face in faces:
            shape = predictor(gray_img, face)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(face)
            face_samples.append(gray_img[y:y + h, x:x + w])
            id = int(os.path.split(image_path)[-1].split(".")[1])
            ids.append(id)

    return face_samples, ids

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    path = 'TrainingImage'

    faces, ids = get_images_and_labels(path, detector, predictor)
    recognizer.train(faces, np.array(ids))
    recognizer.save('TrainingImageLabel/Trainer.yml')

if __name__ == "__main__":
    train_model()
"""


































# without dlib frontal just the predictor

"""import csv
import cv2 # type: ignore
import dlib # type: ignore
import numpy as np
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

def get_images_and_labels(path, detector, predictor):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]
    face_samples = []
    ids = []
    
    for image_path in image_paths:
        img = cv2.imread(image_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_img)
        
        for face in faces:
            shape = predictor(gray_img, face)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(face)
            face_samples.append(gray_img[y:y + h, x:x + w])
            id = int(os.path.split(image_path)[-1].split(".")[1])
            ids.append(id)
            
    return face_samples, ids

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    path = 'TrainingImage'
    
    faces, ids = get_images_and_labels(path, detector, predictor)
    recognizer.train(faces, np.array(ids))
    recognizer.save('TrainingImageLabel/Trainer.yml')

if __name__ == "__main__":
    train_model()
"""