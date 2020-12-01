import cv2
import numpy as np
from PIL import Image
import os

#data set path
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        #convert to greyscale and set up numpy array for images
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        #extract user id's
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        # detect faces
        faces = detector.detectMultiScale(img_numpy)
        for(x,y,w,h) in faces:
            #add detected faces and user id to arrays
            faceSamples.append(img_numpy[y:y+h,x:x+h])
            ids.append(id)

    return faceSamples,ids
print("\n [INFO] Training faces. It will take a few seconds. wait ...")
# preprocess images
faces,ids = getImagesAndLabels(path)
# initiate transfer learning 
recognizer.train(faces,np.array(ids))
# save new model
recognizer.write('trainer.yml')
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
