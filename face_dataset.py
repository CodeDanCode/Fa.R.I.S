import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam.set(cv2.CAP_PROP_FPS,20)

# change face_id to add new user
face_id = 1
print("\n [INFO] Initializing face capture. look at the camera and wait ...")
count = 0
while(True):
    ret,img = cam.read()
    #change if needed. this depends on camera position
    img = cv2.flip(img,-1)
    #convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        count+=1
        #creates the image name with the user id and image count
        cv2.imwrite("dataset/User."+str(face_id)+'.'+str(count)+".jpg",gray[y:y+h,x:x+w])
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xff
    #press escape to exit program or exit when count hits threshold
    if k == 27:
        break
    #change int to perfered number of images
    elif count >= 200:
        break

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
