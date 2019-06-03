
# coding: utf-8

# In[5]:


import numpy as np 
import cv2
import pickle


#eye_cascade = cv2.CascadeClassifier(r'C:\Users\dell\Anaconda3\Library\etc\haarcascades\haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier(r'C:\Users\dell\Anaconda3\Library\etc\haarcascades\haarcascade_frontalface_alt2.xml')
recognizer =cv2.face.LBPHFaceRecognizer_create() #cv2.face.createLBPHFaceRecognizer()#
#smile_cascade = cv2.CascadeClassifier(r'C:\Users\dell\Anaconda3\Library\etc\haarcascades\haarcascade_smile.xml')


recognizer.read("trainner.yml")
labels={"person_name": 1}

with open("labels.pickle", 'rb') as f:
    ogLabels = pickle.load(f)
    labels = {v:k for k,v in ogLabels.items()}

cap =cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
#    img = np.full((100,80,3), 12, dtype = np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray , scaleFactor=1.5  ,minNeighbors = 5)
    for(x, y, w, h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        id_ , conf = recognizer.predict(roi_gray)
        #print(conf)
        #print(id_)
        if (conf >= 4) and (conf <= 85):
            #print(id_)
            #print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color=(255,255,255)
            stroke= 2
            cv2.putText(frame,name,(x,y), font ,1 ,color ,stroke ,cv2.LINE_AA)
        

        img_item = "7.png"
        cv2.imwrite(img_item,roi_gray)
        color = (72,120 ,0)
        stroke = 4
        cv2.rectangle(frame, (x,y) , (x+w, y+h),color , stroke) 
        #subitems = smile_cascade.detectMultiScale(roi_gray)
        #for(ex,ey,ew,eh) in subitems:
            #cv2.rectangle(roi_color,(ex,ey),(ex+ew , ey+eh) , (8,255,8) ,2)


    cv2.imshow('frame' ,frame)
    if cv2.waitKey(20) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#roi  region of interest


# In[4]:


import os
import numpy as np
from PIL import Image
import cv2
import pickle

#baseDirectory  = os.path.dirname(os.path.abspath(__file__))

imageDirectory = os.path.dirname(r"C:\Users\dell\Downloads\images")

face_cascade = cv2.CascadeClassifier(r'C:\Users\dell\Anaconda3\Library\etc\haarcascades\haarcascade_frontalface_alt2.xml')

recognizer =cv2.face.LBPHFaceRecognizer_create() #cv2.face.creaeLBPHFaceRecognizer()#

currentId = 0
label_ids = {}
yLabel = []
xTrain = []

for root , dirs , files in os.walk(imageDirectory):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" ","-").lower()
            #os.path.dirname(path)  can be written as root
            #print(label,path)

            if label in label_ids:
                pass
            else:
                label_ids[label] = currentId
                currentId+=1
            id_=label_ids[label]
            #print(label_ids)

           # yLabel.append(label)
            #xTrain.append(path)
            pilImage = Image.open(path).convert("L")
            size = (550, 550)
            final_image = pilImage.resize(size,Image.ANTIALIAS)
            imageArray = np.array(pilImage, "uint8")
            #print(imageArray)
            faces = face_cascade.detectMultiScale(imageArray, scaleFactor=1.5  ,minNeighbors = 5)

            #print(id_)
            for (x,y,w,h) in faces:
                roi = imageArray[y:y+h , x:x+h]
                xTrain.append(roi)
                yLabel.append(id_)

print(yLabel)
print(xTrain)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids , f)

recognizer.train(xTrain , np.array(yLabel))
recognizer.save("trainner.yml")

