import cv2
import os
import numpy as np
from PIL import Image
def dataset():
    cam=cv2.VideoCapture(0)
    cam.set(3,640)
    cam.set(4,480)
    face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_id=input('Enter ur id:')
    print('initializing and wait for camera')
    count=0
    while True:
        ret, img=cam.read()
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces=face_detector.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),2)
            count+=1
            cv2.imwrite('data/user.'+str(face_id)+'.'+str(count)+'.jpg',gray[y:y+h,x:x+w])
            cv2.imshow('image', img)

        k=cv2.waitKey(100) & 0xff
        if k==27:
            break
        elif count>=50:
            break
    print("Exiting program and cleaning stuff")
    cam.release()
    cv2.destroyAllWindows()

def training():
    path='data'

    recognizer= cv2.face.LBPHFaceRecognizer_create()
    detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
    

    def getImagesandLabels(path):
        imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
        faceSamples=[]
        ids=[]

        for imagePath in imagePaths:

            PIL_img=Image.open(imagePath).convert('L')
            img_numpy=np.array(PIL_img,'uint8')

            id=int(os.path.split(imagePath)[-1].split('.')[1])
            faces=detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+h])
                ids.append(id)

        return faceSamples,ids

    print("Training process is going on....Please wait..")
    faces,ids=getImagesandLabels(path)
    recognizer.train(faces, np.array(ids))

    recognizer.write('train/train.yml')

    print('{0} Faces Trained...Exiting program'.format(len(np.unique(ids))))

def recognition():
    recognizer=cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('train/train.yml')
    cascadePath='haarcascade_frontalface_default.xml'
    faceCascade=cv2.CascadeClassifier(cascadePath);

    font=cv2.FONT_HERSHEY_SIMPLEX

    id=0

    names=['None','Rupak','Jyothi','Found','Found','Found']

    cam=cv2.VideoCapture(0)
    cam.set(3,640)
    cam.set(4,480)

    minW=0.1*cam.get(3)
    minH=0.1*cam.get(4)

    while True:

        ret, img=cam.read()

        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW),int(minH)),
            )
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),2)
            id,confidence=recognizer.predict(gray[y:y+h,x:x+w])

            if(confidence<100):
                id=names[id]
                confidence='{0}%'.format(round(100-confidence))
            else:
                id='unknown'
                confidence='{0}%'.format(round(100-confidence))

            cv2.putText(img, str(id),(x+5,y-5),font,1,(255,255,255),2)
            cv2.putText(img, str(confidence),(x+5,y+h-5),font,1,(255,255,0),1)

        cv2.imshow('camera',img)

        k=cv2.waitKey(10)& 0xff
        if k==27:
            break

    print('Exiting Program and cleanup stuff')
    cam.release()
    cv2.destroyAllWindows()

    

dataset()
training()
recognition()
