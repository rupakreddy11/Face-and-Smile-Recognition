import cv2

faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade=cv2.CascadeClassifier('haarCascade_smile.xml')

cam=cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

while True:
    ret,img=cam.read()
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30,30)
        )
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),((x+w),(y+h)),(255,0,0),2)
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=img[y:y+h, x:x+w]
        
        smiles=smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=15,
            minSize=(25,25)
            )
        for(sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_color, (sx,sy), (sx+sw,sy+sh), (0,255,0),2)

        cv2.imshow('video',img)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
    
    
cam.release()
cv2.destroyAllWindows()
