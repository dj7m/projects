import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path='ImageAttendance'
images=[]
classNames=[]
mylist=os.listdir(path)
print(mylist)
for cls in mylist:
    curImg=cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtstring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')

markAttendance('a')


encodeListknown=findEncodings(images)
print('Encoding Complete')

cap=cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgSmall=cv2.resize(img,(0,0),None,0.25,0.25)
    imgSmall=cv2.cvtColor(imgSmall,cv2.COLOR_BGR2RGB)

    facesCurrFrame=face_recognition.face_locations(imgSmall)
    encodesCurrFrame=face_recognition.face_encodings(imgSmall,facesCurrFrame)

    for encodeFace,faceLoc in zip(encodesCurrFrame,facesCurrFrame):
        matches=face_recognition.compare_faces(encodeListknown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListknown,encodeFace)
        print(faceDis)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-25),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+3,y2-3),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            #markAttendance(name)


    cv2.imshow('webcam',img)
    cv2.waitKey(1)

#imgdj=face_recognition.load_image_file('Basicimages/Dj.jpg')
#imgdj=cv2.cvtColor(imgdj,cv2.COLOR_BGR2RGB)
#imgtest=face_recognition.load_image_file('Basicimages/download.jpg')
#imgtest=cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

#faceLoc=face_recognition.face_locations(imgdj)[0]
#encodedj=face_recognition.face_encodings(imgdj)[0]
#cv2.rectangle(imgdj,(faceLoc[3],faceLoc[0],faceLoc[1],faceLoc[2]),(25,0,255),2)
#print(faceLoc) #returs four values of face location so we wrote the upper line

#faceLocTest=face_recognition.face_locations(imgtest)[0]
#encodeimgtest=face_recognition.face_encodings(imgtest)[0]
#cv2.rectangle(imgtest,(faceLoc[3],faceLoc[0],faceLoc[1],faceLoc[2]),(25,0,255),2)

#results=face_recognition.compare_faces([encodedj],encodeimgtest)
#faceDis=face_recognition.face_distance([encodedj],encodeimgtest)