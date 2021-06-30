import cv2
import numpy as np
import face_recognition



imgdj=face_recognition.load_image_file('Basicimages/Dj.jpg')
imgdj=cv2.cvtColor(imgdj,cv2.COLOR_BGR2RGB)
imgtest=face_recognition.load_image_file('Basicimages/download.jpg')
imgtest=cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)
faceLoc=face_recognition.face_locations(imgdj)[0]
encodedj=face_recognition.face_encodings(imgdj)[0]
cv2.rectangle(imgdj,(faceLoc[3],faceLoc[0],faceLoc[1],faceLoc[2]),(25,0,255),2)
print(faceLoc) #returs four values of face location so we wrote the upper line

faceLocTest=face_recognition.face_locations(imgtest)[0]
encodeimgtest=face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceLoc[3],faceLoc[0],faceLoc[1],faceLoc[2]),(25,0,255),2)

results=face_recognition.compare_faces([encodedj],encodeimgtest)
faceDis=face_recognition.face_distance([encodedj],encodeimgtest)
print(results,faceDis)
cv2.putText(imgtest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.imshow('DJ',imgdj)
cv2.imshow('dj',imgtest)
cv2.waitKey(0)