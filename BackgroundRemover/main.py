import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
segmentor=SelfiSegmentation()
#fpsReader=cvzone.FPS()
imgBg=cv2.imread("images/5.jpg")

#listImg=os.listdir("images")
#print(listImg)
#imgList=[]
#for imgpath in listImg:
#    img=cv2.imread(f'images/{imgpath}')
#    imgList.append(img)
#print(len(imgList))

 indexImg=0

while True:
    success, img = cap.read()
    imgOut=segmentor.removeBG(img,imgBg,threshold=0.8)

    cv2.imshow("Image",img)
    cv2.imshow("Image Out",imgOut)
    cv2.waitKeyEx(1)

