import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap=cv2.VideoCapture(0)
cap.set(3,640) #width
cap.set(4,480)
cap.set(cv2.CAP_PROP_FPS,60) #increasing framerate to 60
segmentor=SelfiSegmentation()
fpsReader=cvzone.FPS()  #adding framerate
imgBg=cv2.imread("images/5.jpg")  #reading images for background input

#setting images to auto fill background
listImg=os.listdir("images")
print(listImg)
imgList=[]
for imgpath in listImg:
    img=cv2.imread(f'images/{imgpath}')
    imgList.append(img)
print(len(imgList))

indexImg=0

while True:
    success, img = cap.read()
    imgOut=segmentor.removeBG(img,imgList[indexImg],threshold=0.8)   #required out put with back ground removed and an image is put in bg

    imgStacked=cvzone.stackImages([img,imgOut],2,1)
    _, imgStacked=fpsReader.update(imgStacked,color=(0,0,255)) #putting framerate over image


    print(indexImg)
    #cv2.imshow("Image",img)
    cv2.imshow("Image Out",imgStacked)
    key=cv2.waitKeyEx(1)
    if key==ord('a'):
        if indexImg>5:
           indexImg -=1
    elif key==ord('d'):
        if indexImg <len(imgList)-1:
           indexImg +=1
    elif key==ord('q'):
        break

