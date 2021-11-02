import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm
from random import choice
from keras.models import load_model
from threading import Thread

def resize(img):
    scale_percent = 20
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return np.expand_dims(resized, axis=0)

##############

brushThickness=15
eraserThickness=50
cTime=0
pTime=0
###############


model = load_model("model.h5")

score = 0
alive = False
latestScore=0
folderPath ="Header"
folderSample="Sample"

myList = os.listdir(folderPath)
mysample = sorted(os.listdir(folderSample))

overlayList = []
samplelist = []

IdxSam = 0
latestIdx = -1
tablescore=[]

drawColor=(255, 0 ,255)

for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}")
    overlayList.append(image)
for imPath in mysample:
    image = cv2.imread(f"{folderSample}/{imPath}")
    samplelist.append(image)
#print(len(overlayList))

header = overlayList[0]
imgsample = samplelist[IdxSam]

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector= htm.handDetector(detectionCon=0.9)
xp, yp = 0 ,0

imgCanvas = np.zeros((720, 1280, 3),np.uint8)


counterenalbe = False
counterTimer = 0

def classify():
    global imgcrop
    global imgCanvas
    global score
    global counterenalbe
    global counterTimer
    global alive
    global latestScore
    global tablescore

    imgcrop = imgCanvas[287:587, 481:781]
    x = resize(cv2.cvtColor(imgcrop, cv2.COLOR_BGR2GRAY))
    print(x.shape)
    prob = model.predict(x)[0]
    print(prob, IdxSam)
    #ans = np.argmax(prob)
    
    if prob[IdxSam] - .5 >= 1e-9:
        latestScore= round(100 * prob[IdxSam])
        score += latestScore

    else:
        latestScore= -100+round(100 * prob[IdxSam])
        score +=latestScore 
    tablescore.append(latestScore)
    print(score)
    cv2.imwrite("Tamgiac.jpg", imgcrop)

    imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    counterenalbe = False
    counterTimer = 0

    alive = False

while True and len(tablescore) <= 30: 
    #1.Import Image
    sucess, img = cap.read()
    img = cv2.flip(img, 1)



    #2. Find Hand Landmarks

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if lmList is not None and len(lmList)!=0 :


        #print(lmList)

        # tip of index and middle finger
        x1, y1= lmList[8][1:]
        x2, y2 = lmList[12][1:]

        #3. Check which finger are up

        fingers = detector.fingersUp()
        #print(fingers)
        #4. If Selection Mode- Two finger are up
        if fingers[1] and fingers[2]:
            xp, yp = 0 ,0
            #print("Selection Mode")
            #Checking for the click
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255,0,255)
                if 550 < x1 <750:
                    header = overlayList[1]
                    drawColor = (255,0,0)
                if 800 < x1 <950:
                    header = overlayList[2]
                    drawColor = (0,255,0)
                if 1050 < x1 <1200:
                    header = overlayList[3]
                    drawColor = (0,0,0)
            cv2.rectangle(img, (x1, y1 - 20), (x2, y2 + 20), drawColor, cv2.FILLED)

            
        #5. If Drawing Mode- Index finger is up
        if fingers[1] and fingers[2] ==False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            #print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1


            if drawColor ==(0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp),(x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp),(x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp),(x1, y1), drawColor, brushThickness)

            
            xp, yp = x1, y1


    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV) 
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img= cv2.bitwise_and(img, imgInv)
    img= cv2.bitwise_or(img, imgCanvas)


    #setting the header image
    #khung hình
    imgkhung = overlayList[4]
    imgGray = cv2.cvtColor(imgkhung, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV) 
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgkhung)
    img[0:125, 0:1280] = header
    

    

    #Xuat hinh mau
    
    
    
    if not counterenalbe:
        #IdxSam=2
        while latestIdx == IdxSam:
            IdxSam = choice(range(len(samplelist)))
        latestIdx = IdxSam
        imgsample = samplelist[IdxSam]
        counterenalbe = True
    if counterTimer > 140 and not alive:
        alive = True
        Thread(target=classify).start()

    counterTimer += 1
    #if phanloai:
        #counterenalbe=True 
    

        #counterTimer=0
    #Random sample
    imgGray = cv2.cvtColor(imgsample, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV) 
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgsample)

    imgcrop = imgCanvas[238:588, 422:772]
    #fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.putText(img, str(int(score)), (10, 200), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    cv2.putText(img, str(int(latestScore)), (10, 300), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)



    #img = cv2.addWeighted(img, 0.5, imgCanvas,0.5,0)
    # cv2.imshow("Crop", imgcrop)
    cv2.imshow("Image", img)
    # cv2.imhow("Canvas", imgCanvas)
    #cv2.imwrite("giaodien.jpg",img)
    cv2.waitKey(1)
#print(tablescore)
# def classify():