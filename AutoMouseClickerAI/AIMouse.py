import cv2
import time
import HandTrackingModule as htm
import autopy
import numpy as np

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
detector = htm.handDetector(maxHands=1)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    cTime = time.time()
    fps = 1/(cTime-pTime) 
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)

    cv2.imshow("image", img)
    cv2.waitKey(1)

