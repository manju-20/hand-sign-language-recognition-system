import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)
classifier = Classifier("C:/Users/manju/PycharmProjects/pythonProject2/venv/Model/keras_model.h5","C:/Users/manju/PycharmProjects/pythonProject2/venv/Model/labels.txt")
offset = 20
imgSize = 300

folder = "C:/Users/manju/PycharmProjects/pythonProject2/venv/Data/2"
counter = 0
lables =["1","2","A","B","C","D","thumbhup","wave"]
while True:
  success, img = cap.read()
  imgOutput = img.copy()
  hands, img = detector.findHands(img)
  if hands:
      hand = hands[0]
      x, y, w, h = hand['bbox']
      imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255
      imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

      imgCropShape = imgCrop.shape



      aspectRatio = h / w

      if aspectRatio >1:
          k = imgSize/h
          wCal = math.ceil(k*w)
          imgResize = cv2.resize(imgCrop, (wCal, imgSize))
          imgResizeShape = imgResize.shape
          wGap = math.ceil((imgSize-wCal)/2)
          imgWhite[:, wGap:wCal+wGap] = imgResize
          prediction, index = classifier.getPrediction(img)
          print(prediction,index)
      else:
          k = imgSize / w
          hCal = math.ceil(k * h)
          imgResize = cv2.resize(imgCrop, (imgSize, hCal))
          imgResizeShape = imgResize.shape
          hGap = math.ceil((imgSize - hCal) / 2)
          imgWhite[hGap:hCal + hGap, :] = imgResize


      cv2.imshow("ImageCrop", imgCrop)
      cv2.imshow("ImageWhite", imgWhite)

  cv2.imshow("Image", imgOutput)
  cv2.waitKey(1)

