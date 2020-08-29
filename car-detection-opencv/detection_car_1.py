# -*- coding: utf-8 -*-

import cv2

cascade_src = '/Users/shiv_vansh/Downloads/Vehicle-And-Pedestrian-Detection-Using-Haar-Cascades-master/Main Project/Main Project/Car Detection/cars.xml'

video_src = '/Users/shiv_vansh/Desktop/computer-vision/cardetectionyolo/video.mp4'

cap = cv2.VideoCapture(video_src)

car_cascade = cv2.CascadeClassifier(cascade_src)


while True:
    ret, frames = cap.read()
   
    if (type(frames) == type(None)):
        break
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)


    for (x,y,w,h) in cars:
        plate = frames[y:y + h, x:x + w]
        cv2.rectangle(frames,(x,y),(x +w, y +h) ,(51 ,51,255),2)
        cv2.rectangle(frames, (x, y - 40), (x + w, y), (51,51,255), -2)
        cv2.putText(frames, 'Car', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('car',plate)
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
    
    cv2.imshow('video', frames)
   
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
