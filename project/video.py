import cv2
cap = cv2.VideoCapture('video/dog.mp4')
while True:
    ret , next = cap.read()
    if ret:
        cv2.imshow('video',next)
    else:
        break
    cv2.waitKey(1)