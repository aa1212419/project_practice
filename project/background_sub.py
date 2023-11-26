import cv2
import numpy as np

# 創建一個 VideoCapture 對象，讀取影片
cap = cv2.VideoCapture("project/video/11:27.mp4")

# 創建一個背景分割器
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

# 創建一個空列表，用於記錄標記的座標
max_length = 2
coordinates = []

pause = False

while True:
    if not pause:
        ret, frame = cap.read()#ret為布林值代表讀取與否 frame為影片偵，通常對frame處理
        if not ret:
            break
        fg_mask = bg_subtractor.apply(frame)
        _,thresh = cv2.threshold(fg_mask, 127, 255, 0)
        # 對前景掩碼進行後處理，去除小的雜訊
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # 尋找前景物體的輪廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 繪製檢測到的前景物體
        for contour in contours:
            area = cv2.contourArea(contour)
             # 只繪製足夠大的前景區域
            x, y, w, h = cv2.boundingRect(contour)
            if(x+w * 1.5 > y+h) and area > 1000:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 在表中記錄標記的座標
                coordinates.append((x + w // 2, y + h // 2))
                
            if len(coordinates) > max_length:
                coordinates.pop(0)  # 移除最舊的座標

        # 繪製連結各個時段標記的路徑
        for i in range(1, len(coordinates)):
            cv2.line(frame, coordinates[i - 1], coordinates[i], (0, 0, 255), 1)

        
        cv2.imshow("Foreground Mask", thresh)
        cv2.imshow("Original Video", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key  == ord('q'):
            break
        elif key  == ord('p'):  # 按 'p' 暫停或恢復
            while True:
                key = cv2.waitKey(1) 
                if key == ord('p') or key == ord('q'):
                    break
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()