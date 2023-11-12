import cv2

# 創建一個 VideoCapture 對象，讀取影片
cap = cv2.VideoCapture("project_practice/project/video/fourway.mp4")

# 創建一個背景分割器
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# 創建一個空列表，用於記錄標記的座標
coordinates = []

while True:
    ret, frame = cap.read()#ret為布林值代表讀取與否 frame為影片偵，通常對frame處理
    if not ret:
        break
    # 使用背景分割器處理當前幀
    fg_mask = bg_subtractor.apply(frame)

    # 對前景掩碼進行後處理，去除小的雜訊
    fg_mask = cv2.erode(fg_mask, None, iterations=2)
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)

    # 尋找前景物體的輪廓
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 繪製檢測到的前景物體
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # 只繪製足夠大的前景區域
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 在表中記錄標記的座標
            coordinates.append((x + w // 2, y + h // 2))

    # 繪製連結各個時段標記的路徑
    for i in range(1, len(coordinates)):
        cv2.line(frame, coordinates[i - 1], coordinates[i], (0, 0, 255), 1)

    cv2.imshow("Original Video", frame)
    cv2.imshow("Foreground Mask", fg_mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
