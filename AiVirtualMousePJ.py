import cv2                              # ไลบรารีประมวลผลภาพ
import numpy as np                      # ไลบรารีคำนวณตัวเลข
import HandTrackingModule as htm        # โมดูลตรวจจับมือที่เขียนเอง
import time                             # สำหรับจับเวลาและคำนวณ FPS
import autopy                           # สำหรับควบคุมเมาส์
from ultralytics import YOLO            # ใช้ YOLO จาก ultralytics ตรวจจับวัตถุ

# =================== การตั้งค่าพื้นฐาน ====================
wCam, hCam = 1920, 1080                 # ขนาดวิดีโอกล้อง
frameR = 100                            # ระยะขอบสำหรับควบคุมเมาส์
smoothening = 7                         # ค่าความลื่นไหลของเมาส์

pTime = 0                               # เวลาเฟรมก่อนหน้า
plocX, plocY = 0, 0                     # ตำแหน่งเมาส์ก่อนหน้า
clocX, clocY = 0, 0                     # ตำแหน่งเมาส์ปัจจุบัน

# =================== เริ่มต้นกล้อง ========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video device")
    exit()

cap.set(3, wCam)                        # ตั้งค่าความกว้างเฟรม
cap.set(4, hCam)                        # ตั้งค่าความสูงเฟรม

# =================== โหลดโมเดลและตัวช่วย =================
detector = htm.handDetector(maxHands=1)         # ตรวจจับมือข้างเดียว
wScr, hScr = autopy.screen.size()               # ขนาดหน้าจอคอมพิวเตอร์
yolo_model = YOLO("yolov8n.pt")                 # โหลดโมเดล YOLO
mode = "mouse_control"                          # โหมดเริ่มต้น

# =================== วนลูปหลักของโปรแกรม =================
while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image")
        break

    # ตรวจจับมือและวาดเส้นโครงร่าง
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # ============ โหมดควบคุมเมาส์ด้วยมือ ============
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # ตำแหน่งปลายนิ้วชี้
        fingers = detector.fingersUp()

        if len(fingers) >= 5:
            if fingers[1] == 1 and fingers[2] == 0:  # ชี้นิ้วเดียว
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                autopy.mouse.move(wScr - clocX, clocY)
                plocX, plocY = clocX, clocY

            if len(lmList) >= 9:
                length, img, lineInfo = detector.findDistance(4, 8, img)
                if length < 40:
                    autopy.mouse.click()

    # ============ โหมดตรวจจับวัตถุด้วย YOLO ============
    if mode == "object_detection":
        results = yolo_model(img)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls[0])]
                confidence = box.conf[0]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3, cv2.LINE_AA)

    # ============ แสดงผล FPS และคำสั่ง =============
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"Mode: {mode}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "Press 'M' to switch to Mouse Control", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, "Press 'O' to switch to Object Detection", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, "Press 'Q' to quit", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, f"FPS: {int(fps)}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", img)

    # ================= ควบคุมด้วยคีย์บอร์ด ================
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        mode = "mouse_control"
    elif key == ord('o'):
        mode = "object_detection"

# =================== ปิดการใช้งาน ====================
cap.release()
cv2.destroyAllWindows()
