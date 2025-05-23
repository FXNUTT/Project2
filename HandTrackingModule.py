import cv2                               # น ำเข้ำไลบรำรี OpenCV ส ำหรับประมวลผลภำพ 
import mediapipe as mp                   # น ำเข้ำ MediaPipe ส ำหรับกำรตรวจจับท่ำทำงและ landmark ของมือ 
import time                              # น ำเข้ำไลบรำรี time ส ำหรับจัดกำรเวลำ 
import math                              # น ำเข้ำไลบรำรี math ส ำหรับค ำนวณทำงคณิตศำสตร์ เช่น ระยะห่ำง 
import numpy as np                       # น ำเข้ำ NumPy ส ำหรับกำรจัดกำรข้อมูลในรูปแบบอำเรย์ 
 
# นิยำมคลำส handDetector ส ำหรับตรวจจับมือและวิเครำะห์ต ำแหน่ง landmark 
class handDetector(): 
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5): 
        self.mode = mode                           # ก ำหนดโหมดกำรท ำงำน (static/dynamic) 
        self.maxHands = maxHands                   # ก ำหนดจ ำนวนมือสูงสุดที่ต้องกำรตรวจจับ 
        self.detectionCon = detectionCon           # ก ำหนดควำมน่ำเชื่อถือขั้นต ่ำส ำหรับกำรตรวจจับมือ 
        self.trackCon = trackCon                   # ก ำหนดควำมน่ำเชื่อถือขั้นต ่ำส ำหรับกำรติดตำม landmark 
 
        self.mpHands = mp.solutions.hands          # เข้ำถึงโมดูล hands จำก MediaPipe 
        self.hands = self.mpHands.Hands(    # สร้ำงอ็อบเจ็กต์ Hands ของ MediaPipe ด้วยพำรำมิเตอร์ที่ก ำหนด 
            static_image_mode=self.mode, 
            max_num_hands=self.maxHands, 
            min_detection_confidence=self.detectionCon, 
            min_tracking_confidence=self.trackCon 
        ) 
        self.mpDraw = mp.solutions.drawing_utils   # เข้ำถึงโมดูลวำดรูปของ MediaPipe ส ำหรับวำด 
landmark 
        self.tipIds = [4, 8, 12, 16, 20]           # ก ำหนด index ของปลำยนิ ้วของแต่ละนิ ้วใน 
landmark 
 
    # ฟังก์ชันส ำหรับค้นหำและวำด landmark ของมือในภำพ 
    def findHands(self, img, draw=True): 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # แปลงภำพจำก BGR (OpenCV) เป็น 
RGB (MediaPipe) 
        self.results = self.hands.process(imgRGB)          # ประมวลผลภำพเพื่อค้นหำมือและ 
landmark 
        if self.results.multi_hand_landmarks:              # หำกตรวจพบ landmark ของมือในภำพ 
            for handLms in self.results.multi_hand_landmarks:  # วนลูปผ่ำนแต่ละมือที่ตรวจจับได้ 
                if draw:                                   # หำกต้องกำรวำด landmark บนภำพ 
                    self.mpDraw.draw_landmarks(img, handLms,  # วำด landmark และกำรเชื่อมต่อ
 ของมือบนภำพ 
                                               self.mpHands.HAND_CONNECTIONS) 
        return img                                        # คืนค่ำภำพที่มีกำรวำด landmark (ถ้ำมี) 
 
    # ฟังก์ชันส ำหรับค้นหำต ำแหน่ง landmark และกล่องล้อมรอบมือในภำพ 
    def findPosition(self, img, handNo=0, draw=True): 
        xList = []                                           # สร้ำงรำยกำรเพื่อเก็บค่ำพิกัด x ของแต่ละ 
landmark 
        yList = []                                           # สร้ำงรำยกำรเพื่อเก็บค่ำพิกัด y ของแต่ละ 
landmark 
        bbox = []                                            # ตัวแปรส ำหรับเก็บขอบเขต 
(bounding box) ของมือ 
        self.lmList = []                                # รีเซ็ตรำยกำร landmark ของมือให้เป็นค่ำว่ำง 
        if self.results.multi_hand_landmarks:           # หำกมีกำรตรวจจับ landmark ของมือ 
            myHand = self.results.multi_hand_landmarks[handNo]  
           # เลือกมือที่ต้องกำร (ค่ำเริ่มต้นคือมือแรก) 
            for id, lm in enumerate(myHand.landmark):   # วนลูปผ่ำนแต่ละ landmark พร้อมทั้งรับ 
index 
                h, w, c = img.shape                     # รับขนำดของภำพ (สูง, กว้ำง, ช่องสี) 
                cx, cy = int(lm.x * w), int(lm.y * h)    # ค ำนวณพิกัดกลำงของ landmark ในภำพจริง 
                xList.append(cx)                         # เพิ่มพิกัด x ลงในรำยกำร 
                yList.append(cy)                         # เพิ่มพิกัด y ลงในรำยกำร 
                self.lmList.append([id, cx, cy])  # เพิ่มข้อมูล landmark (id, x, y) ลงในรำยกำร 
                if draw:                                 # หำกต้องกำรวำด landmark 
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)  # วำด
 วงกลมที่ต ำแหน่ง landmark 
 
            xmin, xmax = min(xList), max(xList)              # ค ำนวณต ำแหน่ง x ต ่ำสุดและสูงสุด
 ส ำหรับ bounding box 
            ymin, ymax = min(yList), max(yList)              # ค ำนวณต ำแหน่ง y ต ่ำสุดและสูงสุด
 ส ำหรับ bounding box 
            bbox = xmin, ymin, xmax, ymax                    # ก ำหนด bounding box 
 
            if draw:                                         # หำกต้องกำรวำด bounding box 
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), 
                              (0, 255, 0), 2)               # วำดสี่เหลี่ยมรอบมือ โดยขยำยขอบเล็กน้อย 
 
        return self.lmList, bbox                 # คืนค่ำ landmark list และ bounding box 
 
    # ฟังก์ชันส ำหรับตรวจสอบว่ำนิ ้วไหนยกขึ ้น (finger up) 
    def fingersUp(self): 
        fingers = []                             # สร้ำงรำยกำรเพื่อเก็บสถำนะของนิ ้ว (1 = ยก, 0 = ไม่ยก) 
        if len(self.lmList) > 0:                 # ตรวจสอบว่ำมี landmark อยู ่ในรำยกำรหรือไม่ 
            # ตรวจสอบนิ ้วหัวแม่มือ 
            if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]: 
                fingers.append(1)                        # หำกนิ ้วหัวแม่มือยกขึ ้น ให้เพิ่ม 1 ลงในรำยกำร 
            else: 
                fingers.append(0)                        # หำกไม่ใช่ให้เพิ่ม 0 ลงในรำยกำร 
 
            # ตรวจสอบนิ ้วที่เหลือ (นิ ้วชี ้, นิ ้วกลำง, นิ ้วนำง, นิ ้วก้อย) 
            for id in range(1, 5): 
                if len(self.lmList) > self.tipIds[id] and len(self.lmList) > 
self.tipIds[id] - 2:  # ตรวจสอบ index ว่ำมีอยู ่ในขอบเขตหรือไม่ 
                    if self.lmList[self.tipIds[id]][2] < 
self.lmList[self.tipIds[id] - 2][2]: 
                        fingers.append(1)                # หำกนิ ้วยกขึ ้น (ต ำแหน่ง y ของปลำยนิ ้วสูงกว่ำ 
landmark ที่สองจำกปลำยนิ ้ว) ให้เพิ่ม 1 
                    else: 
                        fingers.append(0)                # หำกไม่ใช่ให้เพิ่ม 0 
 
        # ตรวจสอบให้แน่ใจว่ำรำยกำร fingers มีสมำชิกครบ 5 ตัว 
        if len(fingers) < 5: 
            fingers = fingers + [0] * (5 - len(fingers))    # เติม 0 ให้ครบ 5 ค่ำในกรณีที่ขำด 
 
        return fingers                                      # คืนค่ำรำยกำรสถำนะของนิ ้ว 
 
    # ฟังก์ชันส ำหรับค ำนวณระยะห่ำงระหว่ำงสอง landmark 
    def findDistance(self, p1, p2, img, draw=True, r=15, t=3): 
        x1, y1 = self.lmList[p1][1:]                  # ดึงพิกัด (x, y) ของ landmark ที่ p1 
        x2, y2 = self.lmList[p2][1:]                  # ดึงพิกัด (x, y) ของ landmark ที่ p2 
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2       # ค ำนวณพิกัดจุดกึ่งกลำงระหว่ำง landmark ทั้งสอง 
 
        if draw:                                           # หำกต้องกำรวำดเส้นและวงกลม 
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)  # วำดเส้นเชื่อมระหว่ำง 
landmark ที่ p1 และ p2 
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)  # วำดวงกลมที่ 
landmark ที่ p1 
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)  # วำดวงกลมที่ 
landmark ที่ p2 
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)      # วำดวงกลมที่จุดกึ่งกลำง 
 
        length = math.hypot(x2 - x1, y2 - y1)       # ค ำนวณระยะห่ำงระหว่ำง landmark ด้วยสูตรพีทำโกรัส 
 
        return length, img, [x1, y1, x2, y2, cx, cy]       # คืนค่ำระยะห่ำง, ภำพที่วำด, และข้อมูลพิกัด 
 
# ฟังก์ชัน main ส ำหรับทดสอบกำรท ำงำนของ handDetector 
def main(): 
    pTime = 0                                           # ก ำหนดเวลำเริ่มต้นส ำหรับกำรค ำนวณ FPS 
    cTime = 0                                           # ตัวแปรเก็บเวลำในขณะนั้น 
    cap = cv2.VideoCapture(0)                           # เปิดกล้องเว็บแคม (ใช้ 0 ส ำหรับกล้องหลัก) 
 
    if not cap.isOpened():                              # ตรวจสอบว่ำกล้องเปิดได้หรือไม่ 
        print("Error: Could not open video stream.")   # แสดงข้อควำมผิดพลำดหำกไม่สำมำรถเปิดกล้องได้ 
        return                                          # ออกจำกฟังก์ชัน main 
 
    detector = handDetector()                       # สร้ำงอ็อบเจ็กต์ handDetector ส ำหรับตรวจจับมือ 
 
    while True:                                         # เริ่มลูปหลักเพื่อประมวลผลเฟรมจำกกล้อง 
        success, img = cap.read()                       # อ่ำนเฟรมจำกกล้อง 
        if not success:                                 # หำกไม่สำมำรถอ่ำนเฟรมได้ 
            print("Failed to grab frame")               # แสดงข้อควำมผิดพลำด 
            break                                       # ออกจำกลูป 
 
        img = detector.findHands(img)   # ใช้ handDetector เพื่อค้นหำและวำด landmark ของมือบนภำพ 
        lmList, bbox = detector.findPosition(img)       # ค้นหำต ำแหน่ง landmark และ 
bounding box ของมือ 
        if len(lmList) != 0:                            # หำกมี landmark ถูกตรวจจับ 
            print(lmList[4])            # แสดงผลต ำแหน่งของ landmark ที่มี index 4 (เช่น จุดที่น่ำสนใจ) 
 
        cTime = time.time()                             # บันทึกเวลำปัจจุบัน 
        fps = 1 / (cTime - pTime)                       # ค ำนวณ FPS จำกควำมแตกต่ำงระหว่ำงเวลำปัจจุบัน
 และเวลำก่อนหน้ำ 
        pTime = cTime                                   # อัปเดตเวลำเฟรมก่อนหน้ำ 
 
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, 
                    (255, 0, 255), 3)                    # วำดค่ำ FPS ลงบนภำพที่แสดง 
 
        cv2.imshow("Image", img)                        # แสดงภำพในหน้ำต่ำงชื่อ "Image" 
        if cv2.waitKey(1) & 0xFF == ord('q'):           # รอรับกำรกดปุ ่ม 'q' เพื่อออกจำกลูป 
            break                                       # ออกจำกลูปหลัก 
 
    cap.release()                                       # ปล่อยกำรเข้ำถึงกล้องหลังจำกจบกำรใช้งำน 
    cv2.destroyAllWindows()                             # ปิดหน้ำต่ำงทั้งหมดที่เปิดโดย OpenCV 
 
# ตรวจสอบว่ำรันสคริปต์นี ้เป็นโปรแกรมหลักหรือไม่ 
if __name__ == "__main__": 
    main()                                              # เรียกใช้ฟังก์ชัน main เพื่อเริ่มโปรแกรม
