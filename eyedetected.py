'''
@brief Closed eyes detection for fatigue driving demo
@note first we should to find face and eyes by Classifier
      then we cut out the pictures of the two eyes respectively
      then we find the eyeballs and get the outline of the eyeballs
      finally we fit with circle to judge whether the eyes are closed
@简介：疲劳驾驶闭眼检测小历程
@注释：通过面部识别和眼部模型识别找出面部和眼部
      再把两个眼睛分别截取出来（剔除眉毛干扰）
      做二值化处理，寻找轮廓，对眼珠进行圆形拟合
@ auther Raspberry & Mr.Pang. Beijing. 2021.12
@ warning: 本程序仅供交流学习使用，请勿商用，如有违反，概不负责
'''
import numpy as np
import cv2

def nothing(args):
    pass

# Classifier for face recognition and eye recognition
faceCascade = cv2.CascadeClassifier(r"Model\haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier(r'Model\haarcascade_eye.xml')

# turn on the camera, value = 0 means open the computer's default camera
cap = cv2.VideoCapture(0)
ok = True
flag = 0 # this value used to be a flags to show differences between you open your eyes or close your eyes
cv2.namedWindow('eye1')
cv2.namedWindow('eye2')
# use this to change the threshold
cv2.createTrackbar("th","eye1",20,255,nothing)
cv2.createTrackbar("th","eye2",20,255,nothing)

while ok:
    # 读取摄像头的图片
    # read the picture from the camera
    ok, img = cap.read()
    img2 = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Gray conversion
    # 通过人脸识别分类器找到人脸
    # Find face through face recognition classifier
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(32, 32))
    # 在找到人脸的基础上寻找眼球
    # Look for eyeballs on the basis of finding faces
    for (x, y, w, h) in faces:
        fac_gray = gray[y: (y+h), x: (x+w)]
        result = []
        eyes = eyeCascade.detectMultiScale(fac_gray, 1.3, 8,cv2.CASCADE_SCALE_IMAGE,(40,40),(80,80))
        # 眼睛坐标的换算，将相对位置换成绝对位置
        # Change relative position to absolute position
        for (ex, ey, ew, eh) in eyes:
            flag = 0
            print("未检测到闭眼", eyes[0])
            result.append((x+ex, y+ey, ew, eh))
    if flag == 1:
        print("检测到闭眼")
    flag = 1
    # 在检测到的人脸部分绘制矩阵
    # Draw a matrix in the detected face part
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (128, 255, 0), 2)
    try :
        for (ex, ey, ew, eh) in result:
            cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            # 通过位置数据来判断是左眼还是右眼
            # Determine whether it is the left eye or the right eye through the position data
            if result[0][0] > result[1][0]:
                img_eye_l = cv2.resize(img2[result[0][1]:result[0][1]+result[0][3], result[0][0]:result[0][0]+result[0][2]], (300, 300))
                img_eye_r = cv2.resize(img2[result[1][1]:result[1][1]+result[1][3], result[1][0]:result[1][0]+result[1][2]], (300, 300))
            if result[0][0] < result[1][0]:
                img_eye_l = cv2.resize(img2[result[1][1]:result[1][1]+result[1][3], result[1][0]:result[1][0]+result[1][2]], (300, 300))
                img_eye_r = cv2.resize(img2[result[0][1]:result[0][1]+result[0][3], result[0][0]:result[0][0]+result[0][2]], (300, 300))

            # 复制眼睛的图像
            # Copy images of eyes
            img_eye_l = img_eye_l[60:240,10:290]
            img_eye_r = img_eye_r[60:240,10:290]
            img_eye_l_gray = cv2.cvtColor(img_eye_l,cv2.COLOR_RGB2GRAY)
            img_eye_r_gray = cv2.cvtColor(img_eye_r,cv2.COLOR_RGB2GRAY)
            X1 = cv2.getTrackbarPos("th","eye1")
            X2 = cv2.getTrackbarPos("th","eye2")

            # 阈值处理
            # Threshold processing
            ret1,adaptive_l = cv2.threshold(img_eye_l_gray,X1,255,cv2.THRESH_BINARY_INV)
            ret2,adaptive_r = cv2.threshold(img_eye_r_gray,X2,255,cv2.THRESH_BINARY_INV)

            # 腐蚀与膨胀处理
            # Expansion and corrosion treatment
            mask_l = cv2.erode(adaptive_l, None, iterations=2)
            mask_r = cv2.erode(adaptive_r, None, iterations=2)
            mask_l = cv2.dilate(mask_l, None, iterations=2)
            mask_r = cv2.dilate(mask_r, None, iterations=2)

            # 边缘检测，寻找轮廓
            # edge detection
            cnts_l = cv2.findContours(mask_l.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            cnts_r = cv2.findContours(mask_r.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

            # 这里通过与圆形进行简单拟合来判断是否是眼珠
            # Fit with the circle to judge whether it is an an eyeball
            if len(cnts_l) >0:
                c = max(cnts_l, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if radius > 3:
                    cv2.circle(img_eye_l, (int(x), int(y)), int(radius), (255, 0, 0), 2)
                    cv2.circle(img_eye_l, center, 5, (0, 0, 255), -1)
                else:
                    print("Not Dect Eyes")

            if len(cnts_r) >0:
                c = max(cnts_r, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if radius > 3:
                    cv2.circle(img_eye_r, (int(x), int(y)), int(radius), (255, 0, 0), 2)
                    cv2.circle(img_eye_r, center, 5, (0, 0, 255), -1)
                else:
                    print("Not Dect Eyes")

            cv2.imshow("eye1",img_eye_l)
            cv2.imshow("eye2",img_eye_r)
            cv2.imshow("eye_l",adaptive_l)
            cv2.imshow("eye_r",adaptive_r)
    except:
        pass

    cv2.imshow('video', img)
    k = cv2.waitKey(1)

    # 按ESC退出程序
    # press 'ESC' to quit
    if k == 27:
        break
 
cap.release()
cv2.destroyAllWindows()