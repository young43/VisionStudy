
import cv2
import threading
import time
import numpy as np


from slidewindow import SlideWindow
from warper import Warper


def process_image(frame):
    # grayscle
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # blur
    # kernel_size = 5
    # blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # sharpening(외각을 더 또렷하게 해주는 효과)
    # -1이면 이미지크기와 동일
    kernel_sharpen = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0
    sharp_img = cv2.filter2D(gray, -1, kernel_sharpen)

    # canny edge
    # low 이하는 버림 / high 이상은 취함
    # 중간에 있는 값은 high와 연결되었을 때 취함
    # low_threshold = 60
    # high_threshold = 70
    # edges_img = cv2.Canny(np.uint8(blur_gray), low_threshold, high_threshold)

    # warper
    warp_img = warper.warp(sharp_img)

    # threshold 임계값으로 low값은 버리고, high값은 취함
    ret, thres_img = cv2.threshold(warp_img, 100, 255, cv2.THRESH_BINARY)

    # slide window
    img1, x_location = slidewindow.slidewindow(thres_img)

    return img1, x_location


warper = None
slidewindow  = SlideWindow()

def main():
    flag = False
    cap = cv2.VideoCapture("canny2.avi")

    while True:

        # 이미지를 캡쳐
        ret, img = cap.read()

        # 캡쳐되지 않은 경우 처리
        if not ret:
            break
        if cv2.waitKey(0) & 0xFF == 27:
            break

        # Warper 객체 생성 (초기 1번만)
        if not flag:
            flag = True
            global  warper
            warper = Warper(img)

        # warper, slidewindow 실행
        slideImage, x_location = process_image(img)

        # cv2.imshow("originImage", img)
        # cv2.imshow("warper", warper.warp(img))
        cv2.imshow("slidewindow", slideImage)

        #cv2.imshow("processImg", img1)



if __name__ == '__main__':
    main()