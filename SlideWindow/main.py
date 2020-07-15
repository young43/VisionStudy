
import cv2
import threading
import time
import numpy as np


from slidewindow import SlideWindow
from warper import Warper

warper = Warper()
slidewindow  = SlideWindow()


def process_image(frame):
    # grayscle
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    # canny edge
    low_threshold = 60  # 60
    high_threshold = 70  # 70
    edges_img = cv2.Canny(np.uint8(blur_gray), low_threshold, high_threshold)
    # warper
    img = warper.warp(edges_img)

    #img1, x_location = slidewindow.slidewindow(img)
    #return img1, x_location
    return img

def main():
    cap = cv2.VideoCapture("canny1.avi")

    while (True):

        # 이미지를 캡쳐
        ret, img = cap.read()

        # 캡쳐되지 않은 경우 처리
        if not ret:
            break

        cv2.imshow("Image", img)
        print(ret)
        #img1, x_location = process_image(img)
        #cv2.imshow("processImg", img1)

    # img = cv2.imread("image.jpg", cv2.IMREAD_COLOR)
    # img1, x_location = process_image(img)

if __name__ == '__main__':
    main()