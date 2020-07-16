import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import *
from matplotlib.pyplot import *


class SlideWindow:
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.leftx = None
        self.rightx = None

    def slidewindow(self, img):

        x_location = None
        # init out_img, height, width
        # 3채널로 변경해줌
        out_img = np.dstack((img, img, img)) * 255
        height = img.shape[0]
        width = img.shape[1]

        # num of windows and init the height
        window_height = 5
        # 표시할 사각형 갯수
        nwindows = 30

        # find nonzero location in img, nonzerox, nonzeroy is the array flatted one dimension by x,y
        nonzero = img.nonzero()
        # print nonzero(행/열)
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # print nonzerox
        # init data need to sliding windows
        margin = 20
        minpix = 10
        left_lane_inds = []
        right_lane_inds = []

        # first location and segmenation location finder
        # draw line
        # 130 -> 150 -> 180
        pts_left = np.array([[width / 2 - 70, height], [width / 2 - 70, height - 60],
                             [width / 2 - 170, height - 80], [width / 2 - 170, height]], np.int32)
        # cv2.polylines(out_img, [pts_left], False, (0, 255, 0), 1)

        pts_right = np.array([[width / 2 + 57, height], [width / 2 + 57, height - 80],
                              [width / 2 + 120, height - 110], [width / 2 + 120, height]], np.int32)
        # cv2.polylines(out_img, [pts_right], False, (255, 0, 0), 1)

        # 중간선
        pts_catch = np.array([[0, 340], [width, 340]], np.int32)
        cv2.polylines(out_img, [pts_catch], False, (0, 120, 120), 1)

        # indicies before start line(the region of pts_left)
        # nonzero 영역에서 left,right 영역의 인덱스를 추출
        good_left_inds = ((nonzerox >= 35) & (nonzeroy >= nonzerox*0.4 + 320) & (nonzerox <= 200)).nonzero()[0]
        good_right_inds = ((nonzerox >= 490) & (nonzeroy >= nonzerox*(-0.48) + 520) & (nonzerox <= 620)).nonzero()[0]

        # left line exist, lefty current init
        # 현재 좌표(평균)
        y_current = 0
        x_current = 0

        # check the minpix before left start line
        # if minpix is enough on left, draw left, then draw right depends on left
        # else draw right, then draw left depends on right

        # 왼쪽을 우선적으로 보고, 차선이 없으면 오른쪽을 바라봄.
        if len(good_left_inds) > minpix:
            line_flag = 1

            x_current = np.int(np.mean(nonzerox[good_left_inds]))
            y_current = np.int(np.mean(nonzeroy[good_left_inds]))

        elif len(good_right_inds) > minpix:
            line_flag = 2

            x_current = np.int(np.mean(nonzerox[good_right_inds]))
            y_current = np.int(np.mean(nonzerox[good_right_inds]))
        else:
            # left/right 아무 선도 찾지 못했을 때
            line_flag = 3


        if line_flag != 3:
            # it's just for visualization of the valid inds in the region

            # 찾은 좌표들을 circle로 표시
            if line_flag == 1:
                for i in range(len(good_left_inds)):
                    cv2.circle(out_img, (nonzerox[good_left_inds[i]], nonzeroy[good_left_inds[i]]), 1, (0, 255, 0), -1)
            else:
                for i in range(len(good_right_inds)):
                    cv2.circle(out_img, (nonzerox[good_right_inds[i]], nonzeroy[good_right_inds[i]]), 1, (255, 0, 0), 1)


            # window sliding and draw
            for window in range(0, nwindows):

                # left 차선
                if line_flag == 1:

                    # rectangle x,y range init
                    # 사각형 30개를 표시하기 위함
                    win_y_low = y_current - (window + 1) * window_height
                    win_y_high = y_current  - window * window_height
                    win_x_low = x_current - margin
                    win_x_high = x_current + margin

                    # draw rectangle
                    cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 1)
                    cv2.rectangle(out_img, (670-margin, win_y_low), (670+margin, win_y_high), (255, 0, 0), 1)

                    # indicies of dots in nonzerox in one square
                    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                      (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

                    # check num of indicies in square and put next location to current
                    if len(good_left_inds) > minpix:
                        x_current = np.int(np.mean(nonzerox[good_left_inds]))
                    elif nonzeroy[left_lane_inds] != [] and nonzerox[left_lane_inds] != []:
                        # polyfit: y에대한 x 직선
                        p_left = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2)
                        x_current = np.int(np.polyval(p_left, win_y_high))

                    # 320~350 is for recognize line which is yellow line in processed image(you can check in imshow)
                    if win_y_low >= 320 and win_y_low < 350:
                        # 0.165 is the half of the road(0.33)
                        # 현재 car의 위치를 계산
                        x_location = x_current + int(width * 0.175)

                    left_lane_inds.extend(good_left_inds)

                else:  # change line from left to right above(if)

                    win_y_low = y_current - (window + 1) * window_height
                    win_y_high = y_current - window * window_height
                    win_x_low = x_current - margin
                    win_x_high = x_current + margin

                    cv2.rectangle(out_img, (120-margin, win_y_low), (120+margin, win_y_high), (0, 255, 0), 1)
                    cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (255, 0, 0), 1)

                    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                       (nonzerox >= win_y_low) & (nonzerox < win_y_high)).nonzero()[0]

                    if len(good_right_inds) > minpix:
                        x_current = np.int(np.mean(nonzerox[good_right_inds]))
                    elif nonzeroy[right_lane_inds] != [] and nonzerox[right_lane_inds] != []:
                        p_right = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2)
                        x_current = np.int(np.polyval(p_right, win_y_high))

                    if win_y_low >= 320 and win_y_low < 350:
                        # 0.165 is the half of the road(0.33)
                        x_location = x_current - int(width * 0.175)

                    right_lane_inds.extend(good_right_inds)

        return out_img, x_location