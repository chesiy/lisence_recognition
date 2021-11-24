import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from utils import plt_show

class Correction:
    def __init__(self,vertices, rect, img):
        self.vertices = vertices
        self.rect = rect
        self.img = img

    def get_ok_plate(self):
        point_set_0, point_set_1 = self.tileCorrection()
        res_img = self.transform(point_set_0, point_set_1)

        return res_img

    def transform(self, point_set_0, point_set_1):
        # 变换矩阵
        mat = cv2.getPerspectiveTransform(point_set_0, point_set_1)
        # 投影变换
        lic = cv2.warpPerspective(self.img, mat, (440, 140))

        return lic

    def tileCorrection(self):
        point_set_0, point_set_1, new_box = [],[],[]
        # 畸变情况1
        # print(self.rect[2])
        if self.rect[2] < 45:
            new_right_point_x = self.vertices[3, 0]
            new_right_point_y = int(
                self.vertices[2, 1] - (self.vertices[3, 0] - self.vertices[2, 0]) / (
                        self.vertices[1, 0] - self.vertices[2, 0]) * (
                        self.vertices[2, 1] - self.vertices[1, 1]))
            new_left_point_x = self.vertices[2, 0]
            new_left_point_y = int(
                self.vertices[3, 1] + (self.vertices[3, 0] - self.vertices[2, 0]) / (
                        self.vertices[3, 0] - self.vertices[0, 0]) * (
                        self.vertices[0, 1] - self.vertices[3, 1]))
            # 校正后的四个顶点坐标
            point_set_1 = np.float32([[440, 0], [0, 0], [0, 140], [440, 140]])
            # 校正前平行四边形四个顶点坐标
            new_box = np.array(
                [(self.vertices[3, 0], self.vertices[3, 1]), (new_left_point_x, new_left_point_y),
                 (self.vertices[2, 0], self.vertices[2, 1]),
                 (new_right_point_x, new_right_point_y)])
            point_set_0 = np.float32(new_box)

        # 畸变情况2
        elif self.rect[2] > 45:
            new_right_point_x = self.vertices[0, 0]
            new_right_point_y = int(
                self.vertices[1, 1] - (self.vertices[0, 0] - self.vertices[1, 0]) / (self.vertices[3, 0] - self.vertices[1, 0]) * (
                            self.vertices[1, 1] - self.vertices[3, 1]))
            new_left_point_x = self.vertices[1, 0]
            new_left_point_y = int(
                self.vertices[0, 1] + (self.vertices[0, 0] - self.vertices[1, 0]) / (self.vertices[0, 0] - self.vertices[2, 0]) * (
                            self.vertices[2, 1] - self.vertices[0, 1]))
            # 校正后的四个顶点坐标
            point_set_1 = np.float32([[440, 0], [0, 0], [0, 140], [440, 140]])
            # 校正前平行四边形四个顶点坐标
            new_box = np.array(
                [(self.vertices[0, 0], self.vertices[0, 1]), (new_left_point_x, new_left_point_y), (self.vertices[1, 0], self.vertices[1, 1]),
                 (new_right_point_x, new_right_point_y)])
            point_set_0 = np.float32(new_box)

        return point_set_0, point_set_1


class HoughCorrection:
    def __init__(self, img):
        self.img = img

    def Hough_correct(self):
        degree = self.CalcDegree()
        res_img = self.rotateImage(degree)
        return res_img

    # 通过霍夫变换计算角度
    def CalcDegree(self):
        # midImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
        dstImage = cv2.Canny(self.img, 50, 200, 3)
        lineimage = self.img.copy()

        # 通过霍夫变换检测直线
        # 第4个参数就是阈值，阈值越大，检测精度越高
        lines = cv2.HoughLines(dstImage, 1, np.pi / 180, 30)
        sum = 0
        # 依次画出每条线段
        for i in range(len(lines)):
            for rho, theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(round(x0 + 1000 * (-b)))
                y1 = int(round(y0 + 1000 * a))
                x2 = int(round(x0 - 1000 * (-b)))
                y2 = int(round(y0 - 1000 * a))
                # 只选角度最小的作为旋转角度
                sum += theta
                cv2.line(lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)

        # 对所有角度求平均，这样做旋转效果会更好
        average = sum / len(lines)
        angle = 90 - average/np.pi * 180
        # print('angle', angle)
        return angle

    # 旋转图像degree角度
    def rotateImage(self, degree):
        # 旋转中心为图像中心
        h, w = self.img.shape[:2]

        point_set_1 = np.float32([[0, h], [w, h], [w, 0]])

        point_set_0 = np.float32([[0, h], [w - h * math.tan(math.pi * degree / 180), h], [w, 0]])

        M = cv2.getAffineTransform(point_set_0, point_set_1)
        # 仿射变换，背景色填充为白色
        rotate = cv2.warpAffine(self.img, M, (w, h), borderValue=(0, 0, 0))

        return rotate


class CutBorder:
    def __init__(self,img):
        self.img = img
    def find_border(self):
        x_histogram = np.sum(self.img, axis=1)
        x_min = np.min(x_histogram)
        x_average = np.sum(x_histogram) / x_histogram.shape[0]
        x_threshold = (x_min + x_average) / 2

        wave_peaks = self.find_waves(x_threshold, x_histogram)
        # print(wave_peaks)
        wave = []
        if len(wave_peaks) == 0:
            print("peak less 0:")
        else:
            # print(wave_peaks)
            # 认为水平方向，最大的波峰为车牌区域
            wave = max(wave_peaks, key=lambda x: x[1] - x[0])
            # print(wave)

        return wave

    def find_waves(self,threshold, histogram):
        up_point = -1    # 上升点
        is_peak = False
        if histogram[0] > threshold:
            up_point = 0
            is_peak = True
        wave_peaks = []
        for i, x in enumerate(histogram):
            if is_peak and x < threshold:
                if i - up_point > 2:
                    is_peak = False
                    wave_peaks.append((up_point, i))
            elif not is_peak and x >= threshold:
                is_peak = True
                up_point = i
        if is_peak and up_point != -1 and i - up_point > 4:
            wave_peaks.append((up_point, i))
        return wave_peaks