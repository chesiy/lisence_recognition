import cv2
from utils import plt_show,plt_show0,gray_guss
import numpy as np

class LocatePlate:
    def __init__(self, img, is_green):
        self.img = img
        self.is_green = is_green

    def locate_by_color(self):
        self.imgProcess()
        img_bin = self.preIdentification()
        points = self.fixPosition(img_bin)
        vertices,rect = self.findVertices(points)

        return vertices, rect

    def locate_by_edge(self):
        gray_image = gray_guss(self.img)
        # x方向上的边缘检测（增强边缘信息）
        Sobel_x = cv2.Sobel(gray_image, cv2.CV_16S, 2, 0)
        absX = cv2.convertScaleAbs(Sobel_x)
        image = absX

        # 图像阈值化操作——获得二值化图
        ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
        # 显示灰度图像
        # plt_show(image)
        # 形态学（从图像中提取对表达和描绘区域形状有意义的图像分量）——闭操作
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 50))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX, iterations=1)

        # 腐蚀（erode）和膨胀（dilate）
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        # x方向进行闭操作（抑制暗细节）
        image = cv2.dilate(image, kernelX)
        image = cv2.erode(image, kernelX)
        # y方向的开操作
        image = cv2.erode(image, kernelY)
        image = cv2.dilate(image, kernelY)
        # 中值滤波（去噪）
        image = cv2.medianBlur(image, 21)

        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for item in contours:
            rect = cv2.boundingRect(item)
            x = rect[0]
            y = rect[1]
            width = rect[2]
            height = rect[3]
            # 根据轮廓的形状特点，确定车牌的轮廓位置并截取图像
            if (width > (height * 3)) and (width < (height * 4)) and height > 100:
                image = self.img[y:y + height, x:x + width]

        return image

    # 预处理
    def imgProcess(self):
        # 高斯模糊
        img_Gas = cv2.GaussianBlur(self.img, (5, 5), 0)
        # RGB通道分离
        self.img_B = cv2.split(img_Gas)[0]
        self.img_G = cv2.split(img_Gas)[1]
        self.img_R = cv2.split(img_Gas)[2]
        # 读取灰度图和HSV空间图
        self.img_gray = cv2.cvtColor(img_Gas, cv2.COLOR_BGR2GRAY)
        self.img_HSV = cv2.cvtColor(img_Gas, cv2.COLOR_BGR2HSV)

    # 初步识别
    def preIdentification(self):
        if self.is_green:
            for i in range(self.img.shape[0]):
                for j in range(self.img.shape[1]):
                    # 普通绿色车牌，同时排除透明反光物质的干扰
                    if ((self.img_HSV[:, :, 0][i, j] - 55) ** 2 < 20 ** 2) and (self.img_G[i, j] > 100) and (self.img_B[i, j] < 120) and (
                            self.img_R[i, j] < 100):
                        self.img_gray[i, j] = 255
                    else:
                        self.img_gray[i, j] = 0
        else:
            for i in range(self.img.shape[0]):
                for j in range(self.img.shape[1]):
                    # 普通蓝色车牌，同时排除透明反光物质的干扰
                    if ((self.img_HSV[:, :, 0][i, j] - 115) ** 2 < 15 ** 2) and (self.img_B[i, j] > 70) and (self.img_R[i, j] < 40):
                        self.img_gray[i, j] = 255
                    else:
                        self.img_gray[i, j] = 0

        # 定义核
        kernel_small = np.ones((3, 3))
        kernel_big = np.ones((7, 7))

        img_gray = cv2.GaussianBlur(self.img_gray, (3, 3), 0)  # 高斯平滑
        img_di = cv2.dilate(img_gray, kernel_small, iterations=5)  # 腐蚀5次
        img_close = cv2.morphologyEx(img_di, cv2.MORPH_CLOSE, kernel_big)  # 闭操作
        img_close = cv2.GaussianBlur(img_close, (3, 3), 0)  # 高斯平滑
        _, img_bin = cv2.threshold(img_close, 100, 255, cv2.THRESH_BINARY)  # 二值化

        # plt_show(img_bin)

        return img_bin

    # 定位
    def fixPosition(self, img_bin):
        # 检测所有外轮廓，只留矩形的四个顶点
        contours, _ = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # 形状及大小筛选校验
        det_x_max = 0
        det_y_max = 0
        num = 0
        # print('aaa',contours)
        for i in range(len(contours)):
            x_min = np.min(contours[i][:, :, 0])
            x_max = np.max(contours[i][:, :, 0])
            y_min = np.min(contours[i][:, :, 1])
            y_max = np.max(contours[i][:, :, 1])
            det_x = x_max - x_min
            det_y = y_max - y_min
            if (det_x / det_y > 1.8) and (det_x > det_x_max) and (det_y > det_y_max):
                det_y_max = det_y
                det_x_max = det_x
                num = i
        # 获取最可疑区域轮廓点集
        points = np.array(contours[num][:, 0])
        # print(points)
        return points

    def findVertices(self, points):
        # 获取最小外接矩阵，中心点坐标，宽高，旋转角度
        rect = cv2.minAreaRect(points)

        # 获取矩形四个顶点，浮点型
        box = cv2.boxPoints(rect)
        # 取整
        box = np.int0(box)
        # 获取四个顶点坐标
        left_point_x = np.min(box[:, 0])
        right_point_x = np.max(box[:, 0])
        top_point_y = np.min(box[:, 1])
        bottom_point_y = np.max(box[:, 1])

        left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
        right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
        top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
        bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
        # 上下左右四个点坐标
        vertices = np.array([[top_point_x, top_point_y], [bottom_point_x, bottom_point_y], [left_point_x, left_point_y],
                             [right_point_x, right_point_y]])
        # print('vvv',vertices)
        return vertices, rect

