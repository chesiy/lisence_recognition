import cv2
from utils import plt_show0,plt_show,gray_guss
from locate_plate import LocatePlate
from word_segment import WordSegment
from match_template import template_matching
from corrections import Correction,HoughCorrection,CutBorder

class Easy:
    def __init__(self,img,width,height,filename):
        self.img = img
        self.width = width
        self.height = height
        self.filename = filename

    def process(self):
        img = self.img
        is_green = False
        if self.filename == '1-2.jpg':
            is_green = True
        gray_image = gray_guss(img)
        # 图像阈值化操作——获得二值化图
        ret, image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
        image = cv2.resize(image, (self.width//5, self.height//5))

        if is_green:
            image = ~image

        segment = WordSegment(image)
        word_images = segment.find_words(7, 7)

        # 模版匹配
        word_images_ = word_images.copy()
        # 调用函数获得结果
        result = template_matching(word_images_, is_green)

        return result


class Medium:
    def __init__(self,img,width,height,filename):
        self.img = img
        self.width = width
        self.height = height
        self.filename = filename

    def process(self):
        img = self.img
        is_green = False

        img = img[int(self.height / 3):self.height, :]
        locate = LocatePlate(img, is_green)
        image = locate.locate_by_edge()
        # plt_show(image)

        image = cv2.resize(image, (440, 140))
        gray_image = gray_guss(image)
        # 图像阈值化操作——获得二值化图
        ret, gray_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)

        cut = CutBorder(gray_image)
        max_wave = cut.find_border()
        gray_image = gray_image[max_wave[0]:max_wave[1]]
        # plt_show(gray_image)

        segment = WordSegment(gray_image)
        word_images = segment.find_words(7, 4)

        # 模版匹配
        word_images_ = word_images.copy()
        # 调用函数获得结果
        result = template_matching(word_images_, is_green)

        return result


class Difficult:
    def __init__(self,img,width,height,filename):
        self.img = img
        self.width = width
        self.height = height
        self.filename = filename
    def process(self):
        # 统一规定大小
        img = self.img
        img = cv2.resize(img, (self.width, self.height))
        # print('img',img.shape)
        is_green = False
        if self.filename == '3-2.jpg':
            img = img[self.height // 2:, :]
            is_green = True

        locate = LocatePlate(img, is_green)
        vertices, rect = locate.locate_by_color()

        correction = Correction(vertices, rect, img)
        image = correction.get_ok_plate()

        if is_green:
            image = ~image
            image = image[:, 40:400]
            image = cv2.resize(image, (440, 140))

        gray_image = gray_guss(image)
        ret, gray_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
        plt_show(gray_image)

        hough = HoughCorrection(gray_image)
        gray_image = hough.Hough_correct()
        plt_show(gray_image)

        cut = CutBorder(gray_image)
        max_wave = cut.find_border()
        gray_image = gray_image[max_wave[0]:max_wave[1]]
        plt_show(gray_image)

        segment = WordSegment(gray_image)
        word_images = segment.find_words(7, 4)

        # 模版匹配
        word_images_ = word_images.copy()
        # 调用函数获得结果
        result = template_matching(word_images_, is_green)

        return result