import cv2
from utils import plt_show,plt_show0
import numpy as np
from matplotlib import pyplot as plt

class WordSegment:
    def __init__(self,img):
        self.img = img
    def find_words(self,kernel_h, kernel_w):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_h,kernel_w))
        dia_image = cv2.dilate(self.img, kernel)
        # plt_show(dia_image)
        # 查找轮廓
        contours, hierarchy = cv2.findContours(dia_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        words = []
        word_images = []
        #对所有轮廓逐一操作
        for item in contours:
            rect = cv2.boundingRect(item)
            x = rect[0]
            y = rect[1]
            weight = rect[2]
            height = rect[3]
            word = [x,y,weight,height]
            words.append(word)
            # cv2.rectangle(image, (x,y), (x+weight, y+height), (0, 255, 255), thickness=3)
        # 排序，车牌号有顺序。words是一个嵌套列表
        words = sorted(words,key=lambda s:s[0],reverse=False)
        # print(words)
        # plt_show0(image)
        #word中存放轮廓的起始点和宽高
        for word in words:
            # 筛选字符的轮廓
            if (word[3] > (word[2])) and (word[3] < (word[2] * 3)) and (word[3] > 50):
                splite_image = self.img[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
                word_images.append(splite_image)

        # for i,j in enumerate(word_images):
        #     plt.subplot(1,8,i+1)
        #     plt.imshow(word_images[i],cmap='gray')
        # plt.show()

        return word_images