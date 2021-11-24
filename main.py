import cv2
import os
from solutions import Difficult,Medium,Easy
from utils import evaluate
import time

def main():
    reco_res = {}
    ground_truth = {'1-1.jpg': '沪EWM957', '1-2.jpg': '沪AF02976', '1-3.jpg': '鲁NBK268',
                    '2-1.jpg': '沪EWM957', '2-2.jpg': '豫B20E68', '2-3.jpg': '沪A93S20',
                    '3-1.jpg': '沪EWM957', '3-2.jpg': '沪ADE6598', '3-3.jpg': '皖SJ6M07'}

    levels = ['easy','medium','difficult']
    proc_time = {}
    for level in levels:
        time_start = time.time()
        path = './images/'+level
        if level == 'difficult':
            for filename in os.listdir(path):
                img = cv2.imread(path+'/'+filename)
                IMG_h = 800
                IMG_w = 1000
                rec = Difficult(img,IMG_w,IMG_h,filename)
                res = rec.process()
                print(level,filename,res)
                reco_res[filename]=res
        if level == 'medium':
            for filename in os.listdir(path):
                img = cv2.imread(path+'/'+filename)
                height = len(img)
                width = len(img[0])
                rec = Medium(img,width,height,filename)
                res = rec.process()
                print(level,filename,res)
                reco_res[filename]=res
        if level == 'easy':
            for filename in os.listdir(path):
                img = cv2.imread(path+'/'+filename)
                height = len(img)
                width = len(img[0])
                rec = Easy(img,width,height,filename)
                res = rec.process()
                print(level,filename,res)
                reco_res[filename]=res
        time_end = time.time()
        proc_time[level]=(time_end-time_start)/3


    precision = evaluate(reco_res, ground_truth)
    print('识别结果：', reco_res)
    print('正确答案：', ground_truth)
    print('准确率：', precision)
    print('识别时间：',proc_time)

if __name__ == '__main__':
    main()