import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

template = ['0','1','2','3','4','5','6','7','8','9',
            'A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
            '沪','皖','豫','鲁']

# 读取一个文件夹下的所有图片，输入参数是文件名，返回模板文件地址列表
def read_directory(directory_name):
    referImg_list = []
    for filename in os.listdir(directory_name):
        referImg_list.append(directory_name + "/" + filename)
    return referImg_list

def get_lists(path):
    chinese_words_list = []
    eng_words_list = []
    eng_num_words_list = []

    c_word = read_directory("./"+path+"/province")
    for i in range(len(c_word)):
        chinese_words_list.append([c_word[i]])
    e_word = read_directory("./"+path+"/alphabets")
    for i in range(len(e_word)):
        eng_words_list.append([e_word[i]])
    numb = read_directory("./"+path+"/numbers")
    for i in range(len(numb)):
        eng_num_words_list.append([numb[i]])
    eng_num_words_list+=eng_words_list

    return chinese_words_list,eng_words_list,eng_num_words_list

# 读取一个模板地址与图片进行匹配，返回得分
def template_score(template,image):
    #将模板进行格式转换
    temp_file = np.fromfile(template, dtype=np.uint8)
    # print('aaa',temp_file.shape)
    template_img=cv2.imdecode(temp_file,cv2.IMREAD_COLOR)
    template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)

    #模板图像阈值化处理——获得黑白图
    ret, template_img = cv2.threshold(template_img, 0, 255, cv2.THRESH_OTSU)
    template_img = ~template_img
    image_ = image.copy()
    #获得待检测图片的尺寸
    height, width = image_.shape
    # 将模板resize至与图像一样大小
    template_img = cv2.resize(template_img, (width, height))
    # 模板匹配，返回匹配得分
    result = cv2.matchTemplate(image_, template_img, cv2.TM_CCOEFF)
    return result[0][0]

# 对分割得到的字符逐一匹配
def template_matching(word_images, is_green):
    results = []

    if is_green:
        chinese_words_list, eng_words_list, eng_num_words_list = get_lists('refer2')
    else:
        chinese_words_list, eng_words_list, eng_num_words_list = get_lists('refer0')

    # print(chinese_words_list)
    for index,word_image in enumerate(word_images):
        if index==0:
            best_score = []
            for chinese_words in chinese_words_list:
                score = []
                for chinese_word in chinese_words:
                    result = template_score(chinese_word,word_image)
                    score.append(result)
                best_score.append(max(score))
            i = best_score.index(max(best_score))
            r = template[34+i]
            results.append(r)
            continue
        if index==1:
            best_score = []
            for eng_word_list in eng_words_list:
                score = []
                for eng_word in eng_word_list:
                    result = template_score(eng_word,word_image)
                    score.append(result)
                best_score.append(max(score))
            i = best_score.index(max(best_score))
            # print(template[10+i])
            r = template[10+i]
            results.append(r)
            continue
        else:
            best_score = []
            for eng_num_word_list in eng_num_words_list:
                score = []
                for eng_num_word in eng_num_word_list:
                    result = template_score(eng_num_word,word_image)
                    score.append(result)
                best_score.append(max(score))
            i = best_score.index(max(best_score))
            r = template[i]
            results.append(r)
            continue

    results = "".join(results)

    return results