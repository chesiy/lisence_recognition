from matplotlib import pyplot as plt
import cv2

# plt显示灰度图片
def plt_show(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def plt_show0(img):
#cv2与plt的图像通道不同：cv2为[b,g,r];plt为[r, g, b]
    b,g,r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()

# 图像去噪灰度处理
def gray_guss(image):
    image = cv2.GaussianBlur(image, (11, 11), 0)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image

def evaluate(res, truth):
    T = 0
    F = 0
    for key in res.keys():
        if res[key] == truth[key]:
            T += 1
        else:
            F += 1
    return 1.0*T/(T+F)
