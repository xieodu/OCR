import numpy as np
import torch
import cv2
from torch.autograd import Variable
from torchvision import transforms
from models.cnn import Net
from toonnx import to_onnx

# 定义全局变量
refPt = []
cropping = False

# 鼠标事件回调函数
def click_and_crop(event, x, y, flags, param):
    global refPt, cropping

    # 左键按下，开始框选
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # 鼠标移动，实时绘制矩形框
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            image_copy = img.copy()
            cv2.rectangle(image_copy, refPt[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("image", image_copy)

    # 左键释放，完成框选
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False

        # 绘制矩形框
        cv2.rectangle(img, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", img)

use_cuda = False
model = Net(10)
# 注意：此处应把pth文件改为你训练出来的params_x.pth，x为epoch编号，
# 一般来讲，编号越大，且训练集（train）和验证集（val）上准确率差别越小的（避免过拟合），效果越好。
model.load_state_dict(torch.load('output/params_30.pth'))
# model = torch.load('output/model.pth')
model.eval()
if use_cuda and torch.cuda.is_available():
    model.cuda()

#to_onnx(model, 3, 28, 28, 'output/params.onnx')

img = cv2.imread('33333.jpg')


#############对图像进行裁剪###############
# 计算裁剪区域的起始坐标
x = (img.shape[1] )// 2  # 横坐标起始位置
y = (img.shape[0] )// 2  # 纵坐标起始位置
# 指定裁剪区域的大小
width = 250  # 裁剪宽度
height = 250  # 裁剪高度
# 对图像进行裁剪
img = img[y-height:y+height-80, x-width:x+width]
# 显示裁剪后的图像
#cv2.imshow("Cropped Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 创建窗口并绑定鼠标事件回调函数
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

while True:
    # 显示图像
    cv2.imshow("image", img)
    key = cv2.waitKey(1) & 0xFF

    # 按键 'r' 重新选择区域
    if key == ord("r"):
        refPt = []  # 清除先前的矩形框
        image_copy = img.copy()  # 重新加载原始图像
        cv2.imshow("image", image_copy)

    # 按键 'c' 完成选择，截取区域并退出
    elif key == ord("c"):
        if len(refPt) == 2:
            roi = img[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
            cv2.imshow("ROI", roi)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
# 关闭窗口


############放大图片，用插值的方式###############
# # 定义放大倍数
# scale_factor = 2.0
# # 计算调整后的尺寸
# new_width = int(img.shape[1] * scale_factor)
# new_height = int(img.shape[0] * scale_factor)
# # 进行尺寸调整，使用双线性插值
# img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
# # 显示调整后的图像
# cv2.imshow("Resized Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##########直接对图像进行压缩################
#img = cv2.resize(img, (560, 560))
# 进行图像压缩
#compressed_img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1]  # 使用JPEG格式进行压缩，设置压缩质量为70（可以根据需求调整）
# 保存压缩后的图像
#with open('compressed_image.jpg', 'wb') as f:
 #   f.write(compressed_img)
# 解码压缩后的图像
#decoded_img = cv2.imdecode(compressed_img, cv2.IMREAD_COLOR)


##################进行锐化操作############
# 定义锐化滤波器
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]], dtype=np.float32)

# 进行卷积操作
img = cv2.filter2D(roi, -1, kernel)

# 显示结果
cv2.imshow('Sharpened Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

##############图像二值化##############
# 分离图像通道
b, g, r = cv2.split(img)
# 进行图像二值化
threshold_value = 50  # 阈值
max_value = 255  # 最大像素值
_, binary_b = cv2.threshold(b, threshold_value, max_value, cv2.THRESH_BINARY)
_, binary_g = cv2.threshold(g, threshold_value, max_value, cv2.THRESH_BINARY)
_, binary_r = cv2.threshold(r, threshold_value, max_value, cv2.THRESH_BINARY)
# 合并通道
img = cv2.merge((binary_b, binary_g, binary_r))
# 显示二值化后的图像
cv2.imshow("Binary Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# #############对图像进行边缘检测，并自动寻找轮廓进行裁剪#################
# # 执行边缘检测（以Canny边缘检测为例）
# edges = cv2.Canny(img, threshold1=50, threshold2=100)
# # 查找轮廓
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # 选择感兴趣的轮廓（此处以最大轮廓为例）
# largest_contour = max(contours, key=cv2.contourArea)
# # 获取轮廓的边界框
# x, y, w, h = cv2.boundingRect(largest_contour)
# # 裁剪图像
# cropped_image = img[y:y+h, x:x+w]
# # 显示裁剪后的图像
# cv2.imshow('Cropped Image', cropped_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# ###########反转图像###############
img = cv2.bitwise_not(img)
# 显示结果图像
cv2.imshow('Inverted Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

################设置膨胀##########################
# 设置膨胀核的大小和形状
kernel_size = 3
kernel = np.ones((kernel_size, kernel_size), np.uint8)
# 执行膨胀操作
img = cv2.dilate(img, kernel, iterations=1)
# 显示结果
cv2.imshow('Dilated Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#
# ############ 进行高斯平滑处理############
# smoothed_img = cv2.GaussianBlur(inverted_img, (5, 5), 0)
#
# # 显示原始图像和平滑后的图像
# cv2.imshow('Smoothed Image', smoothed_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)

img_tensor = transforms.ToTensor()(img)
img_tensor = img_tensor.unsqueeze(0)
if use_cuda and torch.cuda.is_available():
    prediction = model(Variable(img_tensor.cuda()))
else:
    prediction = model(Variable(img_tensor))
pred = torch.max(prediction, 1)[1]
print(prediction)
print(pred)
cv2.imshow("image", img)
cv2.waitKey(0)
