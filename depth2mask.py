from matplotlib import pyplot as plt
import cv2
import numpy as np
import random

img = cv2.imread("./interior/color/0.png", 0)
 
"""
在Sobel函数的第二个参数这里使用了cv2.CV_16S。
因为OpenCV文档中对Sobel算子的介绍中有这么一句：
“in the case of 8-bit input images it will result in truncated derivatives”。
即Sobel函数求完导数后会有负值，还有会大于255的值。而原图像是uint8，即8位无符号数，
所以Sobel建立的图像位数不够，会有截断。因此要使用16位有符号的数据类型，即cv2.CV_16S
"""
x = cv2.Sobel(img,cv2.CV_16S,1,0)
y = cv2.Sobel(img,cv2.CV_16S,0,1)

"""
在经过处理后，别忘了用convertScaleAbs()函数将其转回原来的uint8形式。否则将无法显示图像，而只是一副灰色的窗口。

dst = cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
可选参数alpha是伸缩系数，beta是加到结果上的一个值。结果返回uint8类型的图片
"""
absX = cv2.convertScaleAbs(x)   # 转回uint8
absY = cv2.convertScaleAbs(y)

"""
由于Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted(...)函数将其组合起来
dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])
其中alpha是第一幅图片中元素的权重，beta是第二个的权重，gamma是加到最后结果上的一个值。
"""
dst = cv2.addWeighted(absX,0.5,absY,0.5,0)



fld = cv2.ximgproc.createFastLineDetector(60)
lines = fld.detect(dst)
random.shuffle(lines)
line_img = np.zeros(img.shape, dtype=np.uint8)
for line in lines[:100]:
    line_img = fld.drawSegments(line_img, lines[:100])
line_on_image = cv2.cvtColor(line_on_image, cv2.COLOR_BGR2GRAY)
_, line_on_image = cv2.threshold(line_on_image, thresh=20, maxval=255, type=cv2.THRESH_BINARY)
cv2.imshow('line', line_on_image)
cv2.waitKey(0)
exit()
img_h1 = np.hstack([img, absX])
img_h2 = np.hstack([absY, dst])
img_h4 = np.hstack([absX, absY])
img_h3 = np.hstack([dst, line_on_image])
img_all = np.vstack([img_h1, img_h4, img_h3])

plt.figure(figsize=(30,10))
plt.imshow(img_all, cmap=plt.cm.gray)
plt.show()