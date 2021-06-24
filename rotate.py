import cv2
from math import *
import numpy as np

import cv2
from math import *
import numpy as np


def dumpRotateImage(img, degree):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    #平移操作
    matRotation[0,2] += (widthNew - width)//2
    matRotation[1,2] += (heightNew - height)//2

    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

    matRotation2 = cv2.getRotationMatrix2D((widthNew // 2, heightNew // 2), degree, 1)
    imgRotation2 = cv2.warpAffine(img, matRotation2, (widthNew, heightNew), borderValue=(255, 255, 255))
    return imgRotation,matRotation


def draw_box(img, box):
    cv2.line(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
    cv2.line(img, (box[0], box[1]), (box[4], box[5]), (0, 255, 0), 3)
    cv2.line(img, (box[2], box[3]), (box[6], box[7]), (0, 255, 0), 3)
    cv2.line(img, (box[4], box[5]), (box[6], box[7]), (0, 255, 0), 3)
    return img

def get_boxes(matRotation,box):
    res=[]
    tmp=np.dot(matRotation, np.array([[box[0][0]], [box[0][1]], [1]]))
    tmp=[int(i) for i in tmp]
    res.append(tmp)
    tmp=np.dot(matRotation, np.array([[box[1][0]], [box[1][1]], [1]]))
    tmp=[int(i) for i in tmp]
    res.append(tmp)
    tmp=np.dot(matRotation, np.array([[box[2][0]], [box[2][1]], [1]]))
    tmp=[int(i) for i in tmp]
    res.append(tmp)
    tmp=np.dot(matRotation, np.array([[box[3][0]], [box[3][1]], [1]]))
    tmp=[int(i) for i in tmp]
    res.append(tmp)
    return res

# if __name__ == "__main__":
#     image = cv2.imread('OneLine.jpg')
#     imgRotation, imgRotation2, matRotation = dumpRotateImage(image, -10)
#     box = [0, 0, 50, 0, 50, 50, 0, 50]
#
#     # reverseMatRotation = cv2.invertAffineTransform(matRotation)
#     # pt1 = np.dot(reverseMatRotation, np.array([[box[0]], [box[1]], [1]]))
#     # pt2 = np.dot(reverseMatRotation, np.array([[box[2]], [box[3]], [1]]))
#     # pt3 = np.dot(reverseMatRotation, np.array([[box[4]], [box[5]], [1]]))
#     # pt4 = np.dot(reverseMatRotation, np.array([[box[6]], [box[7]], [1]]))
#
#     pt1 = np.dot(matRotation, np.array([[box[0]], [box[1]], [1]]))
#     pt2 = np.dot(matRotation, np.array([[box[2]], [box[3]], [1]]))
#     pt3 = np.dot(matRotation, np.array([[box[4]], [box[5]], [1]]))
#     pt4 = np.dot(matRotation, np.array([[box[6]], [box[7]], [1]]))
#
#     pt1 = [int(i) for i in pt1]
#     pt2 = [int(i) for i in pt2]
#     pt3 = [int(i) for i in pt3]
#     pt4 = [int(i) for i in pt4]
#
#     print(pt1, pt2, pt3, pt4)
#     box2 = [pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1], pt4[0], pt4[1]]
#
#     cv2.imwrite('./drawBox.png', draw_box(imgRotation, box2))
#     cv2.imwrite('./raw.png', draw_box(image, box))
#     # cv2.waitKey(0)