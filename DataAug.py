from imgaug import augmenters as iaa #引入数据增强的包
sometimes = lambda aug: iaa.Sometimes(0.5, aug) #建立lambda表达式，
import cv2
import os
import utils
import numpy as np
seq = iaa.Sequential([         #建立一个名为seq的实例，定义增强方法，用于增强   # loc 噪声均值，scale噪声方差，50%的概率，对图片进行添加白噪声并应用于每个通道
    # iaa.SomeOf((0,3),[
    #     iaa.Sometimes(p=0.5,
    #                   then_list=[iaa.GaussianBlur(sigma=(0, 0.5))]
    #                   ),  ######以p的概率执行then_list的增强方法
    #     # iaa.Sometimes(p=0.25,
    #     #               then_list=[iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.25)]
    #     #               ),  ######以p的概率执行then_list的增强方法
    #     iaa.OneOf([
    #         iaa.Multiply((0.5, 1.5), per_channel=0.5),
    #         iaa.FrequencyNoiseAlpha(
    #             exponent=(-4, 0),
    #             first=iaa.Multiply((0.8, 1.2), per_channel=True),
    #             second=iaa.ContrastNormalization((0.5, 1.5))
    #         )
    #     ]),
    #     iaa.SimplexNoiseAlpha(iaa.OneOf([
    #         iaa.EdgeDetect(alpha=(0.5, 1.0)),
    #         iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
    #     ])),
    #     iaa.Add((-5, 5), per_channel=0.5),
    #     iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
    #     # iaa.Grayscale(alpha=(0.0, 1.0))
    #     ],
    #     random_order=True
    # )
    # 使用下面的0个到5个之间的方法去增强图像。注意SomeOf的用法
    iaa.SomeOf((0, 4),
               [

                   # iaa.OneOf([
                   #     iaa.Multiply((0.5, 1.5), per_channel=0.5),
                   #     iaa.FrequencyNoiseAlpha(
                   #         exponent=(-4, 0),
                   #         first=iaa.Multiply((0.8, 1.2), per_channel=True),
                   #         second=iaa.ContrastNormalization((0.5, 1.5))
                   #     )
                   # ]),
                   # 用高斯模糊，均值模糊，中值模糊中的一种增强。注意OneOf的用法
                   iaa.OneOf([
                       iaa.GaussianBlur((0, 2.5)),
                       iaa.AverageBlur(k=(2, 6)),  # 核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
                       iaa.MedianBlur(k=(3, 9)),
                   ]),

                   # iaa.SimplexNoiseAlpha(iaa.OneOf([
                   #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                   #     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                   # ])),

                   # 加入高斯噪声
                   iaa.AdditiveGaussianNoise(
                       loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                   ),

                   # # 每个像素随机加减-10到10之间的数
                   # iaa.Add((-10, 10), per_channel=0.5),
                   #
                   # 像素乘上0.5或者1.5之间的数字.
                   iaa.Multiply((0.9, 1.2), per_channel=0.5),
                   #
                   # 将整个图像的对比度变为原来的一半或者二倍
                   iaa.ContrastNormalization((0.9, 1.2), per_channel=0.5),

                   # 将RGB变成灰度图然后乘alpha加在原图上
                   iaa.Grayscale(alpha=(0.0, 1.0)),

               ],

               random_order=True  # 随机的顺序把这些操作用在图像上
               )
],random_order=True)

root_path='./HWDBTrainready_images'
out_root_path='./HWDBTrainready_Aug_images'
utils.makeDirectory(out_root_path)
dir0=os.listdir(root_path)
for file_name in dir0:
    images=[]
    tmp=cv2.imread(os.path.join(root_path,file_name))
    images.append(tmp)
    images_aug = seq.augment_images(images)  # 应用数据增强
    write_path=os.path.join(out_root_path,file_name)
    cv2.imwrite(write_path,images_aug[0])





