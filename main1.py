#!/usr/bin/env python
# -*- coding:utf-8 -*-


import cv2
import os
import numpy as np
import random
import utils
import rotate


IMAGES_PATH = './data/test/00012/'  # 图片集地址

IMAGES_FORMAT = ['.png']  # 图片格式

IMAGE_ROW = 5  # 图片间隔，也就是合并成一张图后，一共有几行

IMAGE_COLUMN = 12  # 图片间隔，也就是合并成一张图后，一共有几列

IMAGE_SAVE_PATH = 'final.jpg'  # 图片转换后的地址

wmax=8000      #图像最大宽度
curWmax=0

hInterMin = 0   #行间距离
hInterMax = 15
wInterMax = 300  #列偏移
wInterMin = 20

charInterMin=0   #字符间距范围
charInterMax=10

char_num_min = 4   #字数
char_num_max = 28

totLine_min=10    #行数
totLine_max=20


low_line_rate=0.18    # 少行概率
low_line_min = 4       # 少行下最小
low_line_max = 6       # 少行下最多

HugeWRatio=0.1    #字符大间隔概率
hugeCharInterMin=25  #字符大间隔范围
hugeCharInterMax=100

catInterMin=100     #块间间隔范围
catInterMax=220
catNum1RatioThresh=0.63
catNum2RatioThresh=0.8
catnum3RatioThresh=1

rotate_ratio=2      #旋转角度

rightInter_min=3  #右边边距
rightInter_max=250



char_boxes=list()
char_box=list()

globalH=0
globalW=0
cInters=[]
hInitial=0

posx=0
posy=0

lgt_dir = './HWDB_list_txt'
image_dir = './HWDBTrainready_images'
char_box_dir = './HWDB_char_box'

back_ratio=0.1


def Combine(totBitmap,bitmap,h,w,wmax,hInter,wInter,dirs):

    global cInters
    global globalH,globalW

    block1=np.ones((h, wmax-w), dtype='uint8')*255
    bit=np.hstack((bitmap, block1))

    bitTmp=np.ones(bit.shape, dtype='uint8') * 255
    bitTmp[:,wInter:wmax]=bit[:, 0:(wmax-wInter)]
    bit=bitTmp
    block1=np.ones((hInter, wmax), dtype='uint8')*255
    bit=np.vstack((bit, block1))

    last=np.vstack((totBitmap, bit))

    globalW+=wInter
    char_box=list()
    hmax=last.shape[0]
    lineH=bitmap.shape[0]
    # globalH=hmax-hInter-lineH+hInitial
    globalH=hmax-hInter-lineH
    globalW+=cInters[0]
    i=1
    for dir in dirs:
        # dir=os.path.join(IMAGES_PATH,dir)
        tmpImage=cv2.imread(dir, flags=0)
        h0,w0=tmpImage.shape
        tmp_char_box=[[globalW,globalH],[globalW+w0,globalH],[globalW+w0,globalH+h0],[globalW,globalH+h0]]
        char_box.append(tmp_char_box)
        if i<len(cInters):
            globalW += w0 + cInters[i]
        i+=1

    cInters=list()
    char_boxes.append(char_box)

    return last

def CombineChar(image1,image2,idx):

    h1,w1=image1.shape
    h2,w2=image2.shape
    tmpR=random.random()
    if tmpR<HugeWRatio:
        wInter=random.randint(hugeCharInterMin,hugeCharInterMax)
    else:
        wInter=random.randint(charInterMin,charInterMax)
    # wInter=0

    cInters.append(wInter)

    if h1>h2:
        block1=np.ones((h1-h2,w2),dtype='uint8')*255
        image2=np.vstack((image2,block1))
    elif h2>h1:
        block1=np.ones((h2-h1,w1),dtype='uint8')*255
        image1=np.vstack((image1,block1))

    blockInter=np.ones((image1.shape[0],wInter),dtype='uint8')*255

    res=np.hstack((image1,blockInter))
    res=np.hstack((res,image2))
    return res


def GetOneLine(dirs):
    global globalH,globalW,char_boxes,char_box
    # path1=os.path.join(IMAGES_PATH,'5674.png')
    # path2=os.path.join(IMAGES_PATH,'3215.png')
    # image1=cv2.imread(path1,flags=0)
    # image2=cv2.imread(path2,flags=0)
    #

    image=np.ones((1,1),dtype='uint8')*255
    globalW=1
    globalH=0
    idx=0
    for dir in dirs:
        # dir=os.path.join(IMAGES_PATH,dir)
        tmpImage=cv2.imread(dir,flags=0)
        image=CombineChar(image,tmpImage,idx)
        idx+=1
    return image

def GetCharPaths():
    num=random.randint(char_num_min,char_num_max)
    res=list()
    for i in range(num):
        dir0=os.listdir('./data/test')
        name0=random.sample(dir0,1)
        dir1=os.listdir('./data/test/'+name0[0])
        name1=random.sample(dir1,1)
        res.append('./data/test/'+name0[0]+'/'+name1[0])
    return res


def Rotate(res, char_boxes, boxes, degree):
    global curWmax
    img,mat=rotate.dumpRotateImage(res,degree)
    res_char_boxes=[]
    res_boxes=[]
    for line in char_boxes:
        char_line=[]
        for char_box in line:
            tmp_char_box=rotate.get_boxes(mat,char_box)
            char_line.append(tmp_char_box)
        res_char_boxes.append(char_line)
    for line in boxes:
        tmp_line=rotate.get_boxes(mat,line)
        curWmax = max(curWmax, tmp_line[2][0])
        res_boxes.append(tmp_line)
    return img,res_char_boxes,res_boxes

def ChangeShape(img,h0,w0,bh,bw,flag,back_img,add_back):  #flag=0~back flag=1~img
    global back_ratio
    if h0>bh:
        times=h0/bh
        residue=h0%bh
        img=np.repeat(img,times,0)
        if flag==0:
            block=img[0:residue,:,:]
        elif add_back==0:
            block=np.ones((residue,bw,3),dtype='uint8')*255
        else:
            h,w,k=back_img.shape
            times=residue/h
            residue=residue%h
            block=np.repeat(back_img,times,0)
            tmp=back_img[0:residue,:,:]
            block=np.vstack((block,tmp))

            times=w0/w
            residue=w0%w
            block=np.repeat(block,times,1)
            tmp=block[:,0:residue,:]
            block=np.hstack((block,tmp))

            tmp=np.ones(block.shape,dtype='uint8')*255
            block=cv2.addWeighted(tmp, 1 - back_ratio, block, back_ratio, 0)

        # if(len(block.shape)<3):
        #     block = cv2.cvtColor(block, cv2.COLOR_GRAY2RGB)
        img=np.vstack((img,block))
    if w0>bw:
        times=w0/bw
        residue=w0%bw
        img=np.repeat(img,times,1)
        if flag==0:
            block=img[:,0:residue,:]
        elif add_back==0:
            block=np.ones((h0,residue,3),dtype='uint8')*255
        else:
            h,w,k=back_img.shape
            times=residue/w
            residue=residue%w
            block=np.repeat(back_img,times,0)
            tmp=back_img[:,0:residue,:]
            block=np.vstack((block,tmp))

            times=h0/h
            residue=h0%h
            block=np.repeat(block,times,0)
            tmp=block[0:residue,:,:]
            block=np.vstack((block,tmp))

            tmp=np.ones(block.shape,dtype='uint8')*255
            block=cv2.addWeighted(tmp, 1 - back_ratio, block, back_ratio, 0)

        img=np.hstack((img,block))
    return img[0:h0,0:w0,:]

def AddBack(img,back_img):
    global back_ratio
    img=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    bh,bw,bk=back_img.shape
    h,w,k=img.shape

    h0=max(bh,h)
    w0=max(bw,w)

    img=ChangeShape(img,h0,w0,h,w,1,back_img,0)
    back_img=ChangeShape(back_img,h0,w0,bh,bw,0,back_img,0)

    # ratio=random.uniform(0.1,0.65)
    dst = cv2.addWeighted(img, 1-back_ratio, back_img, back_ratio, 0)
    return dst

def GetOneImg(back_img):

    global char_boxes,curWmax,back_ratio
    global globalH,globalW,hInitial,posx,posy

    # path_dir = os.listdir(IMAGES_PATH)

    hInitial = random.randint(5, 10)
    wInitial = wmax


    globalH=0
    globalW=0

    res=np.ones((hInitial,wInitial), dtype='uint8')*255

    r=random.random()
    if r>low_line_rate:
        totLine = random.randint(totLine_min,totLine_max)
    else:
        totLine = random.randint(low_line_min,low_line_max)

    boxes=list()
    char_boxes=list()
    posy = hInitial
    for line in range(totLine):
        # sample=random.sample(path_dir,5)

        sample=GetCharPaths()

        tmp=GetOneLine(sample)

        hInter=random.randint(hInterMin,hInterMax)
        wInter = random.randint(wInterMin, wInterMax)
        h=tmp.shape[0]
        w=tmp.shape[1]

        res=Combine(res,tmp,h,w,wmax,hInter,wInter,sample)

        posx=wInter

        box = [[int(posx), int(posy)], [int(posx + w), int(posy)], [int(posx + w), int(posy + h)],
               [int(posx), int(posy + h)]]
        boxes.append(box)
        curWmax=max(curWmax,posx+w)
        if (random.randint(0,200))%6==0:
            degree = random.randint(-rotate_ratio, rotate_ratio)
            res, char_boxes, boxes = Rotate(res, char_boxes, boxes, degree)
            resH,resW=res.shape
            if resW<wmax:
                block = np.ones((resH, wmax - resW), dtype='uint8') * 255
                res=np.hstack((block, res))
            elif resW>wmax:
                res=res[:,0:wmax]
        # posy = posy + h + hInter
        posy=res.shape[0]

    res=AddBack(res,back_img)
    rightInter=random.randint(rightInter_min,rightInter_max)
    rightInter=min(wmax-curWmax,rightInter)
    res=res[:,0:curWmax+rightInter,:]

    curWmax=0
    return res,boxes,char_boxes

def GetLastImg(catNum,name):
    global back_ratio
    back_ratio = np.random.normal(0.3,0.2,1)
    back_ratio = back_ratio[0]
    back_ratio = max(back_ratio, 0.05)
    back_ratio = min(back_ratio, 0.65)

    dir= os.listdir('./background')
    back_img_dir= './background/'+random.sample(dir, 1)[0]
    back_img= cv2.imread(back_img_dir)

    totBitmap_file = image_dir+'/'+name + '.jpg'
    totImg,totBoxes,totChar_boxes=GetOneImg(back_img)

    for i in range(1,catNum):
        tmpImg,tmpBoxes,tmpChar_boxes=GetOneImg(back_img)
        th,tw,tk=totImg.shape
        h,w,k=tmpImg.shape
        hh,ww,kk=back_img.shape
        h0=max(th,h)
        # w0=max(tw,w)
        # totImg=ChangeShape(totImg, h0, tw, th, tw, 1, back_img,1)
        # tmpImg=ChangeShape(tmpImg, h0, w, h, w, 1, back_img,1)

        hObj=h0-th
        if hObj>0:
            times=hObj/hh
            residue=hObj%hh
            block=np.repeat(back_img,times,0)
            tmp=back_img[0:residue,:,:]
            block=np.vstack((block,tmp))
            if tw>ww:
                wObj=tw
                times=wObj/ww
                residue=wObj%ww
                block=np.repeat(block,times,1)
                tmp=block[:,0:residue,:]
                block=np.hstack((block,tmp))
            else:
                block=block[:,0:tw:,:]
            tmp = np.ones(block.shape, dtype='uint8') * 255
            block = cv2.addWeighted(tmp, 1 - back_ratio, block, back_ratio, 0)
            totImg=np.vstack((totImg,block))

        hObj=h0-h
        if hObj > 0:
            times=hObj/hh
            residue=hObj%hh
            block=np.repeat(back_img,times,0)
            tmp=back_img[0:residue,:,:]
            block=np.vstack((block,tmp))
            if w>ww:
                wObj=w
                times=wObj/ww
                residue=wObj%ww
                block=np.repeat(block,times,1)
                tmp=block[:,0:residue,:]
                block=np.hstack((block,tmp))
            else:
                block=block[:,0:tw:,:]
            tmp = np.ones(block.shape, dtype='uint8') * 255
            block = cv2.addWeighted(tmp, 1 - back_ratio, block, back_ratio, 0)
            tmpImg=np.vstack((tmpImg,block))

        catInter=random.randint(catInterMin,catInterMax)
        block=ChangeShape(back_img,h0,catInter,back_img.shape[0],back_img.shape[1],0,back_img,0)

        tmp = np.ones(block.shape, dtype='uint8') * 255
        block=cv2.addWeighted(tmp, 1 - back_ratio, block, back_ratio, 0)

        totImg=np.hstack((totImg,block))
        totImg=np.hstack((totImg,tmpImg))

        #坐标转换
        for line,line1 in zip(tmpBoxes,tmpChar_boxes):
            for point in line:
                point[0]+=catInter+tw
            totBoxes.append(line)
            for box in line1:
                for point in box:
                    point[0]+=catInter+tw
            totChar_boxes.append(line1)


    cv2.imwrite(totBitmap_file, totImg)
    np.save(lgt_dir + '/' + name + '.npy', totBoxes)
    np.save(char_box_dir+'/'+name+'.npy',totChar_boxes)

def GetCatNum():
    ratio=random.random()
    if ratio<catNum1RatioThresh:
        return 1
    else:
        return 2

    # else:
    #     return 3


if __name__ == "__main__":

    utils.makeDirectory(lgt_dir)
    utils.makeDirectory(image_dir)
    utils.makeDirectory(char_box_dir)

    NUM=4500
    # cnt=0
    dir0 = os.listdir(image_dir)
    cnt=len(dir0)

    for i in range(NUM):
        name = str(cnt)
        catNum=GetCatNum()
        img = GetLastImg(catNum,name)
        cnt += 1