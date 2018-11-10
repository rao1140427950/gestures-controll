# -*- coding: utf-8 -*-
import cv2
import numpy as np
cap=cv2.VideoCapture(0)

while(1):
	# 获取每一帧
	ret,frame=cap.read()
	# 转换到HSV
	hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

	# 设定阈值
	lower1=np.array([0,51,50]) # [0,51,120]
	upper1=np.array([13,153,255]) # [13,153,255]
	lower2=np.array([167,51,50]) # [167,51,120]
	upper2=np.array([180,153,255]) # [180,153,255]

	# 根据阈值构建掩模
	mask1=cv2.inRange(hsv,lower1,upper1)
	mask2=cv2.inRange(hsv,lower2,upper2)
	mask=cv2.bitwise_or(mask1,mask2)

	# 对原图像和掩模进行位运算
	res=cv2.bitwise_and(frame,frame,mask=mask)

	# 显示图像
	cv2.imshow('frame',frame)
	cv2.imshow('mask',mask)
	cv2.imshow('res',res)
	k=cv2.waitKey(5)&0xFF
	if k==27:
		break

# 关闭窗口
cv2.destroyAllWindows()
