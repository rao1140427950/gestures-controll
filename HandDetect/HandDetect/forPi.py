import cv2 as cv
import numpy as np
import scripts as sc

class getSubImg():
	
	fgbg=None
	fmask=None
	maskSkin=None
	mask=None
	AREA_THRESH = (10000,100000)
	IMG_HEIGHT, IMG_WIDTH =  480, 640
	BLUR_COMPENSATION = 35
	REC_HEIGHT, REC_WIDTH = 200, 200

	def __init__(self):
		self.fgbg=cv.createBackgroundSubtractorMOG2()
		return

	def attach(self, frame):
		self.fmask=self.fgbg.apply(frame)
		self.maskSkin=self.skinDetect(frame)
		
		self.mask=cv.bitwise_and(self.fmask,self.maskSkin)
		self.mask=cv.GaussianBlur(self.mask,(15,15),5)
		ret, self.mask=cv.threshold(self.mask,240,255,cv.THRESH_BINARY)

		u,b,r,l=0,0,0,0
		for i in range(0, self.IMG_HEIGHT, 1):
			if np.max(self.mask[i]) == 255:
				u=i
				break
		for i in range(self.IMG_HEIGHT - 1, -1, -1):
			if np.max(self.mask[i]) == 255:
				b=i
				break
	
		for i in range(0, self.IMG_WIDTH, 1):
			if np.max(self.mask[:,i]) == 255:
				l=i
				break
		for i in range(self.IMG_WIDTH - 1, -1, -1):
			if np.max(self.mask[:,i]) == 255:
				r=i
				break

		s=(b-u+1)*(r-l+1)

		if ((s > self.AREA_THRESH[0]) & (s < self.AREA_THRESH[1])):    # Create dateset
			l = max(0,l-self.BLUR_COMPENSATION)
			u = max(0,u-self.BLUR_COMPENSATION)

			subImg = frame[u:u + self.REC_HEIGHT, l:l + self.REC_WIDTH, :]
			temp=np.zeros(np.shape(subImg))
			temp[:,:,0]=cv.cvtColor(subImg,cv.COLOR_BGR2GRAY)  
			temp[:,:,1]=self.maskSkin[u:u + self.REC_HEIGHT, l:l + self.REC_WIDTH]
			temp[:,:,2]=sc.binaryMask(frame)[u:u + self.REC_HEIGHT, l:l + self.REC_WIDTH]

			return True, temp
		
		return False, None


	def skinDetect(self, frame):
		hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)

		# 设定阈值
		lower1=np.array([0,51,50]) # [0,51,120]
		upper1=np.array([17,153,255]) # [13,153,255]
		lower2=np.array([189,51,50]) # [167,51,120]
		upper2=np.array([178,153,255]) # [180,153,255]

		# 根据阈值构建掩模
		mask1=cv.inRange(hsv,lower1,upper1)
		mask2=cv.inRange(hsv,lower2,upper2)
		mask=cv.bitwise_or(mask1,mask2)

		return mask
