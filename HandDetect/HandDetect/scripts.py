# Some support functions
import cv2 as cv
import numpy as np

def skinDetect(frame):
	hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)

	# 设定阈值
	lower1=np.array([0,51,50]) # [0,51,120]
	upper1=np.array([13,153,255]) # [13,153,255]
	lower2=np.array([189,51,50]) # [167,51,120]
	upper2=np.array([180,153,255]) # [180,153,255]

	# 根据阈值构建掩模
	mask1=cv.inRange(hsv,lower1,upper1)
	mask2=cv.inRange(hsv,lower2,upper2)
	mask=cv.bitwise_or(mask1,mask2)

	return mask

def binaryMask(frame):
    # print('use binaryMask model ...')
    minValue = 70
    # 获取灰度图像
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # 高斯模糊:高斯滤波器中像素的权重与其距中心像素的距离成比例

    blur = cv.GaussianBlur(gray, (5, 5), 2) #(9,9)

    # 图像的二值化提取目标,动态自适应的调整属于自己像素点的阈值，而不是整幅图像都用一个阈值
    th3 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2) #17
    ret, res = cv.threshold(th3, minValue, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return res

class imgSaver():
	__TRAINING_PATH = 'F:/Machine Learning/Gestures/Training Set New/'
	__TESTING_PATH = 'F:/Machine Learning/Gestures/Testing Set New/'
	__NEWSET_PATH = 'F:/Machine Learning/Gestures/new_dataset/'
	__path={'train':__TRAINING_PATH, 'test':__TESTING_PATH, 'new':__NEWSET_PATH}
	__ops=''
	__fid=None
	__index=9850
	def __init__(self, args, start_index = None):
		self.__ops = args
		self.__fid = open(self.__path[self.__ops]+'list.csv', 'a')
		if(start_index != None):
			self.__index = start_index
	
	def __del__(self):
		self.__fid.close()
	
	def save(self, img, lable):
		cv.imwrite(self.__path[self.__ops] + str(self.__index) + '.png', img)
		self.__fid.write(self.__path[self.__ops] + str(self.__index)  + '.png,' + str(lable) + '\n')
		self.__index = self.__index + 1




class myFilter():
	__array=[]
	__length=0
	__weight=[]

	def __init__(self, array, weight):
		self.__array=array
		self.__length=len(self.__array)
		self.__weight=weight

	def attach(self, x):
		self.__array.pop(0)
		self.__array.append(x)
		sum=0
		for i in range(self.__length):
			sum=sum + self.__array[i]*self.__weight[i]
		return int(sum)