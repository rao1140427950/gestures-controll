import cv2 as cv
import numpy as np
import time
import scripts as sc
import win32api as wapi

IMG_HEIGHT, IMG_WIDTH =  480, 640  # 720, 1280
VALID_HEIGHT, VALID_WIDTH = 380, 540
SCREEN_HEIGHT = wapi.GetSystemMetrics(1)
SCREEN_WIDTH = wapi.GetSystemMetrics(0)
AMP_RATE=(SCREEN_WIDTH/VALID_WIDTH, SCREEN_HEIGHT/VALID_HEIGHT)
AREA_THRESH = (10000,100000) #(30000, 300000)
BLUR_COMPENSATION = 35
FILTER_WEIGHT=(0.2,0.3,0.5)

REC_HEIGHT, REC_WIDTH = 200, 200 #=====================================================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
REC_SAVE_HEIGHT, REC_SAVE_WIDTH = 100, 100

cam=cv.VideoCapture(0)
cam.set(cv.cv2.CAP_PROP_FRAME_HEIGHT,IMG_HEIGHT)
cam.set(cv.cv2.CAP_PROP_FRAME_WIDTH,IMG_WIDTH)
fgbg=cv.createBackgroundSubtractorMOG2()

t0=time.time()
flt_l=sc.myFilter([0,0,0],FILTER_WEIGHT)
flt_u=sc.myFilter([0,0,0],FILTER_WEIGHT)

saver=sc.imgSaver('train')

for _ in range(100000):
	ret,frame=cam.read()
	frame=cv.flip(frame,1)
	#frame=cv.GaussianBlur(frame,(5,5),0)
	fmask=fgbg.apply(frame)
	
	maskSkin=sc.skinDetect(frame)
	mask=cv.bitwise_and(fmask,maskSkin)

	mask=cv.GaussianBlur(mask,(15,15),5)
	ret, mask=cv.threshold(mask,240,255,cv.THRESH_BINARY)
	#mask=cv.GaussianBlur(mask,(15,15),5)
	#ret, mask=cv.threshold(mask,230,255,cv.THRESH_BINARY)

	
	u,b,r,l=0,0,0,0
	for i in range(0, IMG_HEIGHT, 1):
		if np.max(mask[i]) == 255:
			u=i
			break
	for i in range(IMG_HEIGHT - 1, -1, -1):
		if np.max(mask[i]) == 255:
			b=i
			break
	
	for i in range(0, IMG_WIDTH, 1):
		if np.max(mask[:,i]) == 255:
			l=i
			break
	for i in range(IMG_WIDTH - 1, -1, -1):
		if np.max(mask[:,i]) == 255:
			r=i
			break

	s=(b-u+1)*(r-l+1)
	'''
	subImg = frame[100:100+REC_HEIGHT,100:100+REC_WIDTH,:]
	subImg=cv.resize(subImg, (REC_SAVE_WIDTH, REC_SAVE_HEIGHT), interpolation=cv.INTER_LINEAR)
	saver.save(subImg, 0)
	'''
	if ((s > AREA_THRESH[0]) & (s < AREA_THRESH[1])):    # Create dateset
		# cv.rectangle(mask,(l,u),(r,b),255)
		# cv.rectangle(frame,(l,u),(r,b),(255,0,0),2)
		l = flt_l.attach(l)
		u = flt_u.attach(u)
		l = max(0,l-BLUR_COMPENSATION)
		u = max(0,u-BLUR_COMPENSATION)

		#===============================================================================================================================
		#l,u=100,100
		subImg = frame[u:u + REC_HEIGHT, l:l + REC_WIDTH, :]
		#subImg = frame[100:100+REC_HEIGHT,100:100+REC_WIDTH,:]
		temp=np.zeros(np.shape(subImg))
		temp[:,:,0]=cv.cvtColor(subImg,cv.COLOR_BGR2GRAY)  
		temp[:,:,1]=maskSkin[u:u + REC_HEIGHT, l:l + REC_WIDTH]
		temp[:,:,2]=sc.binaryMask(frame)[u:u + REC_HEIGHT, l:l + REC_WIDTH]

		temp=cv.resize(temp, (REC_SAVE_WIDTH, REC_SAVE_HEIGHT), interpolation=cv.INTER_LINEAR)
		#cv.imshow('B',maskSkin)
		#saver.save(temp, 2)
		#===============================================================================================================================
		#cv.rectangle(mask, (l,u), (l+REC_WIDTH,u+REC_HEIGHT), 255, 2)
		cv.rectangle(frame, (l,u), (l+REC_WIDTH,u+REC_HEIGHT), (0,255,0), 2)
		#cv.rectangle(frame, (0,0), (VALID_WIDTH,VALID_HEIGHT), (0,0,255), 2)

		#wapi.SetCursorPos([int(l*AMP_RATE[0]), int(u*AMP_RATE[1])])

	#mask=cv.GaussianBlur(mask,(5,5),2)
	#ret, mask=cv.threshold(mask,20,255,cv.THRESH_BINARY)
	#cv.imshow('Mask',mask)
	#cv.imshow('Gray',cv.cvtColor(frame,cv.COLOR_BGR2GRAY))
	#cv.imshow('Bianry',sc.binaryMask(frame))
	#cv.imshow('Canny',cv.threshold(cv.Canny(frame,50,50),100,255,cv.THRESH_BINARY_INV)[1])
	cv.imshow('frame',frame)
	#cv.imshow('fmask',cv.bitwise_and(fmask,fmask,mask=maskSkin))
	#cv.imshow('mask', mask)

	t = time.time()
	dt = t - t0
	t0 = t
	print('fps:', 1/dt)
	
	k=cv.waitKey(100) & 0xFF
	if k==27:
		'''
		cv.imwrite('C:/Users/raosh/Desktop/Bianry10.jpg',sc.binaryMask(frame))
		cv.imwrite('C:/Users/raosh/Desktop/maskSkin10.jpg',maskSkin)
		cv.imwrite('C:/Users/raosh/Desktop/gray10.jpg',cv.cvtColor(frame,cv.COLOR_BGR2GRAY))
		cv.imwrite('C:/Users/raosh/Desktop/frame10.jpg',frame)
		temp=np.zeros(np.shape(frame))
		temp[:,:,0]=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
		temp[:,:,1]=maskSkin
		temp[:,:,2]=sc.binaryMask(frame)
		cv.imwrite('C:/Users/raosh/Desktop/create10.jpg',temp)
		'''
		cv.destroyAllWindows()
		break

	

cam.release()

# ================================ Next step: backgroud mask+skin detect