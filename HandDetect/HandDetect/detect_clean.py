import cv2 as cv
import numpy as np
import time
import scripts_clean as sc

IMG_HEIGHT, IMG_WIDTH = 480, 640#720, 1280   # 480, 640
AREA_THRESH = (10000, 100000)#(30000, 300000) # (10000, 100000)
BLUR_COMPENSATION = 35 #35 #20
FILTER_WEIGHT = (0.2,0.3,0.5)

REC_HEIGHT, REC_WIDTH = 200, 200 #300, 300 # 200, 200
REC_SAVE_HEIGHT, REC_SAVE_WIDTH = 100, 100

cam=cv.VideoCapture(0)
cam.set(cv.cv2.CAP_PROP_FRAME_HEIGHT,IMG_HEIGHT)
cam.set(cv.cv2.CAP_PROP_FRAME_WIDTH,IMG_WIDTH)
fgbg=cv.createBackgroundSubtractorMOG2()
font=cv.FONT_HERSHEY_SIMPLEX

t0=time.time()
flt_l=sc.myFilter([0,0,0],FILTER_WEIGHT)
flt_u=sc.myFilter([0,0,0],FILTER_WEIGHT)

sc.recognizer.init()
sc.driver.init()
#time.sleep(5)


while(True):
	ret,frame=cam.read()
	frame=cv.flip(frame, 1)  # flip to fit the screen
	fmask=fgbg.apply(frame)
	
	maskSkin=sc.skinDetect(frame)
	mask=cv.bitwise_and(fmask, maskSkin)

	mask=cv.GaussianBlur(mask, (13,13), 5)
	ret, mask=cv.threshold(mask, 240, 255, cv.THRESH_BINARY)
	
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
	if ((s > AREA_THRESH[0]) & (s < AREA_THRESH[1])):    # Ignore it if it's too big or small
		l = flt_l.attach(l)  # Smooth the movement
		u = flt_u.attach(u)
		l = max(0, l-BLUR_COMPENSATION)  # Add some compensation
		u = max(0, u-BLUR_COMPENSATION)
		'''
		sc.wapi.mouse_event(sc.wcon.MOUSEEVENTF_LEFTDOWN,0,0,0,0)
		sc.wapi.mouse_event(sc.wcon.MOUSEEVENTF_ABSOLUTE|sc.wcon.MOUSEEVENTF_MOVE,(l*2+1000)*35,(u*2+200)*60)
		#sc.wapi.SetCursorPos([l*2+400,u*2+200])
		sc.wapi.mouse_event(sc.wcon.MOUSEEVENTF_LEFTUP,0,0,0,0)
		'''
		#===================================================================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		subImg = frame[u:u + REC_HEIGHT, l:l + REC_WIDTH, :]  # Capture hand image
		temp=np.zeros(np.shape(subImg))
		temp[:,:,0]=cv.cvtColor(subImg,cv.COLOR_BGR2GRAY)  
		temp[:,:,1]=maskSkin[u:u + REC_HEIGHT, l:l + REC_WIDTH]
		temp[:,:,2]=sc.binaryMask(frame)[u:u + REC_HEIGHT, l:l + REC_WIDTH]
		temp = cv.resize(temp, (REC_SAVE_WIDTH, REC_SAVE_HEIGHT), interpolation=cv.INTER_LINEAR)
		#cv.imshow('A',temp)
		#cv.imwrite('C:/Users/raosh/Desktop/s.jpg',temp)
		predVal = sc.recognizer.predict(temp)
		#===================================================================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		sc.driver.attach(l,u,np.argmax(predVal))
		print(predVal)
		#cv.rectangle(frame, (l,u), (l+REC_WIDTH,u+REC_HEIGHT), (255,0,0), 2)
		#frame=cv.putText(frame,str(np.argmax(predVal)),(l,u),font,1.5,(255,0,0),2)
		

	cv.imshow('Frame', frame)

	t = time.time()
	dt = t - t0
	t0 = t
	print('fps:', 1/dt)
	
	k=cv.waitKey(1) & 0xFF
	if k==27:
		break
	

cam.release()
