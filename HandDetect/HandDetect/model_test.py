import cv2 as cv
import scripts as sc

ORIGNAL_PATH='Gestures/Testing Set/'
NEW_PATH='Gestures/Testing Set New/'

fid1=open(ORIGNAL_PATH+'list.csv','r')
fid2=open(NEW_PATH+'list.csv','a')
for lst in fid1.readlines():
	temp=lst.strip().split(',')
	imgPath=temp[0]
	label=temp[1]
	temp=imgPath.strip().split('/')
	imgName=temp[-1].strip().split('.')[0]
	img=cv.imread(imgPath)
	img=img[0:100,0:1000,:]
	skinMask=sc.skinDetect(img)
	biMask=sc.binaryMask(img)
	img[:,:,0]=cv.cvtColor(img,cv.COLOR_BGR2GRAY)  # [gray,skinMask,biMask]
	img[:,:,1]=skinMask
	img[:,:,2]=biMask     
	cv.imwrite(NEW_PATH+imgName+'.png',img)
	fid2.write(NEW_PATH+imgName+'.png,'+label+'\n')