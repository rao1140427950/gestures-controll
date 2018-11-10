import cv2 as cv

CV_PATH='D:/Anaconda/envs/tensorflow_cpu/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'

#OpenCV face recognition
face_cascade=cv.CascadeClassifier(CV_PATH)
font=cv.FONT_HERSHEY_SIMPLEX

#Open camera and csv file
cam=cv.VideoCapture(0)

while(1):
	ret,frame=cam.read()  #Take a phote
	faces=face_cascade.detectMultiScale(frame)  #Detect faces
	for (x,y,w,h) in faces:
		cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		frame=cv.putText(frame,'Rao',(x,y),font,1.5,(255,0,0),2)
	cv.imshow('Rao1',frame)
	k=cv.waitKey(10) & 0xff
	if k==27:
		break


cam.release()

