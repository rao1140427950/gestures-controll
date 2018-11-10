import tensorflow as tf
import numpy as np
import cv2 as cv
import win32api as wapi
import win32con as wcon

VK_CODE = {  # dict for key event
  'backspace':0x08,
  'tab':0x09,
  'clear':0x0C,
  'enter':0x0D,
  'shift':0x10,
  'ctrl':0x11,
  'alt':0x12,
  'pause':0x13,
  'caps_lock':0x14,
  'esc':0x1B,
  'spacebar':0x20,
  'page_up':0x21,
  'page_down':0x22,
  'end':0x23,
  'home':0x24,
  'left_arrow':0x25,
  'up_arrow':0x26,
  'right_arrow':0x27,
  'down_arrow':0x28,
  'select':0x29,
  'print':0x2A,
  'execute':0x2B,
  'print_screen':0x2C,
  'ins':0x2D,
  'del':0x2E,
  'help':0x2F,
  '0':0x30,
  '1':0x31,
  '2':0x32,
  '3':0x33,
  '4':0x34,
  '5':0x35,
  '6':0x36,
  '7':0x37,
  '8':0x38,
  '9':0x39,
  'a':0x41,
  'b':0x42,
  'c':0x43,
  'd':0x44,
  'e':0x45,
  'f':0x46,
  'g':0x47,
  'h':0x48,
  'i':0x49,
  'j':0x4A,
  'k':0x4B,
  'l':0x4C,
  'm':0x4D,
  'n':0x4E,
  'o':0x4F,
  'p':0x50,
  'q':0x51,
  'r':0x52,
  's':0x53,
  't':0x54,
  'u':0x55,
  'v':0x56,
  'w':0x57,
  'x':0x58,
  'y':0x59,
  'z':0x5A,
  'numpad_0':0x60,
  'numpad_1':0x61,
  'numpad_2':0x62,
  'numpad_3':0x63,
  'numpad_4':0x64,
  'numpad_5':0x65,
  'numpad_6':0x66,
  'numpad_7':0x67,
  'numpad_8':0x68,
  'numpad_9':0x69,
  'multiply_key':0x6A,
  'add_key':0x6B,
  'separator_key':0x6C,
  'subtract_key':0x6D,
  'decimal_key':0x6E,
  'divide_key':0x6F,
  'F1':0x70,
  'F2':0x71,
  'F3':0x72,
  'F4':0x73,
  'F5':0x74,
  'F6':0x75,
  'F7':0x76,
  'F8':0x77,
  'F9':0x78,
  'F10':0x79,
  'F11':0x7A,
  'F12':0x7B,
  'F13':0x7C,
  'F14':0x7D,
  'F15':0x7E,
  'F16':0x7F,
  'F17':0x80,
  'F18':0x81,
  'F19':0x82,
  'F20':0x83,
  'F21':0x84,
  'F22':0x85,
  'F23':0x86,
  'F24':0x87,
  'num_lock':0x90,
  'scroll_lock':0x91,
  'left_shift':0xA0,
  'right_shift ':0xA1,
  'left_control':0xA2,
  'right_control':0xA3,
  'left_menu':0xA4,
  'right_menu':0xA5,
  'browser_back':0xA6,
  'browser_forward':0xA7,
  'browser_refresh':0xA8,
  'browser_stop':0xA9,
  'browser_search':0xAA,
  'browser_favorites':0xAB,
  'browser_start_and_home':0xAC,
  'volume_mute':0xAD,
  'volume_Down':0xAE,
  'volume_up':0xAF,
  'next_track':0xB0,
  'previous_track':0xB1,
  'stop_media':0xB2,
  'play/pause_media':0xB3,
  'start_mail':0xB4,
  'select_media':0xB5,
  'start_application_1':0xB6,
  'start_application_2':0xB7,
  'attn_key':0xF6,
  'crsel_key':0xF7,
  'exsel_key':0xF8,
  'play_key':0xFA,
  'zoom_key':0xFB,
  'clear_key':0xFE,
  '+':0xBB,
  ',':0xBC,
  '-':0xBD,
  '.':0xBE,
  '/':0xBF,
  '`':0xC0,
  ';':0xBA,
  '[':0xDB,
  '\\':0xDC,
  ']':0xDD,
  "'":0xDE,
  '`':0xC0}

# Detect skin color
# frame -> mask
def skinDetect(frame):
	hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)

	# 设定肤色阈值
	lower1=np.array([0,51,50])
	upper1=np.array([13,153,255])
	lower2=np.array([167,51,50])
	upper2=np.array([180,153,255])

	# 根据阈值构建掩模
	mask1=cv.inRange(hsv,lower1,upper1)
	mask2=cv.inRange(hsv,lower2,upper2)
	mask=cv.bitwise_or(mask1,mask2)

	return mask

# Detect outline
# frame -> mask
def binaryMask(frame):
    # print('use binaryMask model ...')
    minValue = 70
    # 获取灰度图像
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # 高斯模糊:高斯滤波器中像素的权重与其距中心像素的距离成比例

    blur = cv.GaussianBlur(gray, (5, 5), 2) #(7,7)

    # 图像的二值化提取目标,动态自适应的调整属于自己像素点的阈值，而不是整幅图像都用一个阈值
    th3 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2) #13
    ret, res = cv.threshold(th3, minValue, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return res

# A filter to smooth the movement
# Weighted average filter
class myFilter():
	__array=[]
	__length=0
	__weight=[]

	def __init__(self, array, weight):  # To set the init value
		self.__array=array
		self.__length=len(self.__array)
		self.__weight=weight

	# Attach the value and get the filtered value
	# val -> val
	def attach(self, x):
		self.__array.pop(0)
		self.__array.append(x)
		sum=0
		for i in range(self.__length):
			sum=sum + self.__array[i]*self.__weight[i]
		return int(sum)

# A virtual driver to imitate mouse & keyboard event
class driver():
	__pre=[0,0,0] # [l,u,label]
	__flag=[0,0] # [state,wait_time]  see doc.
	__mouse_h=0
	__mouse_w=0
	__MOUSE_WHEEL_THRESH=10
	__VALID_HEIGHT, __VALID_WIDTH = 1800, 900
	__SCREEN_HEIGHT = 0
	__SCREEN_WIDTH = 0
	__AMP_RATE= (0, 0)
	__START_POINT=(600, 50)

	# Decrease the wait time for the event
	@classmethod
	def dec_flag_waitTime(cls):
		cls.__flag[1] = cls.__flag[1] - 1;
		if cls.__flag[1] <= 0:
			cls.__flag = [0,0]
		return

	# Get screen resolution and set amp rate
	@classmethod
	def init(cls):
		cls.__SCREEN_HEIGHT = wapi.GetSystemMetrics(1)
		cls.__SCREEN_WIDTH = wapi.GetSystemMetrics(0)
		#cls.__AMP_RATE=(cls.__SCREEN_WIDTH/cls.__VALID_WIDTH, cls.__SCREEN_HEIGHT/cls.__VALID_HEIGHT)
		cls.__AMP_RATE=(2, 2)

	# React to the gesture
	@classmethod
	def attach(cls,l,u,label):
		if label == 0:  # Nothing happens
			cls.dec_flag_waitTime()

		elif label == 2:  # Get move mouse command
			wapi.SetCursorPos([int(l*cls.__AMP_RATE[0]+cls.__START_POINT[0]), int(u*cls.__AMP_RATE[1]+cls.__START_POINT[1])])
			if cls.__pre[2] != 2:
				wapi.mouse_event(wcon.MOUSEEVENTF_LEFTUP,0,0,0,0)
			cls.dec_flag_waitTime()

		elif label == 1:  #  Get mouse click command
			wapi.SetCursorPos([int(l*cls.__AMP_RATE[0]+cls.__START_POINT[0]), int(u*cls.__AMP_RATE[1]+cls.__START_POINT[1])])
			if cls.__pre[2] != 1:
				wapi.mouse_event(wcon.MOUSEEVENTF_LEFTDOWN,0,0,0,0)
			cls.dec_flag_waitTime()

		elif label == 3:  #  See doc.
			if cls.__flag[0] == 8:
				temp = u - cls.__mouse_h
				temp2 = l - cls.__mouse_w
				if temp > cls.__MOUSE_WHEEL_THRESH:
					for i in range(50):
						wapi.mouse_event(wcon.MOUSEEVENTF_WHEEL,0,0,1)
				if temp < -cls.__MOUSE_WHEEL_THRESH:
					for i in range(50):
						wapi.mouse_event(wcon.MOUSEEVENTF_WHEEL,0,0,-1)
				if temp2 > cls.__MOUSE_WHEEL_THRESH:
					wapi.keybd_event(VK_CODE['right_arrow'],0,0,0)
					wapi.keybd_event(VK_CODE['right_arrow'],0,wcon.KEYEVENTF_KEYUP,0)
				if temp2 < -cls.__MOUSE_WHEEL_THRESH:
					wapi.keybd_event(VK_CODE['left_arrow'],0,0,0)
					wapi.keybd_event(VK_CODE['left_arrow'],0,wcon.KEYEVENTF_KEYUP,0)
			cls.__mouse_h = u
			cls.__mouse_w = l
			cls.__flag = [8, 3]

		elif label == 4:  # In cooling-off period
			if cls.__flag[0] == 4:
				cls.dec_flag_waitTime()
			elif cls.__flag[0] == 2:  # Amplify the page
				wapi.keybd_event(VK_CODE['ctrl'],0,0,0)
				wapi.mouse_event(wcon.MOUSEEVENTF_WHEEL,0,0,100)
				wapi.keybd_event(VK_CODE['ctrl'],0,wcon.KEYEVENTF_KEYUP,0)
				cls.__flag = [4, 4]
			else:  # Set flag
				cls.__flag = [5, 5]

		elif label == 5:
			if cls.__flag[0] == 4:
				cls.dec_flag_waitTime()
			elif cls.__flag[0] == 5:
				cls.__flag = [6, 5]
			elif cls.__flag[0] == 1:
				cls.__flag = [2, 5]
			else:
				cls.dec_flag_waitTime()

		elif label == 6:
			if cls.__flag[0] == 4:
				cls.dec_flag_waitTime()
			elif cls.__flag[0] == 6:
				wapi.keybd_event(VK_CODE['ctrl'],0,0,0)
				wapi.mouse_event(wcon.MOUSEEVENTF_WHEEL,0,0,-100)
				wapi.keybd_event(VK_CODE['ctrl'],0,wcon.KEYEVENTF_KEYUP,0)
				cls.__flag = [4, 4]
			else:  # Set flag
				cls.__flag = [1, 5]

		cls.__pre = [l,u,label]


# A class to recognize
class recognizer():
	__sess=tf.Session()
	__model_save_path='F:/Machine Learning/Gestures/checkpoint'
	__y_pred=None
	__IMG_HEIGHT=100 #==========================================================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	__IMG_WIDTH=100
	__x=tf.placeholder(tf.float32,[None,__IMG_HEIGHT,__IMG_WIDTH,3])

	@classmethod
	def init(cls):
		cls.build_graph()
		saver=tf.train.Saver()
		#Get saved model
		save_model=tf.train.latest_checkpoint(cls.__model_save_path)
		#Restore saved model
		saver.restore(cls.__sess, save_model)

	@classmethod
	def build_graph(cls):
		LABEL_DEPTH=7

		x_image=tf.reshape(cls.__x,[-1,cls.__IMG_HEIGHT,cls.__IMG_WIDTH,3])

		# Conv layer 1
		filter1=tf.Variable(tf.truncated_normal([11,11,3,64],stddev=0.0001)) #96
		bias1=tf.Variable(tf.truncated_normal([64],stddev=0.0001))           #96
		conv1=tf.nn.conv2d(x_image,filter1,strides=[1,4,4,1],padding='SAME')
		conv1=tf.nn.relu(tf.add(conv1,bias1))

		# Pool layer 1
		pool1=tf.nn.avg_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
		# LRN layer 1: Local response normalization
		norm1=tf.nn.lrn(pool1,5,bias=1,alpha=0.001/9.0,beta=0.75)

		# Conv layer 2
		filter2=tf.Variable(tf.truncated_normal([5,5,64,78],stddev=0.01)) #256
		bias2=tf.Variable(tf.truncated_normal([78],stddev=0.1))           #256
		conv2=tf.nn.conv2d(norm1,filter2,strides=[1,2,2,1],padding='SAME') #[1,1,1,1]
		conv2=tf.nn.relu(tf.add(conv2,bias2))
		
		pool5=tf.nn.avg_pool(conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
		pool5=tf.reshape(pool5,[-1,4*4*78])
		w1=tf.Variable(tf.truncated_normal([4*4*78,256],stddev=0.1))
		b1=tf.Variable(tf.truncated_normal([256],stddev=0.1))
		y1=tf.add(tf.matmul(pool5,w1),b1)
		y1=tf.nn.relu(y1)
		y1=tf.nn.dropout(y1,0.5)

		w2=tf.Variable(tf.truncated_normal([256,LABEL_DEPTH],stddev=0.1))
		b2=tf.Variable(tf.truncated_normal([LABEL_DEPTH],stddev=1))
		y_pred=tf.nn.softmax(tf.add(tf.matmul(y1,w2),b2))
		cls.__y_pred=y_pred

	@classmethod
	def predict(cls,img):
		# img=cv.cvtColor(img,cv.COLOR_BGR2RGB) #=============================================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		image=tf.cast(img,tf.float32)
		image=tf.image.per_image_standardization(image)
		image=tf.reshape(image,[1,cls.__IMG_HEIGHT,cls.__IMG_WIDTH,3])
		image=cls.__sess.run(image)

		return cls.__sess.run(cls.__y_pred, feed_dict={cls.__x:image})
