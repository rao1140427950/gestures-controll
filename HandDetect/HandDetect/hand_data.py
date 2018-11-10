import tensorflow as tf
CAPACITY=150
MIN_AFTER_DEQUEUE=50

# A class to read training data in my dataset
class train_data:

	__CSV_PATH='Gestures/Training Set New/list.csv'  # CSV file path
	__image_list=[]
	__label_list=[]
	__height=100  # The dimensions of each image ===================================================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	__width=100
	__label_depth=7
	__num_threads=32


	@classmethod
	def init(cls,path=__CSV_PATH):  # Decode csv file and init image_list, label_list
		fid=open(path)
		for lst in fid.readlines():
			temp=lst.strip().split(",")
			cls.__image_list.append(temp[0])
			cls.__label_list.append(int(temp[1]))  # Convert to 0,1 mode (for one-hot)

	@classmethod
	def get_batch(cls,batch_size,capacity=CAPACITY,min_after_dequeue=MIN_AFTER_DEQUEUE,img_height=__height,img_width=__width):  # Get batch in specific size from the data set
		image=tf.cast(cls.__image_list,tf.string)
		label=tf.cast(cls.__image_list,tf.int32)  # Modify data type as well as transfer array to tf tensor
		label=tf.one_hot(cls.__label_list,cls.__label_depth)  # Change to one-hot mode
		
		input_queue=tf.train.slice_input_producer([image,label])  # Create a queue
		
		#  Get images and labels
		image_contents=tf.read_file(input_queue[0])
		image=tf.image.decode_png(image_contents,channels=3) #===============================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		image=tf.image.resize_image_with_crop_or_pad(image,img_height,img_width)
		image=tf.image.per_image_standardization(image)
		label=input_queue[1]

		#  Create batch
		image_batch,label_batch=tf.train.shuffle_batch([image,label],batch_size,capacity,min_after_dequeue,num_threads=cls.__num_threads)
		
		return image_batch,label_batch


# A class to read testing data in my dataset
class test_data:

	__CSV_PATH='F:/Machine Learning/Gestures/Testing Set New/list.csv'  # CSV file path
	__image_list=[]
	__label_list=[]
	__height=100  # The dimensions of each image ==================================================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	__width=100
	__label_depth=7
	__num_threads=32


	@classmethod
	def init(cls,path=__CSV_PATH):  # Decode csv file and init image_list, label_list
		fid=open(path)
		for lst in fid.readlines():
			temp=lst.strip().split(",")
			cls.__image_list.append(temp[0])
			cls.__label_list.append(int(temp[1]))  # Convert to 0,1 mode (for one-hot)

	@classmethod
	def get_batch(cls,batch_size,capacity=CAPACITY,min_after_dequeue=MIN_AFTER_DEQUEUE,img_height=__height,img_width=__width):  # Get batch in specific size from the data set
		image=tf.cast(cls.__image_list,tf.string)
		label=tf.cast(cls.__image_list,tf.int32)  # Modify data type as well as transfer array to tf tensor
		label=tf.one_hot(cls.__label_list,cls.__label_depth)  # Change to one-hot mode
		
		input_queue=tf.train.slice_input_producer([image,label])  # Create a queue
		
		#  Get images and labels
		image_contents=tf.read_file(input_queue[0])
		image=tf.image.decode_png(image_contents,channels=3) #===============================================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		image=tf.image.resize_image_with_crop_or_pad(image,img_height,img_width)
		image=tf.image.per_image_standardization(image)
		label=input_queue[1]

		#  Create batch
		image_batch,label_batch=tf.train.shuffle_batch([image,label],batch_size,capacity,min_after_dequeue,num_threads=cls.__num_threads)
		
		return image_batch,label_batch


