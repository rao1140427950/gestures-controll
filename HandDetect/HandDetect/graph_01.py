# Build neuron net work 
# Model use: AlexNet
import hand_data as hdata
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import numpy as np

# Init training data
hdata.test_data.init()
hdata.train_data.init()

LEARNING_RATE=0.0001
BATCH_SIZE=30
TEST_BATCH_SIZE=20
MODE_SAVE_PATH='Gestures/checkpoint/Gestures_AlexNet.ckpt'
CHECK_POINT='Gestures/checkpoint'
IMG_HEIGHT,IMG_WIDTH = 100,100
LABEL_DEPTH=7

# Define placeholder to feed training data
x=tf.placeholder(tf.float32,[None,IMG_HEIGHT,IMG_WIDTH,3])
y=tf.placeholder(tf.float32,[None,LABEL_DEPTH])
_image,_label=hdata.train_data.get_batch(BATCH_SIZE)
t_image,t_label=hdata.test_data.get_batch(TEST_BATCH_SIZE)
x_image=tf.reshape(x,[-1,IMG_HEIGHT,IMG_WIDTH,3])

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
filter2=tf.Variable(tf.truncated_normal([5,5,64,96],stddev=0.01)) #256
bias2=tf.Variable(tf.truncated_normal([96],stddev=0.1))           #256
conv2=tf.nn.conv2d(norm1,filter2,strides=[1,2,2,1],padding='SAME') #[1,1,1,1]
conv2=tf.nn.relu(tf.add(conv2,bias2))
'''
# Pool layer 2
pool2=tf.nn.avg_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# LRN layer 2:
norm2=tf.nn.lrn(pool2,5,bias=1,alpha=0.001/9.0,beta=0.75)

# Conv layer 3
filter3=tf.Variable(tf.truncated_normal([3,3,96,96],stddev=0.01)) # 256,384
bias3=tf.Variable(tf.truncated_normal([96],stddev=0.1))            # 384
conv3=tf.nn.conv2d(norm2,filter3,strides=[1,1,1,1],padding='SAME')
conv3=tf.nn.relu(tf.add(conv3,bias3))

# Conv layer 4
filter4=tf.Variable(tf.truncated_normal([3,3,384,384],stddev=0.01))
bias4=tf.Variable(tf.truncated_normal([384],stddev=0.1))
conv4=tf.nn.conv2d(conv3,filter4,strides=[1,1,1,1],padding='SAME')
conv4=tf.nn.relu(tf.add(conv4,bias4))


# Conv layer 5
filter5=tf.Variable(tf.truncated_normal([3,3,384,256],stddev=0.01))
bias5=tf.Variable(tf.truncated_normal([256],stddev=0.1))
conv5=tf.nn.conv2d(conv4,filter5,strides=[1,1,1,1],padding='SAME')
conv5=tf.nn.relu(tf.add(conv5,bias5))

# Pool layer 5
pool5=tf.nn.avg_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
pool5=tf.reshape(pool5,[-1,4*4*256])
'''


#============================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
pool5=tf.nn.avg_pool(conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
pool5=tf.reshape(pool5,[-1,4*4*96])
w1=tf.Variable(tf.truncated_normal([4*4*96,256],stddev=0.1))
b1=tf.Variable(tf.truncated_normal([256],stddev=0.1))
y1=tf.add(tf.matmul(pool5,w1),b1)
y1=tf.nn.relu(y1)
y1=tf.nn.dropout(y1,0.5)

w2=tf.Variable(tf.truncated_normal([256,LABEL_DEPTH],stddev=0.1))
b2=tf.Variable(tf.truncated_normal([LABEL_DEPTH],stddev=1))
y_pred=tf.nn.softmax(tf.add(tf.matmul(y1,w2),b2))
#+++++++++++++++++++++++++++++!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

'''
# Link layer 1
w1=tf.Variable(tf.truncated_normal([4*4*256,4096],stddev=0.1))
b1=tf.Variable(tf.truncated_normal([4096],stddev=0.1))
y1=tf.add(tf.matmul(pool5,w1),b1)
y1=tf.nn.relu(y1)
y1=tf.nn.dropout(y1,0.5)

# Link layer 2
w2=tf.Variable(tf.truncated_normal([4096,2048],stddev=0.1))
b2=tf.Variable(tf.truncated_normal([2048],stddev=0.1))
y2=tf.add(tf.matmul(y1,w2),b2)
y2=tf.nn.relu(y2)
y2=tf.nn.dropout(y2,0.5)


# Link layer3, classify layer
w3=tf.Variable(tf.truncated_normal([2048,LABEL_DEPTH],stddev=0.1))
b3=tf.Variable(tf.truncated_normal([LABEL_DEPTH],stddev=1))
y_pred=tf.nn.softmax(tf.add(tf.matmul(y2,w3),b3))
'''

# Define cost
cost=-tf.reduce_sum(tf.multiply(y,tf.log(y_pred)))
train=tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

# Define accuracy
correct_pred=tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.global_variables_initializer()

# Training
t0=time.time()
xx=np.linspace(0,250,250)
yy=np.linspace(0,0,250)
yy2=np.linspace(0,0,250)
saver=tf.train.Saver()
with tf.Session() as sess:

	#=====================================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	sess.run(init)
	#save_model=tf.train.latest_checkpoint(CHECK_POINT)
	#saver.restore(sess,save_model)
	#====================================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	coord=tf.train.Coordinator()
	threads=tf.train.start_queue_runners(coord=coord)

	for i in range(1,5000):
		image,label=sess.run([_image,_label])
		sess.run(train, feed_dict={x:image,y:label})
		if i%50 == 0:
			image,label=sess.run([t_image,t_label])
			image1,label1=sess.run([_image,_label])
			train_accuracy1,loss=sess.run([accuracy,cost],feed_dict={x:image,y:label})
			train_accuracy2=sess.run(accuracy,feed_dict={x:image1,y:label1})
			print('Step:',i,'Accuracy:',train_accuracy1,train_accuracy2,'Loss: ',loss,'Time:',time.time()-t0)
			yy[int(i/50)]=train_accuracy1
			yy2[int(i/50)]=train_accuracy2

	print('Training Finished!')
	input('Press any key to continue...')
	# For saving
	saver=tf.train.Saver()
	saver.save(sess,MODE_SAVE_PATH)
	print('Model Saving Finished!')

	coord.request_stop()
	coord.join(threads)

# Plot
plt.subplot(2,1,1)
plt.plot(xx,yy)
plt.subplot(2,1,2)
plt.plot(xx,yy2)
plt.show()