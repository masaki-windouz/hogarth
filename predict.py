from keras.layers import Input, Flatten, Dense, Dropout, LeakyReLU, BatchNormalization, Conv2D
import os
from keras import optimizers
from keras.utils import np_utils,plot_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
from keras.models import Model,load_model
from model import return_data
import os, glob
import cv2

def main():
	os.environ ['KMP_DUPLICATE_LIB_OK'] = 'True'
	categories = ["hula","india","jawa","myan","nichibu"]
	curve = ["1","2","3","4","5","6","7"]

	images = glob.glob(os.path.join(".","R",categories[0],"*.png"))
	print(images)

	# image = Image.open(images[0])
	image = cv2.imread(images[2],cv2.IMREAD_COLOR)
	data = np.asarray(image)
	print(data.shape)
	size = (32,32)
	# img_resize = image.resize((32,32))
	img_resize = cv2.resize(image,size)
	data = np.asarray(img_resize)
	print(data.shape)
	data = data[np.newaxis,:,:,:]
	# sys.exit()
	categories = ["1","2","3","4","5","6","7"]
	# X_train, X_test, y_train, y_test = return_data()
	#モデルの学習
	model = load_model("./models-32/model_04_0.86.h5")
	predict = model.predict(data)
	print(predict)
	cv2.imshow("image",image)
	cv2.waitKey(0)
	cv2.destroyAllWindous()
	# true = 0
	# ndata = len(y_test)
	# for i in range(len(y_test)):
	# 	# print(predict[i],y_test[i])
	# 	pre = np.argmax(predict[i])
	# 	test = np.argmax(y_test[i])
	# 	print(pre,test)
	# 	if pre == test:
	# 		true += 1
	# print("正解率 = ",true*100/ndata)


if __name__ == '__main__':
	main()
