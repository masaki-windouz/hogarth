from keras.layers import Input, Flatten, Dense, Dropout, LeakyReLU, BatchNormalization, Conv2D,MaxPooling2D
import os
from keras import optimizers
from keras.utils import np_utils,plot_model
import numpy as np
import matplotlib.pyplot as plt
import sys
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
import keras
os.environ ['KMP_DUPLICATE_LIB_OK'] = 'True'
MODE = "Fanctional"

# MODE = "Sequential"
def return_data():
	global categories, nb_classes, input_shape
	categories = ["1","2","3","4","5","6","7"]

	nb_classes = len(categories)

	(X_train, X_test, y_train, y_test) = np.load("hogarth-curve_data.npy",allow_pickle=True)

	#データの正規化
	X_train = X_train.astype("float32") / 255
	X_test  = X_test.astype("float32")  / 255

	#kerasで扱えるようにcategoriesをベクトルに変換
	y_train = np_utils.to_categorical(y_train, nb_classes)
	y_test  = np_utils.to_categorical(y_test, nb_classes)
	_,height,width,chanel = X_train.shape
	print(X_train.shape)
	input_shape = (height,width,chanel)


	return X_train, X_test, y_train, y_test
def hogarth_model():


	if MODE == "Sequential":
		model = Sequential()
		# 入力画像の大きさ=(250,250,3)
		model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=input_shape))
		# model.add(layers.Conv2D(32,(3,3),activation="relu"))
		model.add(layers.MaxPooling2D((2,2)))
		model.add(layers.Conv2D(64,(3,3),activation="relu"))
		model.add(layers.MaxPooling2D((2,2)))
		model.add(layers.Conv2D(128,(3,3),activation="relu"))
		model.add(layers.MaxPooling2D((2,2)))
		model.add(layers.Conv2D(128,(3,3),activation="relu"))
		model.add(layers.MaxPooling2D((2,2)))
		model.add(layers.Flatten())
		model.add(layers.Dropout(0.5))
		model.add(layers.Dense(512,activation="relu"))
		model.add(layers.Dense(nb_classes,activation="sigmoid")) #分類先の種類分設定
	else:

		conv2D_filters = [32,32,64,64]

		# conv2D_kernel_size = [6,6,6,6]
		conv2D_kernel_size = [3,3,3,3]

		# conv2D_strides = [1,5,1,5]
		conv2D_strides = [1,2,1,2]

		input_layer = Input(shape=input_shape)
		x = input_layer

		for i in range(len(conv2D_filters)):
		# for i in range(2):
			conv_layer = Conv2D(
				filters = conv2D_filters[i]
				,kernel_size = conv2D_kernel_size[i]
				# ,strides = conv2D_strides[i]
				,padding = "same"
				,name = "conv2D_" + str(i)
				)
			x = conv_layer(x)
			x = MaxPooling2D((conv2D_strides[i],conv2D_strides[i]))(x)
			x = BatchNormalization()(x)
			x = LeakyReLU()(x)
			x = Dropout(0.5)(x)
			
		x = Flatten()(x)
		# x = BatchNormalization()(x)
		# x = Dense(512)(x)
		# x = LeakyReLU()(x)
		# x = Dropout(0.5)(x)	
		# x = Dense(256)(x)
		# x = BatchNormalization()(x)
		# x = LeakyReLU()(x)
		# x = Dropout(0.5)(x)
		
		x = Dense(128)(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = Dropout(0.5)(x)
		output_layer = Dense(nb_classes,activation="softmax")(x)
		model = Model(input_layer,output_layer,name="conv2D_model")
	from keras.optimizers import Adam
	opt = Adam(lr=0.0001)
	model.compile(
		loss="categorical_crossentropy",
		# optimizer=optimizers.RMSprop(lr=1e-4),
		optimizer = opt,
		metrics=["acc"]
	)
	plot_model(model,show_shapes=True,expand_nested=True,to_file="model.png")
	model.summary()
	return model



def main():

	X_train, X_test, y_train, y_test = return_data()
	# sys.exit()

	#モデルの学習
	model = hogarth_model()
	if MODE == "Fanctional":
		os.makedirs("models",exist_ok = True)
		model_checkpoint = ModelCheckpoint(
			filepath = os.path.join("models",'model_{epoch:02d}_{val_acc:.2f}.h5')
			,monitor="val_loss"
			,verbose=1
			)
		my_callbacks = keras.callbacks.EarlyStopping(patience=3)
		
		model.fit(
			X_train,
			y_train,
			epochs=20,
			batch_size=8,
			shuffle = True,
			validation_data=(X_test,y_test),
			callbacks = [model_checkpoint,my_callbacks]

		)
		model.save("model.h5",include_optimizer=False)
	else:
		model = model.fit(
			X_train,
			y_train,
			epochs=20,
			batch_size=128,
			validation_data=(X_test,y_test)
		)
		acc = model.history['acc']
		val_acc = model.history['val_acc']
		loss = model.history['loss']
		val_loss = model.history['val_loss']

		epochs = range(len(acc))

		plt.plot(epochs, acc, 'bo', label='Training acc')
		plt.plot(epochs, val_acc, 'b', label='Validation acc')
		plt.title('Training and validation accuracy')
		plt.legend()
		plt.savefig('精度を示すグラフのファイル名')

		plt.figure()

		plt.plot(epochs, loss, 'bo', label='Training loss')
		plt.plot(epochs, val_loss, 'b', label='Validation loss')
		plt.title('Training and validation loss')
		plt.legend()
		plt.savefig('損失値を示すグラフのファイル名')

		
	#モデルの保存

	# json_string = model.model.to_json()
	# open(os.path.join("./model","hogarth_model.json"),"w").write(json_string)
	# yaml_string = model.model.to_yaml()
	# open(os.path.join("./model","hogarth_model.yaml"),"w").write(yaml_string)

	# #重みの保存

	# hdf5_file = "./hogarth_weight.hdf5"
	# model.model.save_weights(os.path.join("./model",hdf5_file))

if __name__ == '__main__':
	main()
