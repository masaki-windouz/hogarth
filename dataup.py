#画像の水増し

import os
import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array, array_to_img
import shutil
from size import return_size

# 画像を拡張する関数
def draw_images(generator, x, dir_name, index):
    save_name = 'extened-' + str(index)
    g = generator.flow(x, batch_size=1, save_to_dir=output_dir,
                       save_prefix=save_name, save_format='png')

    # 1つの入力画像から何枚拡張するかを指定（今回は50枚）
    for i in range(5000):
        bach = g.next()
input_shape = return_size()

categories = ["1","2","3","4","5","6","7"]
# 出力先ディレクトリの設定
for x in range(len(categories)):
    output_dir = os.path.join(".","train",categories[x])
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    print(output_dir)
    os.makedirs(output_dir)

    # 拡張する画像の読み込み
    images = glob.glob(os.path.join(".","origin",categories[x],"*.png"))
    print(images)
    # ImageDataGeneratorを定義
    datagen = ImageDataGenerator(rotation_range= 180, #回転角度の変更域の指定
                                width_shift_range=0.2, #水平方向への移動域の指定
                                height_shift_range=0.2, #垂直方向への移動域の指定
                                # zoom_range=0.08, #拡大率の変更域の指定
                                # zoom_rangeを入れると、元画像がゆがむため外す
                                # fill_mode='constant',# 余白を埋める方法
                                horizontal_flip=True,
                                vertical_flip=True,
                                # channel_shift_range=20 #チャンネル変化の指定
                                fill_mode="constant"
                                ,cval = 255
                                )

    # 読み込んだ画像を順に拡張
    for i in range(len(images)):
        img = load_img(images[i])
        img = img.resize(input_shape)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        draw_images(datagen, x, output_dir, i)