#############################################################
# image resize

import cv2
import numpy as np
import os
from PIL import Image
from keras.preprocessing import image
from keras.preprocessing.image import array_to_img

img_path = 'pictures/'
NO = 1

def resize(x):
    x_out = []

    for i in range(len(x)):
        img = cv2.resize(x[i],dsize=(96,96))
        x_out.append(img)

    return np.array(x_out)

x = []

while True:
    if not os.path.exists(img_path + str(NO) + ".jpg"):
        break
    img = Image.open(img_path + str(NO) + ".jpg")
    img = image.img_to_array(img)
    x.append(img)
    NO += 1

x_train = resize(x)


###########################################################
# Data Augmentation

from keras.preprocessing.image import ImageDataGenerator

X_train = []
aug_num = 6000 # DataAugを何枚用意するか
NO = 1

datagen = ImageDataGenerator(
           rotation_range=10,
           width_shift_range=0.2,
           height_shift_range=0.2,
           fill_mode="constant",
           cval=180,
           horizontal_flip=True,
           vertical_flip=True)

for d in datagen.flow(x_train, batch_size=1):
    X_train.append(d[0])
    # datagen.flowは無限ループするため必要な枚数取得できたらループを抜ける
    if (NO % aug_num) == 0:
        print("finish")
        break
    NO += 1

X_train = np.array(X_train)
X_train /= 255


###########################################################
# Load CIFAR10 images

from keras.datasets import cifar10
from keras.utils import to_categorical

# dataset
(x_ref, y_ref), (x_test, y_test) = cifar10.load_data()
x_ref = x_ref.astype('float32') / 255

#refデータからランダムに6000個抽出
number = np.random.choice(np.arange(0,x_ref.shape[0]),6000,replace=False)

x, y = [], []

x_ref_shape = x_ref.shape

for i in number:
    temp = x_ref[i]
    x.append(temp.reshape((x_ref_shape[1:])))
    y.append(y_ref[i])

x_ref = np.array(x)
y_ref = to_categorical(y)

X_ref = resize(x_ref)


#############################################################
# Learn

from keras.applications import MobileNetV2, VGG16
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras import backend as K
from keras.engine.network import Network

input_shape = (96, 96, 3)
classes = 10
batchsize = 128
#feature_out = 512 #secondary network out for VGG16
feature_out = 1280 #secondary network out for MobileNet
alpha = 0.5 #for MobileNetV2
lambda_ = 0.1 #for compact loss

#損失関数
def original_loss(y_true, y_pred):
    lc = 1/(classes*batchsize) * batchsize**2 * K.sum((y_pred -K.mean(y_pred,axis=0))**2,axis=[1]) / ((batchsize-1)**2)
    return lc

#学習
def train(x_target, x_ref, y_ref, epoch_num):

    # VGG16読み込み, S network用
    print("Model build...")
    #mobile = VGG16(include_top=False, input_shape=input_shape, weights='imagenet')

    # mobile net読み込み, S network用
    mobile = MobileNetV2(include_top=True, input_shape=input_shape, alpha=alpha, weights='imagenet')

    #最終層削除
    mobile.layers.pop()

    # 重みを固定
    for layer in mobile.layers:
        if layer.name == "block_13_expand": # "block5_conv1": for VGG16
            break
        else:
            layer.trainable = False

    model_t = Model(inputs=mobile.input,outputs=mobile.layers[-1].output)

    # R network用　Sと重み共有
    model_r = Network(inputs=model_t.input,
                      outputs=model_t.output,
                      name="shared_layer")

    #Rに全結合層を付ける
    prediction = Dense(classes, activation='softmax')(model_t.output)
    model_r = Model(inputs=model_r.input,outputs=prediction)

    #コンパイル
    optimizer = SGD(lr=5e-5, decay=0.00005)
    model_r.compile(optimizer=optimizer, loss="categorical_crossentropy")
    model_t.compile(optimizer=optimizer, loss=original_loss)

    model_t.summary()
    model_r.summary()

    print("x_target is",x_target.shape[0],'samples')
    print("x_ref is",x_ref.shape[0],'samples')

    ref_samples = np.arange(x_ref.shape[0])
    loss, loss_c = [], []

    print("training...")

    #学習
    for epochnumber in range(epoch_num):
        x_r, y_r, lc, ld = [], [], [], []

        #ターゲットデータシャッフル
        np.random.shuffle(x_target)

        #リファレンスデータシャッフル
        np.random.shuffle(ref_samples)
        for i in range(len(x_target)):
            x_r.append(x_ref[ref_samples[i]])
            y_r.append(y_ref[ref_samples[i]])
        x_r = np.array(x_r)
        y_r = np.array(y_r)

        for i in range(int(len(x_target) / batchsize)):

            #batchsize分のデータロード
            batch_target = x_target[i*batchsize:i*batchsize+batchsize]
            batch_ref = x_r[i*batchsize:i*batchsize+batchsize]
            batch_y = y_r[i*batchsize:i*batchsize+batchsize]

            #target data
            #学習しながら、損失を取得
            lc.append(model_t.train_on_batch(batch_target, np.zeros((batchsize, feature_out))))

            #reference data
            #学習しながら、損失を取得
            ld.append(model_r.train_on_batch(batch_ref, batch_y))

        loss.append(np.mean(ld))
        loss_c.append(np.mean(lc))

        if (epochnumber+1) % 5 == 0:
            print("epoch:",epochnumber+1)
            print("Descriptive loss:", loss[-1])
            print("Compact loss", loss_c[-1])

    #結果グラフ
    #plt.plot(loss,label="Descriptive loss")
    #plt.xlabel("epoch")
    #plt.legend()
    #plt.show()

    #plt.plot(loss_c,label="Compact loss")
    #plt.xlabel("epoch")
    #plt.legend()
    #plt.show()

    return model_t

model = train(X_train, X_ref, y_ref, 5)


#############################################################
# Store ML model (Folder "model" : model, weights, train.csv)

train_num = 1000# number of training data

model_path = "model/"
if not os.path.exists(model_path):
    os.mkdir(model_path)

train = model.predict(X_train)

# model save
model_json = model.to_json()
open(model_path + 'model.json', 'w').write(model_json)
model.save_weights(model_path + 'weights.h5')
np.savetxt(model_path + "train.csv",train[:train_num],delimiter=",")
