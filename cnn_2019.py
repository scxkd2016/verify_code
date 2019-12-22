#!/usr/bin/env python
# coding: utf-8

# # 导入必要的库
# 我们需要导入一个叫 [captcha](https://github.com/lepture/captcha/) 的库来生成验证码。
# 我们生成验证码的字符由数字和大写字母组成。
# pip install captcha numpy matplotlib tensorflow-gpu
# In[1]:
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random

# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import string

characters = string.digits + string.ascii_lowercase
# print(characters)
width, height, n_len, n_class = 100, 40, 4, len(characters)

# # 防止 tensorflow 占用所有显存
# In[2]:
import tensorflow as tf
import tensorflow.keras.backend as K
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# K.set_session(sess)

# # 定义数据生成器
# In[3]:
from tensorflow.keras.utils import Sequence


class CaptchaSequence(Sequence):
    def __init__(self, characters, batch_size, steps, n_len=4, width=128, height=64):
        self.characters = characters
        self.batch_size = batch_size
        self.steps = steps
        self.n_len = n_len
        self.width = width
        self.height = height
        self.n_class = len(characters)
        self.generator = ImageCaptcha(width=width, height=height)

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        y = [np.zeros((self.batch_size, self.n_class), dtype=np.uint8) for i in range(self.n_len)]
        for i in range(self.batch_size):
            random_str = ''.join([random.choice(self.characters) for j in range(self.n_len)])
            X[i] = np.array(self.generator.generate_image(random_str)) / 255.0
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, self.characters.find(ch)] = 1
        return X, y


# # 测试生成器
# In[4]:
def decode(y):
    y = np.argmax(np.array(y), axis=2)[:, 0]
    return ''.join([characters[x] for x in y])


data = CaptchaSequence(characters, batch_size=1, steps=1)
X, y = data[0]
# plt.imshow(X[0])
# plt.title(decode(y))


# # 定义网络结构
# In[5]:
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

input_tensor = Input((height, width, 3))
x = input_tensor
for i, n_cnn in enumerate([2, 2, 2, 2, 2]):
    for j in range(n_cnn):
        x = Conv2D(32 * 2 ** min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = MaxPooling2D(2)(x)

x = Flatten()(x)
x = [Dense(n_class, activation='softmax', name='c%d' % (i + 1))(x) for i in range(n_len)]
model = Model(inputs=input_tensor, outputs=x)

# # 网络结构可视化
# 可视化需要安装下面的包：
# pip install pydot
# sudo apt install graphviz
# In[10]:
from tensorflow.keras.utils import plot_model
from IPython.display import Image

plot_model(model, to_file='cnn.png', show_shapes=True)
Image('cnn.png')
# In[7]:
model.summary()

from PIL import Image


def gen_data(file_path, characters):
    image_list = os.listdir(file_path)
    X = np.zeros((len(image_list), 40, 100, 3), dtype=np.float32)
    y = [np.zeros((len(image_list), len(characters)), dtype=np.uint8) for i in range(4)]
    for i in range(len(image_list)):
        label = image_list[i].split("_")[0]
        if len(label) != 4:
            print("filename: " + str(label))
            continue
        im = Image.open(os.path.join(file_path, image_list[i]))
        X[i] = (np.array(im) / 255.0)
        for j, ch in enumerate(label):
            y[j][i, :] = 0
            y[j][i, characters.find(ch)] = 1
    return X, y


# # 训练模型
# In[ ]:
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import *
import os

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定第一块GPU可用
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu50%的显存
    config.gpu_options.allow_growth = True  # 程序按需申请内存
    sess = tf.Session(config=config)
    # train_data = CaptchaSequence(characters, batch_size=128, steps=1000)
    # valid_data = CaptchaSequence(characters, batch_size=128, steps=100)
    callbacks = [EarlyStopping(patience=3), CSVLogger('cnn.csv'), ModelCheckpoint('cnn_best.h5', save_best_only=True)]
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-4, amsgrad=True),
                  metrics=['accuracy'])
    # model.fit_generator(train_data, epochs=100, validation_data=valid_data, workers=4, use_multiprocessing=True,
    #                     callbacks=callbacks)
    X_train, y_train = gen_data('C:\\Users\\miee06\\Desktop\\train', characters)
    model.fit(x=X_train, y=y_train, epochs=50, callbacks=callbacks)
    model.save('cnn_best.h5', include_optimizer=False)
    # ### 载入最好的模型继续训练一会
    # In[9]:
    # model.load_weights('cnn_best.h5')
    # callbacks = [EarlyStopping(patience=3), CSVLogger('cnn.csv', append=True),
    #              ModelCheckpoint('cnn_best.h5', save_best_only=True)]
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=Adam(1e-4, amsgrad=True),
    #               metrics=['accuracy'])
    # model.fit_generator(train_data, epochs=100, validation_data=valid_data, workers=4, use_multiprocessing=True,
    #                 callbacks=callbacks)
    #model.fit(x=X_train, y=y_train, epochs=20, callbacks=callbacks)
    # loss,accuracy = model.evaluate()
    # print('\ntrain: loss',loss )
    # print('train: accurancy', accuracy)
    # In[10]:
    #model.load_weights('cnn_best.h5')
    # # 测试模型
    # In[11]:
    # X, y = data[0]
    # y_pred = model.predict(X)
    # plt.title('real: %s\npred:%s'%(decode(y), decode(y_pred)))
    # plt.imshow(X[0], cmap='gray')
    # plt.axis('off')
    # plt.show()
    # # 计算模型总体准确率
    # In[12]:
    # from tqdm import tqdm
    # X_test, y_test = gen_data('C:\\Users\\miee06\\Desktop\\test', characters)
    #
    #
    # def evaluate(model):
    #     batch_acc = 0
    #     y_pred = model.predict(X_test)
    #     y_pred = np.argmax(y_pred, axis=-1).T
    #     y_true = np.argmax(y_test, axis=-1).T
    #     batch_acc += (y_true == y_pred).all(axis=-1).mean()
    #     return batch_acc / len(y_test)


    # print('==================================================  ' + str(evaluate(model)))
    # # 保存模型
    # In[13]:
    # model.save('cnn.h5', include_optimizer=False)

    # # 可视化训练曲线
    # pip install pandas
    # In[14]:
    import pandas as pd

    df = pd.read_csv('cnn.csv')
    df[['loss', 'val_loss']].plot()
    plt.show()
    # In[15]:
    df[['loss', 'val_loss']].plot(logy=True)
    plt.show()
