import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
from captcha.image import ImageCaptcha
import random
import numpy as np
from tensorflow.keras.models import load_model
import string
from PIL import Image
import sys
import os
import logging
logging.basicConfig(level=logging.INFO,filename='test.log')
from logger import Logger

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


def decode(y):
    y = np.argmax(np.array(y), axis=2)[:, 0]
    return ''.join([characters[x] for x in y])


if __name__ == '__main__':
    log = Logger('all.log', level='debug')
    Logger('error.log', level='error').logger.error('error')
    try:
        log.logger.info("打印信息")
        characters = string.digits + string.ascii_lowercase
        model = load_model('D:\\PythonCode\\verify_code\\cnn.h5')
        # model = load_model('cnn.h5')
        X1 = np.zeros((1, 40, 100, 3), dtype=np.float32)
        y1='q6ul'
        im = Image.open(r'D:\\PythonCode\\verify_code\\temp_pics\\q6ul_6695ab6dde1e537cc3fe3458c72c7677.jpg')
        # im = Image.open(r'temp_pics/q6ul_6695ab6dde1e537cc3fe3458c72c7677.jpg')
        X1[0]=np.array(im) / 255.0
        y_pred = model.predict(X1)
        print(sys.argv[1])
        print(decode(y_pred))
        plt.title('real: %s\npred:%s' % (y1, decode(y_pred)))
        plt.imshow(X1[0], cmap='gray')
        plt.axis('off')
        plt.show()
    except Exception as e:
        log.logger.error(e)