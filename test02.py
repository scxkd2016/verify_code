import os
import numpy as np
from PIL import Image
import string
from tensorflow.keras.models import load_model


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


def evaluate(model):
    batch_acc = 0
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1).T
    y_true = np.argmax(y_test, axis=-1).T
    batch_acc += (y_true == y_pred).all(axis=-1).mean()
    return batch_acc


if __name__ == '__main__':
    characters = string.digits + string.ascii_lowercase
    model = load_model('cnn.h5')
    X_test, y_test = gen_data('C:\\Users\\miee06\\Desktop\\test', characters)
    print(evaluate(model))
