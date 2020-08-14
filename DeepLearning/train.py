import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from model import efficient_model

def data_loader():
    root_data_path = "./dataset/"
    classes = {"beauty":0, "cute":1, "ero":2, "others":3}
    images, labels = [], []
    for class_ in classes:
        label = classes[class_]
        dir_name = root_data_path+class_+"/"
        dirs = os.listdir(dir_name)
        for dir_ in dirs:
            path = dir_name + dir_
            images.append(image.img_to_array(image.load_img(path, target_size=(300,300,3))))
            labels.append(label)
    x = np.asarray(images, dtype='float32') / 255.0
    y = to_categorical(labels, dtype='float32')
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=10101, stratify=y)
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = data_loader()
    net = efficient_model()
    #net.load_weights("./weight.h5")
    net.compile(optimizer='rmsprop', 
                loss={'class_output': 'categorical_crossentropy'},
                metrics={'class_output': 'accuracy'})
    callbacks = [
        ModelCheckpoint('checkpoint/ep{epoch:03d}-loss{accuracy:.3f}-val_loss{accuracy:.3f}.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    ]
    net.fit(X_train, y_train, epochs=25, batch_size=17, callbacks=callbacks, validation_data=(X_test, y_test))

if __name__ == "__main__":
    main()