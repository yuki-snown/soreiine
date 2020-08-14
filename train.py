import os
import numpy as np
from PIL import Image
from keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from efficientnet.keras import EfficientNetB4

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

def efficient_model():
    input_layer = Input(shape=(300, 300, 3))  # 最初の層
    efficient_net = EfficientNetB4(
        weights='noisy-student',  # imagenetでもよい 
        include_top=False,  # 全結合層は自分で作成するので要らない
        input_tensor = input_layer,  # 入力
        pooling='max')
    for layer in efficient_net.layers:  # 転移学習はしない
        layer.trainable = True
    x = efficient_net.output 
    x = Dense(units=128, activation='relu')(x)
    output = Dense(units=4, activation='softmax', name='class_output')(x)
    return Model(inputs=input_layer, outputs=output)

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