from keras.layers import Input, Dense
from keras.models import Model
from efficientnet.keras import EfficientNetB4

def efficient_model():
    input_layer = Input(shape=(300, 300, 3))
    efficient_net = EfficientNetB4(
        weights='noisy-student',
        include_top=False,
        input_tensor = input_layer,
        pooling='max')
    for layer in efficient_net.layers:
        layer.trainable = True
    x = efficient_net.output 
    x = Dense(units=128, activation='relu')(x)
    output = Dense(units=4, activation='softmax', name='class_output')(x)
    return Model(inputs=input_layer, outputs=output)
