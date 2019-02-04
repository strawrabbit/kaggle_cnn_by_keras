from keras import layers
from keras import models
import keras
from keras.layers.core import Activation, Dense
from sklearn.model_selection import train_test_split
import pandas as pd
from keras import backend as K

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

train_X,test_X,train_y,test_y = \
train_test_split(train_data.iloc[:,1:],train_data.iloc[:,0])

train_X = train_X.values
test_X = test_X.values
# kerasのbackground種類によって処理を場合分け
if K.image_data_format() == 'channels_first':
    train_X = train_X.reshape(train_X.shape[0], 1, 28, 28)
    x_test = test_X.reshape(test_X.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    train_X = train_X.reshape(train_X.shape[0], 28, 28, 1)
    test_X = test_X.reshape(test_X.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

# データの加工
train_X = train_X.astype('float32')
train_X /= 255
test_X = test_X.astype('float32')
test_X /= 255
train_y = train_y.astype('float32')
train_y = keras.utils.np_utils.to_categorical(train_y, 10)
test_y = test_y.astype('float32')
test_y = keras.utils.np_utils.to_categorical(test_y, 10)


class DeepConvNet_by_keras:
    """

    ネットワーク構成は下記の通り
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        affine - relu - dropout - affine - dropout - softmax
    """

    def __new__(self, input_shape, hidden_size=50, output_size=10):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                                     input_shape=input_shape))
        self.model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

        self.model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(layers.ZeroPadding2D(padding=(2, 2)))
        self.model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

        self.model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Flatten())
        self.model.add(Dense(output_size, activation=None))
        self.model.add(layers.Dropout(0.5))

        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        return self.model

model = DeepConvNet_by_keras(input_shape=input_shape)
history = model.fit(train_X, train_y,
                    batch_size=128, epochs=20,
                    verbose=1, validation_data=(test_X, test_y))
test_data = test_data.values
if K.image_data_format() == 'channels_first':
    test_data = test_data.reshape(test_data.shape[0], 1, 28, 28)
else:
    test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

# test_dataもtrainのように加工
test_data = test_data.astype('float32')
test_data /= 255
predict = model.predict(test_data)
predict = pd.DataFrame({'ImageId': [i + 1 for i in range(predict.shape[0])], 'Label': predict.argmax(axis=1)})
predict_csv = predict.to_csv('predict_csv.csv', index=False)