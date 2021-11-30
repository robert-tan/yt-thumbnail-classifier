# from read_data import read_dataset
import boto3

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import MaxPool2D, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import categorical_crossentropy


# s3 = boto3.resource('s3')
# bucketname = "cs230-project"
# filename = "train1427_processed.tfrecord"
# obj = s3.Object(bucketname, filename)
# body = obj.get()['Body'].read()
# print(body)
tf.enable_eager_execution()

PATH = "../data/train0002_processed.tfrecord"

def read_dataset(file_path):
    dataset = tf.data.TFRecordDataset(file_path)

    def read_tfrecord(serialized_example):
        feature_description = {
            'thumbnail': tf.io.VarLenFeature(tf.string),
            'n_W': tf.io.FixedLenFeature((), tf.int64),
            'n_H': tf.io.FixedLenFeature((), tf.int64),
            'n_C': tf.io.FixedLenFeature((), tf.int64),
            'labels': tf.io.VarLenFeature(tf.int64),
        }
        example = tf.io.parse_single_example(serialized_example, feature_description)
        return example

    return dataset.map(read_tfrecord)


def labeled_dataset(parsed_dataset):
    labeled_data_sets = []

    for example in parsed_dataset:
        thumbnail_bytes = example["thumbnail"].values[0].numpy()
        n_W = example["n_W"].numpy()
        n_C = example["n_C"].numpy()
        n_H = example["n_H"].numpy()

        thumbnail_arr = np.frombuffer(thumbnail_bytes, dtype=np.uint8).reshape((n_W, n_H, n_C))
        labels = example["labels"].values[0].numpy() 

        labeled_dataset = [thumbnail_arr, labels]
        labeled_data_sets.append(labeled_dataset)
    return labeled_data_sets


def model():
    # input
    input = Input(shape =(224,224,3))

    # 1st Conv Block
    x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(input)
    x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    # 2nd Conv Block
    x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    # 3rd Conv block
    x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    # 4th Conv block
    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    # 5th Conv block
    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    # Fully connected layers
    x = Flatten()(x)
    x = Dense(units = 4096, activation ='relu')(x)
    x = Dense(units = 4096, activation ='relu')(x)
    output = Dense(units = 25, activation ='softmax')(x)

    # creating the model
    opt = Adam(learning_rate=0.0001)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=opt, loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    return model


parsed_dataset = read_dataset(PATH)
labeled_data_sets = labeled_dataset(parsed_dataset)
x = []
y = []
for data in labeled_data_sets:
  resized_image = tf.image.resize(data[0], (224, 224)).numpy() / 255.0
  x.append(resized_image)
  y.append(data[1])
x = tf.convert_to_tensor(x)
y = tf.convert_to_tensor(y)
train_x, train_y = x[:800], y[:800]
val_x, val_y = x[800:], y[800:]

model = model()
model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=100, verbose=1)