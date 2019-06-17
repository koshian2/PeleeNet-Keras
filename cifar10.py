from pelee_net import PeleeNet
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from PIL import Image
import pickle
import os
from tensorflow.contrib.tpu.python.tpu import keras_support

def generator(X, y, batch_size, use_augmentation, shuffle, scale):
    if use_augmentation:
        base_gen = keras.preprocessing.image.ImageDataGenerator(
            horizontal_flip=True,
            width_shift_range=4.0/32.0,
            height_shift_range=4.0/32.0)
    else:
        base_gen = keras.preprocessing.image.ImageDataGenerator()
    for X_base, y_base in base_gen.flow(X, y, batch_size=batch_size, shuffle=shuffle):
        if scale != 1:
            X_batch = np.zeros((X_base.shape[0], X_base.shape[1]*scale,
                                X_base.shape[2]*scale, X_base.shape[3]), np.float32)
            for i in range(X_base.shape[0]):
                with Image.fromarray(X_base[i].astype(np.uint8)) as img:
                    img = img.resize((X_base.shape[1]*scale, X_base.shape[2]*scale), Image.LANCZOS)
                    X_batch[i] = np.asarray(img, np.float32) / 255.0
        else:
            X_batch = X_base / 255.0
        yield X_batch, y_base

def lr_scheduler(epoch):
    x = 0.4
    if epoch >= 70: x /= 5.0
    if epoch >= 120: x /= 5.0
    if epoch >= 170: x /= 5.0
    return x

def train(use_augmentation, use_stem_block):
    tf.logging.set_verbosity(tf.logging.FATAL)
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    # generator
    batch_size = 512
    scale = 7 if use_stem_block else 1
    train_gen = generator(X_train, y_train, batch_size=batch_size,
                          use_augmentation=use_augmentation, shuffle=True, scale=scale)
    test_gen = generator(X_test, y_test, batch_size=1000,
                         use_augmentation=False, shuffle=False, scale=scale)
    
    # network
    input_shape = (224,224,3) if use_stem_block else (32,32,3)
    model = PeleeNet(input_shape=input_shape, use_stem_block=use_stem_block, n_classes=10)
    model.compile(keras.optimizers.SGD(0.4, 0.9), "categorical_crossentropy", ["acc"])

    # GoogleColab TPU training
    # Plese comment out following 4 lines if train on GPUs
    tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
    strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

    scheduler = keras.callbacks.LearningRateScheduler(lr_scheduler)
    hist = keras.callbacks.History()

    model.fit_generator(train_gen, steps_per_epoch=X_train.shape[0]//batch_size,
                        validation_data=test_gen, validation_steps=X_test.shape[0]//1000,
                        callbacks=[scheduler, hist], epochs=1, max_queue_size=1)
    history = hist.history
    with open(f"pelee_aug_{use_augmentation}_stem_{use_stem_block}.pkl", "wb") as fp:
        pickle.dump(history, fp)

if __name__ == "__main__":
    train(True, True)

