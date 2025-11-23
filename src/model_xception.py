# model_xception.py
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

def conv_block(x, filters, kernel=(3,3), strides=(1,1), name=None):
    x = layers.SeparableConv2D(filters, kernel, padding='same', use_bias=False, name=(None if not name else name+'_sepconv'))(x)
    x = layers.BatchNormalization(name=(None if not name else name+'_bn'))(x)
    x = layers.Activation('relu')(x)
    return x

def entry_flow(inputs):
    # Block 0
    x = layers.Conv2D(32, (3,3), strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(64, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Block 1 — 128 filters
    residual = layers.Conv2D(128, (1,1), strides=2, padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = conv_block(x, 128)
    x = conv_block(x, 128)
    x = layers.MaxPooling2D((3,3), strides=2, padding='same')(x)

    x = layers.add([x, residual])

    # Block 2 — 256 filters
    residual = layers.Conv2D(256, (1,1), strides=2, padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = conv_block(x, 256)
    x = conv_block(x, 256)
    x = layers.MaxPooling2D((3,3), strides=2, padding='same')(x)

    x = layers.add([x, residual])

    # Block 3 — 728 filters
    residual = layers.Conv2D(728, (1,1), strides=2, padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = conv_block(x, 728)
    x = conv_block(x, 728)
    x = layers.MaxPooling2D((3,3), strides=2, padding='same')(x)

    x = layers.add([x, residual])

    return x


def middle_flow(x, repeats=8):
    for i in range(repeats):
        residual = x
        x = conv_block(x, 728)
        x = conv_block(x, 728)
        x = conv_block(x, 728)
        x = layers.add([x, residual])
    return x

def exit_flow(x, classes):
    residual = layers.Conv2D(1024, (1,1), strides=(2,2), padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = conv_block(x, 728)
    x = conv_block(x, 1024)
    x = layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(x)
    x = layers.add([x, residual])

    x = layers.SeparableConv2D(1536, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.SeparableConv2D(2048, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    if classes == 1:
        out = layers.Dense(1, activation='sigmoid', name='predictions')(x)
    else:
        out = layers.Dense(classes, activation='softmax', name='predictions')(x)
    return out

def build_xception(input_shape=(299,299,3), num_classes=1, weights_path=None, load_imagenet_weights=False):
    inp = layers.Input(shape=input_shape)
    x = entry_flow(inp)
    x = middle_flow(x, repeats=8)
    out = exit_flow(x, num_classes)
    model = Model(inputs=inp, outputs=out, name='custom_xception')

    if load_imagenet_weights:
        # звантажити ваги Xception без top
        try:
            WEIGHT_URL = "https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
            fname = tf.keras.utils.get_file('xception_notop.h5', WEIGHT_URL, cache_subdir='models')
            model.load_weights(fname, by_name=True, skip_mismatch=True)
            print("Loaded ImageNet weights (by_name=True, skip_mismatch=True)")
        except Exception as e:
            print("Не вдалося завантажити ваги:", e)

    if weights_path:
        model.load_weights(weights_path)
    return model
