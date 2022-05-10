# Adapted from Hsieh et al.
# https://github.com/kevinhsieh/non_iid_dml/blob/ddc3111a355f62877cafbbba03998f203d1350e5/apps/caffe/examples/cifar10/2parts/gnlenet_train_val.prototxt.template
from tensorflow.keras import layers, models
from tensorflow.keras.layers import BatchNormalization
from tensorflow_addons.layers import GroupNormalization


def get_lenet_model(input, group_normalization=False):
    model = models.Sequential()
    model.add(input)
    model.add(layers.Conv2D(32, 5, padding="same"))
    model.add(layers.MaxPooling2D((3, 3), strides=2))

    if group_normalization:
        model.add(GroupNormalization(groups=2))
    else:
        model.add(BatchNormalization())

    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(32, 5, padding="same"))

    if group_normalization:
        model.add(GroupNormalization(groups=2))
    else:
        model.add(BatchNormalization())

    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((3, 3), strides=2))

    model.add(layers.Conv2D(64, 5, padding="same"))

    if group_normalization:
        model.add(GroupNormalization(groups=2))
    else:
        model.add(BatchNormalization())

    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((3, 3), strides=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    return model
