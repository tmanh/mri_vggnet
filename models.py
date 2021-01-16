import tensorflow as tf
import tensorflow.keras as keras


def conv2d(x, filters=32, kernel_size=3, strides=(1, 1)):
    x = keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ELU()(x)
    return x


def fc(x, units):
    x = keras.layers.Dense(units)(x)
    x = keras.layers.ELU()(x)
    return x


def out_fc(x):
    x = keras.layers.Dense(2)(x)
    x = keras.layers.ELU()(x)
    return x


def create_network(image_size, image_dim):
    """This function is used to create the deep network.

    Args:
        :param image_size: the size of the input image.
        :param image_dim: the number of channels of the input image.
        :param n_classes: the number of classes need to be classified.
        :param classifier_activation: the activation function for the classification.
        :param modality: the modality to use (resnet50, vgg16, googlenet).
        :param fusion_mode: the fusion mode to use (concatenate, sum, paper, adc, t2wi) (using concatenation, element-wise sum, and the proposed one in the paper, only use adc, only use t2wi).
        :param combine_mode: c1, c2, c3. c1 just concat, c2 pca then concat, c3 concat prediction.
        :param training: indicate if we train the deep model or just use it as a feature extractors.
    """

    # define two sets of inputs
    input_1 = keras.Input(shape=(image_size, image_size, image_dim))
    input_2 = keras.Input(shape=(image_size, image_size, image_dim))

    # concat all input
    input_data = keras.layers.Concatenate(axis=3)([input_1, input_2])

    # the following are the convolution blocks used in the paper
    conv1 = conv2d(input_data)
    conv2 = conv2d(conv1)
    conv3 = conv2d(conv2, kernel_size=1)
    drop_1 = keras.layers.Dropout(0.3)(conv3)
    maxpool_1 = keras.layers.MaxPool2D(pool_size=(2, 2))(drop_1)

    conv4 = conv2d(maxpool_1, filters=64)
    conv5 = conv2d(conv4, filters=64)
    conv6 = conv2d(conv5, filters=64, kernel_size=1)
    drop_2 = keras.layers.Dropout(0.3)(conv6)
    maxpool_2 = keras.layers.MaxPool2D(pool_size=(2, 2))(drop_2)

    conv7 = conv2d(maxpool_2, filters=64)
    conv8 = conv2d(conv7, filters=64)
    conv9 = conv2d(conv8, filters=64, kernel_size=1)
    drop_3 = keras.layers.Dropout(0.3)(conv9)
    maxpool_3 = keras.layers.MaxPool2D(pool_size=(2, 2))(drop_3)

    flatten_1 = keras.layers.Flatten()(maxpool_3)

    fc1 = fc(flatten_1, 256)
    drop_4 = keras.layers.Dropout(0.3)(fc1)

    fc2 = fc(drop_4, 64)
    drop_5 = keras.layers.Dropout(0.3)(fc2)

    fc3 = fc(drop_5, 16)
    predict = out_fc(fc3)

    # build model
    model = keras.Model(inputs=(input_1, input_2), outputs=predict)

    return model
