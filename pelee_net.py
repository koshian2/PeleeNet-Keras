import tensorflow.keras as keras
import tensorflow.keras.layers as layers

def conv_bn_relu(input_tensor, ch, kernel, padding="same", strides=1, weight_decay=5e-4):
    x = layers.Conv2D(ch, kernel, padding=padding, strides=strides,
                      kernel_regularizer=keras.regularizers.l2(weight_decay))(input_tensor)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)

def stem_block(input_tensor):
    x = conv_bn_relu(input_tensor, 32, 3, strides=2)
    branch1 = conv_bn_relu(x, 16, 1)
    branch1 = conv_bn_relu(branch1, 32, 3, strides=2)
    branch2 = layers.MaxPool2D(2)(x)
    x = layers.Concatenate()([branch1, branch2])
    return conv_bn_relu(x, 32, 1)

def dense_block(input_tensor, num_layers, growth_rate, bottleneck_width):
    x = input_tensor
    growth_rate = int(growth_rate / 2)

    for i in range(num_layers):
        inter_channel = int(growth_rate*bottleneck_width/4) * 4
        branch1 = conv_bn_relu(x, inter_channel, 1)
        branch1 = conv_bn_relu(branch1, growth_rate, 3)

        branch2 = conv_bn_relu(x, inter_channel, 1)
        branch2 = conv_bn_relu(branch2, growth_rate, 3)
        branch2 = conv_bn_relu(branch2, growth_rate, 3)
        x = layers.Concatenate()([x, branch1, branch2])
    return x

def transition_layer(input_tensor, k, use_pooling=True):
    x = conv_bn_relu(input_tensor, k, 1)
    if use_pooling:
        return layers.AveragePooling2D(2)(x)
    else:
        return x

def PeleeNet(input_shape=(224,224,3), use_stem_block=True, n_classes=1000):
    n_dense_layers = [3,4,8,6]
    bottleneck_width = [1,2,4,4]
    out_layers = [128,256,512,704]
    growth_rate = 32

    input = layers.Input(input_shape)
    x = stem_block(input) if use_stem_block else input
    for i in range(4):
        x = dense_block(x, n_dense_layers[i], growth_rate, bottleneck_width[i])
        use_pooling = i < 3
        x = transition_layer(x, out_layers[i], use_pooling=use_pooling)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(n_classes, activation="softmax")(x)
    return keras.models.Model(input, x)
