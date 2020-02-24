import keras


def build_wsi_classifier(input_shape, lr, output_units):
    """
    Builds a neural network that performs classification on featurized WSIs.

    Args:
        input_shape: shape of features with channels last, for example (400, 400, 128).
        lr (float): learning rate.

    Returns: compiled Keras model.

    """

    def conv_op(x, stride, dropout=0.2):

        # Conv
        l2_reg = keras.regularizers.l2(1e-5)
        x = keras.layers.SeparableConv2D(
            filters=128, kernel_size=3, strides=stride, padding='valid', depth_multiplier=1,
            activation='linear', depthwise_regularizer=l2_reg, pointwise_regularizer=l2_reg,
            bias_regularizer=l2_reg, kernel_initializer='he_uniform'
        )(x)

        # Batch norm
        x = keras.layers.BatchNormalization(axis=-1, momentum=0.99)(x)

        # Activation
        x = keras.layers.LeakyReLU()(x)

        # Dropout
        if dropout is not None:
            x = keras.layers.SpatialDropout2D(dropout)(x)

        return x

    def dense_op(x, n_units, bn, activation, l2_factor):

        # Regularization
        if l2_factor is not None:
            l2_reg = keras.regularizers.l2(l2_factor)
        else:
            l2_reg = None

        # Op
        x = keras.layers.Dense(units=n_units, activation='linear', kernel_regularizer=l2_reg,
                               bias_regularizer=l2_reg, kernel_initializer='he_uniform')(x)

        # Batch norm
        if bn:
            x = keras.layers.BatchNormalization(axis=-1, momentum=0.99)(x)

        # Activation
        if activation == 'lrelu':
            x = keras.layers.LeakyReLU()(x)
        else:
            x = keras.layers.Activation(activation)(x)

        return x

    # Define classifier
    input_x = keras.layers.Input(input_shape)
    x = conv_op(input_x, stride=2)
    x = conv_op(x, stride=2)
    x = conv_op(x, stride=2)
    x = conv_op(x, stride=2)
    x = conv_op(x, stride=2)
    x = conv_op(x, stride=2)
    x = conv_op(x, stride=1)
    x = conv_op(x, stride=1)
    x = keras.layers.Flatten()(x)
    x = dense_op(x, n_units=128, bn=True, activation='lrelu', l2_factor=1e-5)
    x = dense_op(x, n_units=output_units, bn=False, activation='softmax', l2_factor=None)

    # Compile
    model = keras.models.Model(inputs=input_x, outputs=x)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=[keras.metrics.sparse_categorical_accuracy]
    )

    # print('Classifier model:', flush=True)
    # model.summary()

    return model

