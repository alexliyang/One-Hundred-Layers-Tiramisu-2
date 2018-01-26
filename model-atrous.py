from keras.backend import set_image_data_format, set_image_dim_ordering
from keras.layers import Conv2D, Conv2DTranspose, Input, Concatenate, AtrousConvolution2D
from keras.layers.core import Dropout, Activation, Reshape
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

import json

"""
data_format: A string, one of channels_last (default) or channels_first.
channels_last corresponds to inputs with shape (batch, height, width, channels).
channels_first corresponds to inputs with shape (batch, channels, height, width).
It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json.
I recommend using default setting rather than using explicit declaration.

axis=1 <--> data_format='channels_first'
axis=-1 <-> data_format='channels_last'
"""
set_image_data_format('channels_last')
set_image_dim_ordering('tf')

# weight_decay = 0.0001
from keras.regularizers import l2
 
class Tiramisu():

    def __init__(self, input_shape=(224, 224, 3), classes=12,
                 first_conv_filters=48, growth_rate=2, pools=5,
                 block_layers=[4, 5, 7, 10, 12]):

        if type(block_layers) == list:
            assert(len(block_layers) == pools)
        elif type(block_layers) == int:
            block_layers = [block_layers] * pools
        else:
            raise ValueError

        self.input_shape = input_shape
        self.first_conv_filters = first_conv_filters
        self.growth_rate = growth_rate
        self.pools = pools
        self.block_layers = block_layers
        self.classes = classes

        self.kernel_initializer = kernel_initializer='he_uniform'
        self.kernel_regularizer = kernel_regularizer=l2(0.0001)

        self.create()

    def DenseBlock(self, filters):
        def helper(input):
            output = BatchNormalization(gamma_regularizer=l2(0.0001),
                                        beta_regularizer=l2(0.0001))(input)
            output = Activation('relu')(output)
            output = Conv2D(filters, kernel_size=(3, 3), padding='same',
                            kernel_initializer=self.kernel_initializer)(output)
            return output

        return helper

    def Bottleneck(self, filters):
        def helper(input):
            output = BatchNormalization(gamma_regularizer=l2(0.0001),
                                        beta_regularizer=l2(0.0001))(input)
            output = Activation('relu')(output)
            output = Conv2D(filters, kernel_size=(1, 1),
                            kernel_initializer=self.kernel_initializer)(output)
            return output

        return helper

    def AtrousBlock(self, filters):
        def helper(input):
            output = BatchNormalization(gamma_regularizer=l2(0.0001),
                                        beta_regularizer=l2(0.0001))(input)
            output = Activation('relu')(output)
            output = AtrousConvolution2D(filters, nb_row=3, nb_col=3,
                                         atrous_rate=(2,2), border_mode='same',
                                         W_regularizer=l2(0.0001), b_regularizer=l2(0.0001))(output)
            output = Dropout(0.2)(output)
            return output

        return helper

    def create(self):
        input = Input(shape=self.input_shape)

        # We perform a first convolution. All the features maps will be stored in the Tiramisu.
        # first_conv_filters is 48 in the one hundred layers tiramisu.
        tiramisu = Conv2D(self.first_conv_filters, kernel_size=(3, 3), padding='same', 
                          input_shape=self.input_shape,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer)(input)

        for i in range(self.pools):
            for j in range(self.block_layers[i]):
                l = self.DenseBlock(self.growth_rate)(tiramisu)
                tiramisu = Concatenate()([tiramisu, l])
            new_tiramisu = self.AtrousBlock(self.growth_rate)(tiramisu)
            old_tiramisu = self.Bottleneck(self.growth_rate)(tiramisu)
            # tiramisu = Concatenate()([tiramisu, self.AtrousBlock(self.growth_rate)(tiramisu)])
            tiramisu = Concatenate()([old_tiramisu, new_tiramisu])

        tiramisu = Conv2D(self.classes, kernel_size=(1, 1), padding='same',
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer)(tiramisu)
        tiramisu = Reshape((self.input_shape[0] * self.input_shape[1], self.classes))(tiramisu)
        tiramisu = Activation('softmax')(tiramisu)

        self.model = Model(inputs=input, outputs=tiramisu)

        self.model.summary()
        with open('tiramisu_fc_dense_model.json', 'w') as outfile:
            outfile.write(json.dumps(json.loads(self.model.to_json()), indent=3))

Tiramisu()
