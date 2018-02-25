from keras.backend import set_image_data_format, set_image_dim_ordering
import keras.backend as K

from keras.layers import Conv2D, Conv2DTranspose, Dropout, Input, Lambda, Concatenate
from keras.layers.core import Activation, Reshape
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

import json

set_image_data_format('channels_last')
set_image_dim_ordering('tf')

from keras.regularizers import l2
 
class Tiramisu():

    def __init__(self, classes=12,
                 input_shape=(224, 224, 6), origin_shape=(224, 224, 3), superpixel_shape=(224, 224, 3),
                 first_filters=48, growth_rate=16, pools=5,
                 block_layers=[4, 5, 7, 10, 12, 15],
                 atrous=[False, False, False, True, True],
                 bilinear=[False, False, True, True, True]):

        if type(block_layers) == list:
            if len(block_layers) == pools + 1:
                block_layers += block_layers[:-1][::-1]
            else:
                assert(len(block_layers) == pools * 2 + 1)
        elif type(block_layers) == int:
            block_layers = [block_layers] * (pools * 2 + 1)
        else:
            assert(False)

        if type(atrous) == list:
            assert(len(atrous) == pools)
        elif type(atrous) == bool:
            atrous = [atrous] * pools
        else:
            assert(False)

        if type(bilinear) == list:
            assert(len(bilinear) == pools)
        elif type(bilinear) == bool:
            bilinear = [bilinear] * pools
        else:
            assert(False)

        assert(input_shape[0] == origin_shape[0])
        assert(input_shape[1] == origin_shape[1])
        assert(input_shape[0] == superpixel_shape[0])
        assert(input_shape[1] == superpixel_shape[1])
        assert(input_shape[2] == origin_shape[2] + superpixel_shape[2])
        self.input_shape = input_shape
        self.origin_shape = origin_shape
        self.superpixel_shape = superpixel_shape

        self.n_classes = classes
        self.n_first_filters = first_filters
        self.growth_rate = growth_rate
        self.n_downsamples = pools
        self.n_layers_per_block = block_layers
        self.use_atrous = atrous
        self.use_bilinear_interplolation = bilinear

        self.layer = 0
        self.layers = sum(block_layers)

        self.kernel_initializer = 'he_uniform'
        self.regularizer = l2(0.0001)

        self.create()

    def Layer(self, filters):
        def get_survival(survival_end=0.5, mode='linear_decay'):
            self.layer += 1
            if self.layer == 1:
                return 1

            if mode == 'uniform':
                return survival_end
            elif mode == 'linear_decay':
                return 1 - (self.layer / self.layers) * (1 - survival_end)
            else:
                assert(False)

        def stochastic_survival(input, survival=1.0):
            survival = K.random_binomial((1,), p=survival)
            return K.in_test_phase(K.variable(survival, dtype='float32') * input, survival * input)

        def helper(input):
            output = BatchNormalization(gamma_regularizer=self.regularizer,
                                        beta_regularizer=self.regularizer)(input)
            output = Activation('relu')(output)
            output = Conv2D(filters, kernel_size=3, padding='same',
                            kernel_initializer=self.kernel_initializer)(output)
            # survival = get_survival()
            # output = Lambda(stochastic_survival, arguments={'survival': survival})(output)
            output = Dropout(0.2)(output)
            return output

        return helper

    def AtrousLayer(self, filters, rate=2):
        def helper(input):
            output = BatchNormalization(gamma_regularizer=self.regularizer,
                                        beta_regularizer=self.regularizer)(input)
            output = Activation('relu')(output)
            output = Conv2D(filters, kernel_size=3, padding='same',
                            dilation_rate=rate, kernel_regularizer=self.regularizer)(input)
            output = Dropout(0.2)(output)
            return output

        return helper

    def TransitionDown(self, filters):
        def helper(input):
            output = BatchNormalization(gamma_regularizer=self.regularizer,
                                        beta_regularizer=self.regularizer)(input)
            output = Activation('relu')(output)
            output = Conv2D(filters, kernel_size=1, padding='same',
                            kernel_initializer=self.kernel_initializer)(output)
            output = Dropout(0.2)(output)
            output = MaxPooling2D(pool_size=2, strides=2)(output)
            return output

        return helper

    def TransitionUp(self, filters):
        def helper(input, skip_connection):
            output = Conv2DTranspose(filters,  kernel_size=3,
                                     strides=2, padding='same',
                                     kernel_initializer=self.kernel_initializer)(input)
            return Concatenate()([output, skip_connection])

        return helper

    def BilinearUp(self, ratio):
        def bilinear_interplolation(image, size):
            import tensorflow as tf
            return tf.image.resize_images(image, size)

        def helper(input, skip_connection):
            size = K.int_shape(input)
            size = (size[1] * ratio, size[2] * ratio)
            output = Lambda(bilinear_interplolation, arguments={'size': size})(input)
            return Concatenate()([output, skip_connection])

        return helper

    def create(self):
        input = Input(shape=self.input_shape)
        def image_channels_slice(image, channel, before):
            if before:
                return image[:, :, :, :channel]
            else:
                return image[:, :, :, channel:]
        origin = Lambda(image_channels_slice, arguments={'channel': 3, 'before': True})(input)
        superpixel = Lambda(image_channels_slice, arguments={'channel': 3, 'before': False})(input)
        # origin = Lambda(lambda x: x[:, :, :, :self.origin_shape[2]], output_shape=self.origin_shape)(input)
        # superpixel = Lambda(lambda x: x[:, :, :, self.origin_shape[2]:], output_shape=self.superpixel_shape)(input)

        #####################
        # First Convolution #
        #####################

        tiramisu = Conv2D(self.n_first_filters, kernel_size=3, padding='same', 
                          input_shape=self.origin_shape,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.regularizer)(origin)

        #####################
        # Downsampling path #
        #####################

        m = 0
        skip_connections = []

        atrous_rate = 1

        for i in range(self.n_downsamples):
            if self.use_atrous[i]:
                atrous_rate *= 2
                for j in range(self.n_layers_per_block[i]):
                    l = self.AtrousLayer(self.growth_rate, rate=atrous_rate)(tiramisu)
                    tiramisu = Concatenate()([tiramisu, l])
                    m += self.growth_rate
                skip_connections.append(None)
                tiramisu = Concatenate()([tiramisu, self.AtrousLayer(self.growth_rate, rate=atrous_rate)(tiramisu)])
            else:
                for j in range(self.n_layers_per_block[i]):
                    l = self.Layer(self.growth_rate)(tiramisu)
                    tiramisu = Concatenate()([tiramisu, l])
                    m += self.growth_rate
                skip_connections.append(tiramisu)
                tiramisu = self.TransitionDown(m)(tiramisu)

        if skip_connections[0] == None:
            skip_connections[0] = superpixel
        else:
            skip_connections[0] = Concatenate()([skip_connections[0], superpixel])

        skip_connections = skip_connections[::-1]
        self.use_atrous = self.use_atrous[::-1]

        #####################
        #     Bottleneck    #
        #####################

        upsample_tiramisu = []
        for j in range(self.n_layers_per_block[self.n_downsamples]):
            l = self.Layer(self.growth_rate)(tiramisu)
            upsample_tiramisu.append(l)
            tiramisu = Concatenate()([tiramisu, l])

        #######################
        #   Upsampling path   #
        #######################

        for i in range(self.n_downsamples):
            if self.use_atrous[i]:
                assert(skip_connections[i] == None)
                assert(not self.use_bilinear_interplolation[i])
            else:
                upsample_tiramisu = Concatenate()(upsample_tiramisu)

                n_keep_filters = self.growth_rate * self.n_layers_per_block[self.n_downsamples + i]
                if self.use_bilinear_interplolation[i]:
                    tiramisu = self.BilinearUp(2)(upsample_tiramisu, skip_connections[i])
                else:
                    tiramisu = self.TransitionUp(n_keep_filters)(upsample_tiramisu, skip_connections[i])

                upsample_tiramisu = []
                for j in range(self.n_layers_per_block[self.n_downsamples + i + 1]):
                    l = self.Layer(self.growth_rate)(tiramisu)
                    upsample_tiramisu.append(l)
                    tiramisu = Concatenate()([tiramisu, l])

        #####################
        #      Softmax      #
        #####################

        tiramisu = Conv2D(self.n_classes, kernel_size=1, padding='same',
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.regularizer)(tiramisu)
        tiramisu = Reshape((self.input_shape[0] * self.input_shape[1], self.n_classes))(tiramisu)
        tiramisu = Activation('softmax')(tiramisu)

        self.model = Model(inputs=input, outputs=tiramisu)

        self.model.summary()
        with open('tiramisu_fc_dense_model.json', 'w') as outfile:
            outfile.write(json.dumps(json.loads(self.model.to_json()), indent=3))

Tiramisu()
