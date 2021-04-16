import tensorflow as tf


class G_Model:

    WEIGHTS_INIT_DICT    = {"HeUniform"     : tf.keras.initializers.HeUniform(),
                            "HeNormal"      : tf.keras.initializers.HeNormal(), 
                            "GlorotUniform" : tf.keras.initializers.GlorotUniform(),
                            "GlorotNormal"  : tf.keras.initializers.GlorotNormal(),}

    ACTIVATION_FUNC_DICT = {"SWISH"         : tf.keras.layers.Activation(tf.nn.silu),
                            "LEAKYRELU"     : tf.keras.layers.LeakyReLU(),
                            "RELU"          : tf.keras.layers.ReLU(),
                            "ELU"           : tf.keras.layers.ELU(),}

    def __init__(self, PATCH_SIZE=40, NO_OF_DENSE_BLOCKS=3, K=24, NO_OF_UNITS_PER_BLOCK=4, UTILIZE_BIAS=True, WEIGHTS_INIT="HeUniform", ACTIVATION_FUNC="LEAKYRELU"):
        self.K                     = K
        self.PATCH_SIZE            = PATCH_SIZE
        self.NO_OF_DENSE_BLOCKS    = NO_OF_DENSE_BLOCKS
        self.NO_OF_UNITS_PER_BLOCK = NO_OF_UNITS_PER_BLOCK
        self.UTILIZE_BIAS          = UTILIZE_BIAS
        self.WEIGHTS_INIT          = self.WEIGHTS_INIT_DICT[WEIGHTS_INIT]
        self.ACTIVATION_FUNC       = self.ACTIVATION_FUNC_DICT[ACTIVATION_FUNC]

        self.filter_size               = 3
        self.bottleneck_filter_size    = 1
        self.bottleneck_filter_number  = 4*self.K
        self.compressor_filter_size    = 1
        self.first_conv_filter_number  = 2*K
        self.compressor_filter_number  = 2*K


    def dense_unit(self, no_of_filters, f_size, name=None):
        result = tf.keras.Sequential(name=name)
        result.add(tf.keras.layers.BatchNormalization())
        result.add(self.ACTIVATION_FUNC)    
        result.add(tf.keras.layers.Conv3D(no_of_filters, f_size, kernel_initializer=self.WEIGHTS_INIT,
                                          use_bias = self.UTILIZE_BIAS, padding='same', dtype='float64'))
        return result


    def dense_block(self, inputs):
        dense_units_output = []
        dense_units_output.append(inputs)
        dense_unit_output_0 = self.dense_unit(self.K, self.filter_size)(inputs)
        dense_units_output.append(dense_unit_output_0)
        dense_unit_output = tf.keras.layers.Concatenate(dtype='float64')(dense_units_output)
        for i in range(self.NO_OF_UNITS_PER_BLOCK-1):
            if dense_unit_output.shape[-1] > self.bottleneck_filter_number:
                bottleneck_dense_unit_output = self.dense_unit(self.bottleneck_filter_number, self.bottleneck_filter_size)(dense_unit_output)
                dense_unit_output = self.dense_unit(self.K, self.filter_size)(bottleneck_dense_unit_output)
            else:
                dense_unit_output = self.dense_unit(self.K, self.filter_size)(dense_unit_output)
            dense_units_output.append(dense_unit_output)
            dense_unit_output = tf.keras.layers.Concatenate(dtype='float64')(dense_units_output)
        return dense_unit_output


    def create_generator(self):
        dense_blocks_output = []
        inputs = tf.keras.layers.Input(shape=[self.PATCH_SIZE,self.PATCH_SIZE,self.PATCH_SIZE,1], dtype='float64')
        conv1 = tf.keras.layers.Conv3D(self.first_conv_filter_number, self.filter_size, kernel_initializer=self.WEIGHTS_INIT, 
                                       use_bias=self.UTILIZE_BIAS, padding='same', dtype='float64')(inputs)
        dense_blocks_output.append(conv1)
        dense_block_output_0 = self.dense_block(conv1)
        dense_blocks_output.append(dense_block_output_0)
        dense_block_output = tf.keras.layers.Concatenate(dtype='float64')(dense_blocks_output)
        compressor_output = tf.keras.layers.Conv3D(self.compressor_filter_number, self.compressor_filter_size, kernel_initializer=self.WEIGHTS_INIT, 
                                       use_bias=self.UTILIZE_BIAS, padding='same', dtype='float64')(dense_block_output)
        
        for i in range(self.NO_OF_DENSE_BLOCKS-2):
            dense_block_output = self.dense_block(compressor_output)
            dense_blocks_output.append(dense_block_output)
            dense_block_output = tf.keras.layers.Concatenate(dtype='float64')(dense_blocks_output)
            compressor_output = tf.keras.layers.Conv3D(self.compressor_filter_number, self.compressor_filter_size, kernel_initializer=self.WEIGHTS_INIT, 
                                           use_bias=self.UTILIZE_BIAS, padding='same', dtype='float64')(dense_block_output)
        dense_block_output = self.dense_block(compressor_output)
        dense_blocks_output.append(dense_block_output)
        dense_block_output = tf.keras.layers.Concatenate(dtype='float64')(dense_blocks_output)
        reconstruction_output = tf.keras.layers.Conv3D(1, 1, kernel_initializer=self.WEIGHTS_INIT, 
                                       use_bias=self.UTILIZE_BIAS, padding='same', dtype='float64')(dense_block_output)
        final_output =  reconstruction_output + inputs
        return tf.keras.Model(inputs=inputs, outputs=final_output)
