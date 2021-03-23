import tensorflow as tf

filter_size = 3
w_init = tf.keras.initializers.GlorotUniform()

def dense_unit(no_of_filters, f_size, UTILIZE_BIAS, name=None):
    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())
#     result.add(Swish(dtype='float64', trainable=False))    
    result.add(tf.keras.layers.Conv3D(no_of_filters, f_size, kernel_initializer=w_init,
                                      use_bias = UTILIZE_BIAS, padding='same', dtype='float64'))
    return result


def dense_block(inputs, K, NO_OF_UNITS_PER_BLOCK, UTILIZE_BIAS):

    bottleneck_filter_size    = 1
    bottleneck_filter_number  = 4*K

    dense_units_output = []
    dense_units_output.append(inputs)
    dense_unit_output_0 = dense_unit(K, filter_size, UTILIZE_BIAS)(inputs)
    dense_units_output.append(dense_unit_output_0)
    dense_unit_output = tf.keras.layers.Concatenate(dtype='float64')(dense_units_output)
    for i in range(NO_OF_UNITS_PER_BLOCK-1):
        if dense_unit_output.shape[-1] > bottleneck_filter_number:
            bottleneck_dense_unit_output = dense_unit(bottleneck_filter_number, bottleneck_filter_size, UTILIZE_BIAS)(dense_unit_output)
            dense_unit_output = dense_unit(K, filter_size, UTILIZE_BIAS)(bottleneck_dense_unit_output)
        else:
            dense_unit_output = dense_unit(K, filter_size, UTILIZE_BIAS)(dense_unit_output)
        dense_units_output.append(dense_unit_output)
        dense_unit_output = tf.keras.layers.Concatenate(dtype='float64')(dense_units_output)
    return dense_unit_output


def Generator(PATCH_SIZE=64, NO_OF_DENSE_BLOCKS=10, K=8, NO_OF_UNITS_PER_BLOCK=4, UTILIZE_BIAS=True):

    compressor_filter_size    = 1
    first_conv_filter_number  = 2*K
    compressor_filter_number  = 2*K

    dense_blocks_output = []
    
    inputs = tf.keras.layers.Input(shape=[PATCH_SIZE,PATCH_SIZE,PATCH_SIZE,1], dtype='float64')
    conv1 = tf.keras.layers.Conv3D(first_conv_filter_number, filter_size, kernel_initializer=w_init, 
                                   use_bias = UTILIZE_BIAS, padding='same', dtype='float64')(inputs)
    dense_blocks_output.append(conv1)
    dense_block_output_0 = dense_block(conv1, K, NO_OF_UNITS_PER_BLOCK, UTILIZE_BIAS)
    dense_blocks_output.append(dense_block_output_0)
    dense_block_output = tf.keras.layers.Concatenate(dtype='float64')(dense_blocks_output)
    compressor_output = tf.keras.layers.Conv3D(compressor_filter_number, compressor_filter_size, kernel_initializer=w_init, 
                                   use_bias = UTILIZE_BIAS, padding='same', dtype='float64')(dense_block_output)
    
    for i in range(NO_OF_DENSE_BLOCKS-1):
        dense_block_output = dense_block(compressor_output, K, NO_OF_UNITS_PER_BLOCK, UTILIZE_BIAS)
        dense_blocks_output.append(dense_block_output)
        dense_block_output = tf.keras.layers.Concatenate(dtype='float64')(dense_blocks_output)
        compressor_output = tf.keras.layers.Conv3D(compressor_filter_number, compressor_filter_size, kernel_initializer=w_init, 
                                       use_bias = UTILIZE_BIAS, padding='same', dtype='float64')(dense_block_output)
        
    reconstruction_output = tf.keras.layers.Conv3D(1, 1, kernel_initializer=w_init, 
                                   use_bias = UTILIZE_BIAS, padding='same', dtype='float64')(compressor_output)
    
    final_output =  reconstruction_output + inputs
            
    return tf.keras.Model(inputs=inputs, outputs=final_output)