import tensorflow as tf
from logger import log
from loss_functions import supervised_loss, generator_loss, discriminator_loss


class Generator:
    
    WEIGHTS_INIT_DICT    = {"HeUniform"     : tf.keras.initializers.HeUniform(),
                            "HeNormal"      : tf.keras.initializers.HeNormal(), 
                            "GlorotUniform" : tf.keras.initializers.GlorotUniform(),
                            "GlorotNormal"  : tf.keras.initializers.GlorotNormal(),}

    ACTIVATION_FUNC_DICT = {"SWISH"         : tf.keras.layers.Activation(tf.nn.silu),
                            "LEAKYRELU"     : tf.keras.layers.LeakyReLU(alpha=0.2),
                            "RELU"          : tf.keras.layers.ReLU(),
                            "ELU"           : tf.keras.layers.ELU(),}
    
    def __init__(self, PATCH_SIZE=40, NO_OF_DENSE_BLOCKS=2, K=12, NO_OF_UNITS_PER_BLOCK=3,
                 UTILIZE_BIAS=True, WEIGHTS_INIT="HeUniform", ACTIVATION_FUNC="LEAKYRELU"):
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
        
        for i in range(self.NO_OF_DENSE_BLOCKS-1):
            dense_block_output = self.dense_block(compressor_output)
            dense_blocks_output.append(dense_block_output)
            dense_block_output = tf.keras.layers.Concatenate(dtype='float64')(dense_blocks_output)
            compressor_output = tf.keras.layers.Conv3D(self.compressor_filter_number, self.compressor_filter_size, kernel_initializer=self.WEIGHTS_INIT, 
                                           use_bias=self.UTILIZE_BIAS, padding='same', dtype='float64')(dense_block_output)
        reconstruction_output = tf.keras.layers.Conv3D(1, 1, kernel_initializer=self.WEIGHTS_INIT, 
                                       use_bias=self.UTILIZE_BIAS, padding='same', dtype='float64')(compressor_output)
        final_output =  reconstruction_output + inputs
        return tf.keras.Model(inputs=inputs, outputs=final_output)


class Discriminator:
    
    def __init__(self, PATCH_SIZE=40, UTILIZE_BIAS=True):
        self.k                     = 64
        self.filter_size           = 3
        self.PATCH_SIZE            = PATCH_SIZE
        self.UTILIZE_BIAS          = UTILIZE_BIAS
        self.WEIGHTS_INIT          = tf.keras.initializers.HeUniform()
        
    def conv_lay_norm_lrelu(self, filter_number, strides):
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv3D(filter_number, self.filter_size, kernel_initializer=self.WEIGHTS_INIT, 
                                          strides=strides, use_bias=self.UTILIZE_BIAS, padding='same',
                                          dtype='float64'))
        result.add(tf.keras.layers.LayerNormalization())
        result.add(tf.keras.layers.LeakyReLU(alpha=0.2))    
        return result

    def create_discriminator(self):
        inputs = tf.keras.layers.Input(shape=[self.PATCH_SIZE,self.PATCH_SIZE,self.PATCH_SIZE,1], dtype='float64')
        conv1 = tf.keras.layers.Conv3D(self.k, self.filter_size, kernel_initializer=self.WEIGHTS_INIT,  
                                       use_bias=self.UTILIZE_BIAS, padding='same', dtype='float64')(inputs)
        lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv1)
        conv_lay_norm_lrelu1 = self.conv_lay_norm_lrelu(self.k, 2)(lrelu1)
        conv_lay_norm_lrelu2 = self.conv_lay_norm_lrelu(2*self.k, 1)(conv_lay_norm_lrelu1)
        conv_lay_norm_lrelu3 = self.conv_lay_norm_lrelu(2*self.k, 2)(conv_lay_norm_lrelu2)
        conv_lay_norm_lrelu4 = self.conv_lay_norm_lrelu(4*self.k, 1)(conv_lay_norm_lrelu3)
        conv_lay_norm_lrelu5 = self.conv_lay_norm_lrelu(4*self.k, 2)(conv_lay_norm_lrelu4)
        conv_lay_norm_lrelu6 = self.conv_lay_norm_lrelu(8*self.k, 1)(conv_lay_norm_lrelu5)
        conv_lay_norm_lrelu7 = self.conv_lay_norm_lrelu(8*self.k, 2)(conv_lay_norm_lrelu6)
        flatten = tf.keras.layers.Flatten()(conv_lay_norm_lrelu7)
        dense1 = tf.keras.layers.Dense(1024)(flatten)
        lrelu2 = tf.keras.layers.LeakyReLU(alpha=0.2)(dense1)
        output = tf.keras.layers.Dense(1)(lrelu2)
        return tf.keras.Model(inputs=inputs, outputs=output)


class Model3DRLDSRN:

    def __init__(self, PATCH_SIZE=40, BATCH_SIZE=6, LR_G=1e-4, LR_D=1e-4, LAMBDA_ADV=0.1,
     LAMBDA_GRD_PEN=10, MODEL="3DRLDSRN"):
        assert(MODEL in ["3DRLDSRN", "WGANGP-3DRLDSRN", "CYCLE-WGANGP-3DRLDSRN"])
        self.MODEL                 = MODEL
        gen                        = Generator(PATCH_SIZE=PATCH_SIZE)
        self.generator_g           = gen.create_generator()
        self.generator_g_optimizer = tf.keras.optimizers.Adam(LR_G)
        self.load_generator()
        if MODEL=="WGANGP-3DRLDSRN":
            disc                           = Discriminator(PATCH_SIZE=PATCH_SIZE)
            self.discriminator_y           = disc.create_discriminator()
            self.discriminator_y_optimizer = tf.keras.optimizers.Adam(LR_D)
            self.lambda_adv                = LAMBDA_ADV
            self.lambda_grad_pen           = LAMBDA_GRD_PEN
            self.alpha = tf.random.uniform([BATCH_SIZE, 1, 1, 1, 1], 0, 1, dtype='float64')
            self.load_discriminator()
        elif MODEL=="CYCLE-WGANGP-3DRLDSRN":
            disc                           = Discriminator(PATCH_SIZE=PATCH_SIZE)
            self.generator_f               = gen.create_generator()
            self.generator_f_optimizer     = tf.keras.optimizers.Adam(LR_G)
            self.discriminator_y           = disc.create_discriminator()
            self.discriminator_x           = disc.create_discriminator()
            self.discriminator_y_optimizer = tf.keras.optimizers.Adam(LR_D)
            self.discriminator_x_optimizer = tf.keras.optimizers.Adam(LR_D)
            self.lambda_adv                = LAMBDA_ADV
            self.lambda_grad_pen           = LAMBDA_GRD_PEN
            self.alpha = tf.random.uniform([BATCH_SIZE, 1, 1, 1, 1], 0, 1, dtype='float64')
            self.load_discriminator()
    
    def load_generator(self, tensor_path = "tensor_checkpoints/generator_checkpoint/"):
        ckpt = tf.train.Checkpoint(generator_g=self.generator_g, generator_g_optimizer=self.generator_g_optimizer,)
        self.gen_ckpt_manager = tf.train.CheckpointManager(ckpt, tensor_path, max_to_keep=3)
        if self.gen_ckpt_manager.latest_checkpoint:
            ckpt.restore(gen_ckpt_manager.latest_checkpoint)
            log('Generator latest checkpoint restored!!')
        else:
            log("No checkpoint found for generator! Staring from scratch!")

    def load_discriminator(self, tensor_path = "tensor_checkpoints/discriminator_checkpoint/"):
        ckpt = tf.train.Checkpoint(discriminator_y=self.discriminator_y, 
            discriminator_y_optimizer=self.discriminator_y_optimizer,)
        self.disc_ckpt_manager = tf.train.CheckpointManager(ckpt, tensor_path, max_to_keep=3)
        if self.disc_ckpt_manager.latest_checkpoint:
            ckpt.restore(disc_ckpt_manager.latest_checkpoint)
            log('Discriminator latest checkpoint restored!!')
        else:
            log("No checkpoint found for discriminator! Staring from scratch!")

    def save_models(self, epoch, gen=True, disc=False):
        if gen:
            ckpt_save_path = self.gen_ckpt_manager.save()
            log('Saving generator checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))
        if disc:
            ckpt_save_path = self.disc_ckpt_manager.save()
            log('Saving discriminator checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))
        
    @tf.function
    def supervised_train_step(self, real_x, real_y):
        with tf.GradientTape(persistent=True) as tape:
            fake_y             = self.generator_g(real_x, training=True)
            gen_g_super_loss   = supervised_loss(real_y, fake_y)
        gradients_of_generator = tape.gradient(gen_g_super_loss, self.generator_g.trainable_variables)
        self.generator_g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator_g.trainable_variables))
        return gen_g_super_loss

    @tf.function
    def gen_g_train_step(self, real_x, real_y):
        with tf.GradientTape(persistent=True) as tape:
            fake_y           = self.generator_g(real_x, training=True)
            disc_fake_y      = self.discriminator_y(fake_y, training=True)
            gen_g_super_loss = supervised_loss(real_y, fake_y)
            gen_g_adv_loss   = generator_loss(disc_fake_y)
            total_gen_g_loss = gen_g_super_loss + self.lambda_adv*gen_g_adv_loss
        generator_g_gradient = tape.gradient(total_gen_g_loss, self.generator_g.trainable_variables)
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradient, self.generator_g.trainable_variables))
        return gen_g_adv_loss
    
    @tf.function
    def disc_y_train_step(self, real_x, real_y):
        with tf.GradientTape(persistent=True) as tape:
            fake_y         = self.generator_g(real_x, training=True)
            disc_real_y    = self.discriminator_y(real_y, training=True)
            disc_fake_y    = self.discriminator_y(fake_y, training=True)
            differences_y  = fake_y - real_y
            interpolates_y = real_y + (self.alpha*differences_y)
            with tf.GradientTape() as t:
                t.watch(interpolates_y)
                pred_y           = self.discriminator_y(interpolates_y, training=True)
            gradients_y          = t.gradient(pred_y, [interpolates_y])[0]
            slopes_y             = tf.sqrt(tf.reduce_sum(tf.square(gradients_y), axis=[1, 2, 3, 4]))
            gradient_penalty_y   = tf.reduce_mean((slopes_y-1.)**2)
            disc_y_loss          = discriminator_loss(disc_real_y, disc_fake_y)
            total_disc_y_loss    = disc_y_loss + self.lambda_grad_pen*gradient_penalty_y
        discriminator_y_gradient = tape.gradient(total_disc_y_loss, self.discriminator_y.trainable_variables)
        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradient, self.discriminator_y.trainable_variables))
        return total_disc_y_loss
