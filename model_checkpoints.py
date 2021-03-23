import tensorflow as tf
from logger import log
from model import Generator

def get_generator(PATCH_SIZE, LR_G, NO_OF_DENSE_BLOCKS, K, NO_OF_UNITS_PER_BLOCK, UTILIZE_BIAS):
    
    generator_g         = Generator(PATCH_SIZE, NO_OF_DENSE_BLOCKS, K, NO_OF_UNITS_PER_BLOCK, UTILIZE_BIAS)
    generator_optimizer = tf.keras.optimizers.Adam(LR_G)
    
    path = "tensor_checkpoints/"

    ckpt = tf.train.Checkpoint(generator_g=generator_g,
                               generator_optimizer=generator_optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=3)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        log('Latest checkpoint restored!!')
    else:
        log("No checkpoint found! Staring from scratch!")
                
    return generator_g, generator_optimizer, ckpt_manager

def save_generator(ckpt_manager, epoch):
    ckpt_save_path = ckpt_manager.save()
    log('Saving checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))
