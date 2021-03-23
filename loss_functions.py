import tensorflow as tf

mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()

def supervised_loss(real, fake):
#     loss = mse(real, fake) # L2 Loss
    loss = mae(real, fake)  # L1 Loss
    return loss