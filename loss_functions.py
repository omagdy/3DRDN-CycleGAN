import tensorflow as tf

mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()

def supervised_loss(real, fake):
#     loss = mse(real, fake) # L2 Loss
    loss = mae(real, fake)  # L1 Loss
    return loss

def psnr_and_ssim_loss(im1, im2):
    psnr = tf.image.psnr(im1, im2, 1).numpy()
    ssim = tf.image.ssim(im1, im2, 1).numpy()
    psnr = round(psnr,3)
    ssim = round(ssim,3)
    return psnr,ssim
