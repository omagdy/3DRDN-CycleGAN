import tensorflow as tf

mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()

def supervised_loss(real, fake):
#     loss = mse(real, fake) # L2 Loss
    loss = mae(real, fake)  # L1 Loss
    return loss

def psnr_and_ssim_loss(im1, im2, PATCH_SIZE):
    psnr = tf.image.psnr(im1.reshape(PATCH_SIZE,PATCH_SIZE,PATCH_SIZE), im2.reshape(PATCH_SIZE,PATCH_SIZE,PATCH_SIZE), 1)
    ssim = tf.image.ssim(im1.reshape(PATCH_SIZE,PATCH_SIZE,PATCH_SIZE), im2.reshape(PATCH_SIZE,PATCH_SIZE,PATCH_SIZE), 1)
    psnr = round(psnr.numpy(),3)
    ssim = round(ssim.numpy(),3)
    return psnr,ssim
