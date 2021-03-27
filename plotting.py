import random
import tensorflow as tf
import matplotlib.pyplot as plt


def get_psnr_and_ssim(im1, im2, PATCH_SIZE):
    psnr = tf.image.psnr(im1.reshape(PATCH_SIZE,PATCH_SIZE,PATCH_SIZE), im2.reshape(PATCH_SIZE,PATCH_SIZE,PATCH_SIZE), 1)
    ssim = tf.image.ssim(im1.reshape(PATCH_SIZE,PATCH_SIZE,PATCH_SIZE), im2.reshape(PATCH_SIZE,PATCH_SIZE,PATCH_SIZE), 1)
    psnr = round(psnr.numpy(),3)
    ssim = round(ssim.numpy(),3)
    return psnr,ssim


def plot_random_slice(model_patch, r1, r2, PATCH_SIZE):
    model_patch = model_patch.reshape(PATCH_SIZE,PATCH_SIZE,PATCH_SIZE)
    if r1 == 1:
        plt.imshow(model_patch[r2,:,:])
    elif r1 == 2:
        plt.imshow(model_patch[:,r2,:])
    else:
        plt.imshow(model_patch[:,:,r2])
    plt.axis('off')
    
    
def generate_images(model, test_input, tar, PATCH_SIZE, figure_name, epoch="", r1=None, r2=None):
    prediction = model(test_input, training=False).numpy()
#     prediction = model(test_input, training=True).numpy()
    plt.figure(figsize=(15,15))
    display_list = [test_input, prediction, tar]
    title = ['Input Image', 'Predicted Image'+epoch, 'Ground Truth']
    if not r1:
        r1 = random.randrange(3)
    if not r2:
        r2 = random.randrange(PATCH_SIZE)
    for i in range(3):
        ax = plt.subplot(1, 3, i+1)
        plt.title(title[i])
        psnr,ssim = get_psnr_and_ssim(display_list[i],tar,PATCH_SIZE)
#         text = 'PSNR={}\nSSIM={}'.format(psnr,ssim)
        text = 'PSNR='+str(psnr)+'\nSSIM='+str(ssim)
        plt.text(0.5,-0.2, text, size=20, ha="center", 
         transform=ax.transAxes)
        plot_random_slice(display_list[i], r1, r2, PATCH_SIZE)
    # plt.show()
    plt.savefig('generated_plots/{}.png'.format(figure_name), bbox_inches='tight')
    return get_psnr_and_ssim(prediction,tar,PATCH_SIZE)


def plot_evaluations(epochs_plot, psnr_plot, ssim_plot, EPOCH_START=""):
    plt.figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(epochs_plot, psnr_plot, label ='PSNR')
    plt.legend()
    # plt.show()
    plt.savefig('generated_plots/psnr_evaluation_plot_{}.png'.format(EPOCH_START), bbox_inches='tight')    
    plt.figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(epochs_plot, ssim_plot, label ='SSIM')
    plt.legend()
    # plt.show()
    plt.savefig('generated_plots/ssim_evaluation_plot_{}.png'.format(EPOCH_START), bbox_inches='tight')  


def plot_losses(epochs_plot, total_generator_g_error_plot, EPOCH_START=""):
    plt.figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(epochs_plot, total_generator_g_error_plot, label ='Genreator G loss')
    plt.legend()
    # plt.show()
    plt.savefig('generated_plots/losses_plot_{}.png'.format(EPOCH_START), bbox_inches='tight')
