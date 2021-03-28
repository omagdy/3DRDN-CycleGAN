import random
import matplotlib.pyplot as plt
from loss_functions import psnr_and_ssim_loss 


def plot_random_slice(model_patch, r1, r2, PATCH_SIZE):
    model_patch = model_patch.reshape(PATCH_SIZE,PATCH_SIZE,PATCH_SIZE)
    if r1 == 1:
        plt.imshow(model_patch[r2,:,:])
    elif r1 == 2:
        plt.imshow(model_patch[:,r2,:])
    else:
        plt.imshow(model_patch[:,:,r2])
    plt.axis('off')
    
    
def generate_images(prediction, test_input, tar, PATCH_SIZE, figure_name, epoch="", r1=None, r2=None):
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
        psnr,ssim = psnr_and_ssim_loss(display_list[i],tar,PATCH_SIZE)
#         text = 'PSNR={}\nSSIM={}'.format(psnr,ssim)
        text = 'PSNR='+str(psnr)+'\nSSIM='+str(ssim)
        plt.text(0.5,-0.2, text, size=20, ha="center", 
         transform=ax.transAxes)
        plot_random_slice(display_list[i], r1, r2, PATCH_SIZE)
    # plt.show()
    plt.savefig('generated_plots/{}.png'.format(figure_name), bbox_inches='tight')
    return psnr_and_ssim_loss(prediction,tar,PATCH_SIZE)


def plot_evaluations(epochs_plot, training_psnr, validation_psnr, training_ssim, validation_ssim,
                     training_generator_g_error, validation_generator_g_error, EPOCH_START=""):
    plt.figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(epochs_plot, training_generator_g_error, label ='Training Genreator G Error')
    plt.plot(epochs_plot, validation_generator_g_error, label ='Validation Genreator G Error')
    plt.legend()
    # plt.show()
    plt.savefig('generated_plots/losses_plot_{}.png'.format(EPOCH_START), bbox_inches='tight')
    plt.figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(epochs_plot, training_psnr, label ='Training PSNR')
    plt.plot(epochs_plot, validation_psnr, label ='Validation PSNR')
    plt.legend()
    # plt.show()
    plt.savefig('generated_plots/psnr_evaluation_plot_{}.png'.format(EPOCH_START), bbox_inches='tight')    
    plt.figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(epochs_plot, training_ssim, label ='Training SSIM')
    plt.plot(epochs_plot, validation_ssim, label ='Training SSIM')
    plt.legend()
    # plt.show()
    plt.savefig('generated_plots/ssim_evaluation_plot_{}.png'.format(EPOCH_START), bbox_inches='tight')  
