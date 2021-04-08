import random
import matplotlib.pyplot as plt
from loss_functions import psnr_and_ssim_loss, supervised_loss


def plot_random_slice(model_patch, r1, r2):
    if r1 == 1:
        patch_slice = model_patch[r2,:,:]
    elif r1 == 2:
        patch_slice = model_patch[:,r2,:]
    else:
        patch_slice = model_patch[:,:,r2]
    plt.imshow(patch_slice, cmap="gray", interpolation="spline36")
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
        psnr,ssim = psnr_and_ssim_loss(display_list[i],tar)
        error = round(supervised_loss(tar, display_list[i]).numpy(), 3)
#         text = 'PSNR={}\nSSIM={}'.format(psnr,ssim)
        text = 'Error='+str(error)+'\nPSNR='+str(psnr)+'\nSSIM='+str(ssim)
        plt.text(0.5,-0.3, text, size=20, ha="center", 
         transform=ax.transAxes)
        plot_random_slice(display_list[i], r1, r2)
    # plt.show()
    plt.savefig('generated_plots/{}.png'.format(figure_name), bbox_inches='tight')


def plot_evaluations(epochs_plot, training_psnr, validation_psnr, training_ssim, validation_ssim,
                     training_generator_g_error, validation_generator_g_error, EPOCH_START=""):
    plots      = [[training_generator_g_error, validation_generator_g_error],
                  [training_psnr, validation_psnr], 
                  [training_ssim, validation_ssim],]
    labels     = [['Training Genreator G Error', 'Validation Genreator G Error'],
                  ['Training PSNR', 'Validation PSNR'], 
                  ['Training SSIM', 'Validation SSIM'],]
    file_names = ['losses_plot',
                  'psnr_evaluation', 
                  'ssim_evaluation',]
    for plot, label, file_name in zip(plots, labels, file_names):
        plt.figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(epochs_plot, plot[0], label =label[0])
        plt.plot(epochs_plot, plot[1], label =label[1])
        plt.legend()
        # plt.show()
        plt.savefig('generated_plots/{}_{}.png'.format(file_name, EPOCH_START), bbox_inches='tight')
