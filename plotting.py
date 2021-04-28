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
    plt.savefig('generated_images/{}.png'.format(figure_name), bbox_inches='tight')
    plt.close('all')
