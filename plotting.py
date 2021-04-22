import random
import matplotlib.pyplot as plt
from loss_functions import psnr_and_ssim_loss, supervised_loss


class Plots:
    
    def __init__(self, MODEL="3DRLDSRN"):
        self.epochs           = []
        self.train_gen_g_loss = []
        self.train_psnr       = []
        self.train_ssim       = []
        self.valid_gen_g_loss = []
        self.valid_psnr       = []
        self.valid_ssim       = []
        self.model = MODEL
        self.plots      = [[self.train_gen_g_loss, self.valid_gen_g_loss],
                           [self.train_psnr, self.valid_psnr], 
                           [self.train_ssim, self.valid_ssim],]
        self.labels     = [['Training Genreator G Error', 'Validation Genreator G Error'],
                           ['Training PSNR', 'Validation PSNR'], 
                           ['Training SSIM', 'Validation SSIM'],]
        self.file_names = ['losses_plot',
                           'psnr_evaluation', 
                           'ssim_evaluation',]
        if self.model=="WGANGP-3DRLDSRN":
            self.gen_g_adv_loss = []
            self.disc_y_adv_loss = []
            self.plots.append([self.gen_g_adv_loss, self.disc_y_adv_loss])
            self.labels.append(['Generator G Adversarial Loss', 'Discriminator Y Adversarial Loss'])
            self.file_names.append('adversarial_g_y_plots')

    def append_plot_data(self, epoch, t_gen_g_l, t_psnr, t_ssim, v_gen_g_l, v_psnr, v_ssim, gen_g_adv_l=0, disc_y_adv_l=0):
        self.epochs.append(epoch)
        self.train_gen_g_loss.append(t_gen_g_l)
        self.train_psnr.append(t_psnr)
        self.train_ssim.append(t_ssim)
        self.valid_gen_g_loss.append(v_gen_g_l)
        self.valid_psnr.append(v_psnr)
        self.valid_ssim.append(v_ssim)
        if self.model=="WGANGP-3DRLDSRN":
            self.gen_g_adv_loss.append(gen_g_adv_l)
            self.disc_y_adv_loss.append(disc_y_adv_l)

    def plot_evaluations(self, EPOCH_START=""):
        for plot, label, file_name in zip(self.plots, self.labels, self.file_names):
            plt.figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
            plt.plot(self.epochs, plot[0], label =label[0])
            plt.plot(self.epochs, plot[1], label =label[1])
            plt.legend()
            # plt.show()
            plt.savefig('generated_plots/{}_{}.png'.format(file_name, EPOCH_START), bbox_inches='tight')


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
