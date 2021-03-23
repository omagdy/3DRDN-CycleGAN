import random
import matplotlib.pyplot as plt

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
    display_list = [test_input, tar, prediction]
    title = ['Input Image', 'Ground Truth', 'Predicted Image'+epoch]
    if not r1:
        r1 = random.randrange(3)
    if not r2:
        r2 = random.randrange(PATCH_SIZE)
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plot_random_slice(display_list[i], r1, r2, PATCH_SIZE)
    # plt.show()
    plt.savefig('generated_plots/'+figure_name+'.png', bbox_inches='tight')


def plot_losses(epochs_plot, total_generator_g_error_plot):
    plt.figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(epochs_plot, total_generator_g_error_plot, label ='Genreator G loss')
    plt.legend()
    # plt.show()
    plt.savefig('generated_plots/losses_plot.png', bbox_inches='tight')
