from torchvision.utils import make_grid , save_image
import numpy as np
import torch
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show_and_save(file_name,img):
    npimg = np.transpose(img.numpy(),(1,2,0))
    f = "./%s.png" % file_name
    fig = plt.figure(dpi=200)
    fig.suptitle(file_name, fontsize=14, fontweight='bold')
    plt.imshow(npimg)
    plt.imsave(f,npimg)
def plot_loss(loss_list, file_name="loss_plot.png"):
    plt.figure(figsize=(10,5))
    plt.title("Loss During Training")
    plt.plot(loss_list, label="Loss")
    
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(file_name, dpi=200)  # Save the plot as a file
    plt.show()
