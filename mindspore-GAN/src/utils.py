import os
import numpy as np
from mindspore import Tensor
import matplotlib.pyplot as plt
import matplotlib.animation as animation

img_path = "images"
iter_path = 'iter.png'
# os.makedirs(img_path, exist_ok=True)

midimg_path = 'midimages'
os.makedirs(midimg_path, exist_ok=True)
def save_imgs(gen_imgs1, idx): # 保存生成的test图像
    for i3 in range(gen_imgs1.shape[0]):
        plt.subplot(5, 5, i3 + 1)
        plt.imshow(gen_imgs1[i3, 0, :, :]/2+0.5, cmap="gray")
        plt.axis("off")
    plt.savefig(midimg_path+"/{}.png".format(idx))


def paintIters(D_losses, G_losses):
    plt.figure(figsize=(10, 10))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G", color='blue')
    plt.plot(D_losses, label="D", color='orange')
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(iter_path)
    plt.show()

def write2file(loss_d, loss_g, title):
    f=open("%s.txt"%(title), "w")
    for i in range(len(loss_d)):
        f.write(str(loss_d[i]) + ' ')
        f.write(str(loss_g[i]) + '\n')
    f.close()

def showGif(image_list):
    # show_list = []
    fig = plt.figure(figsize=(5, 5), dpi=120)
    for epoch in range(len(image_list)):
        images = []
        for i in range(5):
            row = np.concatenate((image_list[epoch][i * 5:(i + 1) * 5]), axis=1)
            images.append(row)
        img = np.clip(np.concatenate((images[:]), axis=0), 0, 1)
        plt.axis("off")
        # show_list.append([plt.imshow(img)])
    plt.savefig(img_path+"/{}.png".format(idx))
    # ani = animation.ArtistAnimation(fig, show_list, interval=1000, repeat_delay=1000, blit=True)
    # ani.save('./gan.gif', writer='pillow', fps=1)

# D_losses = Tensor([0,1,2,3,4])
# G_losses = Tensor([2.2, 4.1, 3.9, 5.1, 3.65])
# paintIters(D_losses.asnumpy(), G_losses.asnumpy())