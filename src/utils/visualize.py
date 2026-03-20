import matplotlib.pyplot as plt
import torch

def show_image_with_mask(image, mask=None):
    img = image.permute(1,2,0).cpu().numpy()
    plt.imshow(img)
    if mask is not None:
        mask = mask.squeeze().cpu().numpy()
        plt.imshow(mask, alpha=0.3, cmap='jet')
    plt.axis('off')
    plt.show()