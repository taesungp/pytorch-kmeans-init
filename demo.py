import os
import torch
import numpy as np
import kmeans_init
import one_layer
import glob
import torchvision
from PIL import Image


def load_images(opt):
    extensions = ['JPEG', 'jpg', 'png', 'PNG']
    images = []
    for ext in extensions:        
        images += list(glob.glob(os.path.join(opt.source, "*." + ext)))

    imgs = []
    for path in images:
        img = Image.open(path).convert('RGB').resize((opt.image_size, opt.image_size))
        img = np.array(img).astype(np.float32)
        img = np.expand_dims(img, 0)
        imgs.append(img)
        if len(imgs) > opt.batchsize:
            break

    img = np.concatenate(imgs, axis=0)        
    img = img.transpose((0, 3, 1, 2))
    img = img * 2 /255.0 - 1

    return torch.from_numpy(img)

def save_images(images):
    for i, img in enumerate(images):
        savepath = "%03d.png" % i
        h, w, c = img.shape
        img = (img.squeeze() * 255.0).astype(np.uint8)
        img = Image.fromarray(img).resize((w * 10, h * 10))
        img.save(savepath)
        print("The weights were saved at %s" % savepath)
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='./images')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--kmeans_num_iter', type=int, default=3)
    parser.add_argument('--kmeans_use_whitening', action='store_true')
    parser.add_argument('--use_gpu', action='store_true')
    opt = parser.parse_args()

    net = one_layer.OneLayer(3, 6, 9)
    #net = torchvision.models.vgg16() # uncomment to try bigger model

    images = load_images(opt)

    if opt.use_gpu:
        net.cuda()
        images = images.cuda()

    print('Running k-means initialization')
    kmeans_init.kmeans_init(net, images, opt.kmeans_num_iter, opt.kmeans_use_whitening)

    print('visualizing the weights to images')
    visuals = kmeans_init.visualize_weights(net)    
    
    save_images(visuals)
    

    
