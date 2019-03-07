# pytorch-kmeans-init

This repo provides a data-dependent initialization method using k-means clustering on pytorch. The implementation was based on Adam Coates and Andrew Ng, Learning feature representations with k-means (https://www-cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf). 

`kmeans_init.py` provides the method `kmeans_init()` that can be used to replace other initialization methods like `torch.nn.init.normal` and `torch.nn.init.xavier`. The difference is that k-means initialization requires data, usually a minibatch of images, for initialization. 

`kmeans_init()` takes in 3 arguments. First is the minibatch of images that is necessary for initialization. Usually a few images, like a minibatch of 8 images, are sufficient. Second option is `num_iter`, which sets the number of k-means clustering iteration. Lastly, `use_whitening` decies whether to whiten the data before running k-means. According to Coates and Ng, whitening should improve the quality of k-means clustering, but I found that it makes the initialization pretty slow and someitmes unstable for the deep layers of large networks like VGG16. 

``
python demo.py
``

should run kmeans initialization on a single Conv2d layer and output the visualization of weights to 000.png. 

The image below is a sample visualization of the initialized weights of a single layer network initialized using 8 images in `images/`. 

![Result with without whitening](result_without_whitening.png)
