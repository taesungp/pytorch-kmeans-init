import torch
import torchvision
import numpy as np
import torch.nn.functional as F


## Will run kmeans initialization at every Conv2d layers of kernel size 3 or more.
## |images|: images of dimension B x C x H x W
## |num_iter|: how many iterations of k-means to perform (default: 3)
## |use_whitening|: whitening the inputs before running k-means
## helps the quality of initialization by a lot, but it takes much longer.
def kmeans_init(net, input, num_iter=3, use_whitening=False):
    ## Step 1. Install hook for each Conv2d layer of net
    handles = []
    for m in net.modules():
        if m.__class__.__name__.find('Conv2d') != -1 \
           and m.kernel_size[0] > 1 and m.kernel_size[1] > 1:
            handle_prehook = m.register_forward_pre_hook(KMeansHook(num_iter, use_whitening))
            #handle_posthook = m.register_forward_hook(kmeans_posthook)
            handles += [handle_prehook]

    ## Step 2. Run forward pass of the images, in which the hooks will
    ## initialize the layer weights using k-means method
    ## Therefore, all the works are done in pre-hook and post-hook
    net(input)

    ## Step 3. Remove hooks
    [handle.remove() for handle in handles]
    return net


class KMeansHook(object):

    def __init__(self, num_iter, use_whitening):
        super().__init__()
        self.num_iter = num_iter
        self.use_whitening = use_whitening

    def __call__(self, module, inputs):
        ## We follow step 1, 2 and 3of Coates and Ng, 2012
        ## https://www-cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf
        
        x = inputs[0].detach()  # x is the same as x that appears in the paper
        if module.transposed:
            in_channels = module.in_channels
            out_channels = module.out_channels // module.groups
        else:
            in_channels = module.in_channels // module.groups
            out_channels = module.out_channels

        ## Preprocess
        ## Split the activations into patches
        x = x[:, :in_channels] # in case we use grouping, let's just use the first group
        patchsize = module.kernel_size
        chunks = x.split(patchsize[0], dim=2)
        ## drop the last if the last patch is too small
        if chunks[-1].size(2) < patchsize[0]: 
            chunks = chunks[:-1]
        x = torch.cat(chunks, dim=0)                

        chunks = x.split(patchsize[1], dim=3)
        ## drop the last if the last patch is too small
        if chunks[-1].size(3) < patchsize[1]: 
            chunks = chunks[:-1]
        x = torch.cat(chunks, dim=0)

        ## Step 1
        ## normalize the input
        x = self.normalize(x)
        
        ## Step 2
        ## Whiten the input
        ## Code from
        ## https://github.com/pytorch/vision/blob/cee28367704e4a6039b2c15de34248df8a77f837/test/test_transforms.py#L597
        x_flat = x.view(x.size(0), -1)
        if self.use_whitening:
            sigma = torch.mm(x_flat.t(), x_flat) / x_flat.size(0)
            U, ev, _ = torch.svd(sigma)
            zca_epsilon = 1e-10 # this value is much smaller than suggested in paper
            diag = torch.diag(1.0 / torch.sqrt(ev + zca_epsilon))
            principal_components = torch.mm(torch.mm(U, diag), U.t())
            xwhite = torch.mm(principal_components, x_flat.t()).t()
            x = xwhite.view(x.size())

        ## Step 3
        ## K-means
        ## x_flat is a m x k matrix (m observations of k-dim vectors)
        ## I think it differs from the papers notation in that it's a transpose. 
        ## D is out_channels x k.
        
        # randomly initailize D
        D = torch.randn(out_channels, x_flat.size(1), dtype=x.dtype, device=x.device)
        D = self.normalize(D)

        for i in range(self.num_iter):
            XD = torch.mm(x_flat, D.t()) ## XD is of m x out_channels
            maxes, _ = torch.max(XD, dim=1, keepdim=True)
            S = XD.masked_fill_(XD < maxes, 0)
            D += torch.mm(S.t(), x_flat)
            D = self.normalize(D)            

        D = D.view(out_channels, *x.size()[1:])

        # Haven't really tested using transposed convolution
        if module.transposed:
            D = D.transpose((0, 1))

        assert module.weight.size() == D.size()
        module.weight = torch.nn.Parameter(D)


    def normalize(self, z):
        # insert extra dimension at 1 so that instance norm
        # uses mean and var across all inputs
        z = F.instance_norm(z.unsqueeze(1))[:, 0]
        return z
        

## Will visualize activations at every conv2d(3)
def visualize_weights(net):

    conv2ds = get_conv_weights(net)

    weight_visuals = []

    for w in conv2ds:
        oc, ic, kh, kw = w.size()
        if kh < 2 or kw < 2:
            continue
        image = w.detach().cpu().numpy()
        max = image.max(axis=(2,3), keepdims=True)
        min = image.min(axis=(2,3), keepdims=True)
        image = (image - min) / (max - min)
        image = np.concatenate(image, axis=2)
        assert image.shape == (ic, kh, kw * oc)

        if ic != 3:
            image = np.concatenate(image, axis=0)
            image = image[np.newaxis, :]
            assert image.shape == (1, kh * ic, kw * oc)
            
        image = image.transpose((1, 2, 0))
        weight_visuals.append(image)

    return weight_visuals


def get_conv_weights(net):
    modules = []
    #for name, layer in net.named_children():
    for m in net.modules():
        if m.__class__.__name__.find('Conv2d') != -1:
            modules.append(m.weight)

    return modules
        

    
