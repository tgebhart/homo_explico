import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import copy
import numpy as np

import torch
from torch.optim import SGD
from torchvision import models
from torchvision import transforms
from torch.autograd import Variable



class DeepDream():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, im):
        self.model = model
        self.model.eval()
        # Generate a random image
        self.created_image = im
        # Hook the layers to get result of the convolution
        # self.hook_layer()
        # Create the folder to export images if not exists
        # if not os.path.exists('../generated'):
        #     os.makedirs('../generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def dream(self, folder_name, subgraph_indices=[0], percentile=0, lr=0.01):
        # Process image and return variable
        # self.processed_image = preprocess_image(self.created_image, True)
        total_rotation = 0
        self.processed_image = Variable(self.created_image, requires_grad = True)
        # Define optimizer for the image
        # Earlier layers need higher learning rates to visualize whereas later layers need less
        optimizer = SGD([self.processed_image], lr=lr)
        output, hiddens = self.model(self.processed_image, hiddens=True)
        this_hiddens = [hiddens[i][0] for i in range(len(hiddens))]
        muls = self.model.compute_layer_mask(self.processed_image, this_hiddens, subgraph_indices=subgraph_indices, percentile=percentile)
        # self.processed_image = torch.randn(self.created_image.shape)
        for i in range(0, 3000+1):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = self.processed_image
            output, hiddens = self.model(x, hiddens=True)
            this_hiddens = [hiddens[i][0] for i in range(len(hiddens))]
            # muls = model.compute_layer_mask(x, hiddens, thru=2, percentile=0)
            s = torch.zeros((len(this_hiddens)))
            # l+1 because muls contains input activations as well
            for l in range(len(this_hiddens)):
                s[l] = torch.sum(this_hiddens[l]*muls[l+1].reshape(this_hiddens[l].shape).abs())
            # s = this_hiddens[-2][:10]
                # print(s[l].numpy())
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(s) + 0.001*(torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])))
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            # zoom = 1
            # if np.random.random() < 0.05:
            #     zoom = 1.1

            rotation = np.sign(np.random.random()-.5)*90
            total_rotation += rotation
            self.created_image = ndimage.rotate(recreate_image(copy.copy(self.processed_image)), rotation, axes=(2,1), reshape=False)

            # self.created_image = ndimage.zoom(ndimage.rotate(recreate_image(copy.copy(self.processed_image)), np.sign(np.random.random()-.5)*90, axes=(2,1), reshape=False), [1,zoom,zoom])
            # transform = transforms.Compose([transforms.ToPILImage(mode='RGB'), transforms.RandomAffine(degrees=180, scale=(1,1)), transforms.ToTensor()])
            # self.created_image = transform(recreate_image(copy.copy(self.processed_image)).transpose([1,2,0])).data.numpy()
            # self.created_image = transforms.functional.to_tensor(transforms.functional.affine(transforms.functional.to_pil_image(np.moveaxis(recreate_image(copy.copy(self.processed_image)),0,-1),mode='RGB'), angle=np.random.random()*360-180,translate=(0,0),scale=np.random.random(),shear=0))

            if i % 100 == 0:
                print(self.created_image.shape)
                pixels = ndimage.rotate(self.created_image, -total_rotation, axes=(2,1), reshape=False).reshape(3,32,32).transpose([1, 2, 0])

                # Plot
                plt.title('Iteration {}'.format(i))
                plt.imshow(pixels, interpolation='bilinear')
                plt.savefig('/home/tgebhart/projects/homo_explico/logdir/experiments/alexnet_vis/{}/iteration_{}.png'.format(folder_name, i), format='png')
                # plt.show()
                plt.close()
                # im_path = '../generated/ddream_l' + str(self.selected_layer) + \
                #     '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                # save_image(self.created_image, im_path)


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    recreated_im = im_as_var.data.numpy()[0]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    # recreated_im = np.round(recreated_im * 255)
    return recreated_im

#
# if __name__ == '__main__':
#     # THIS OPERATION IS MEMORY HUNGRY! #
#     # Because of the selected image is very large
#     # If it gives out of memory error or locks the computer
#     # Try it with a smaller image
#     cnn_layer = 34
#     filter_pos = 94
#
#     im_path = '../input_images/dd_tree.jpg'
#     # Fully connected layer is not needed
#     pretrained_model = models.vgg19(pretrained=True).features
#     dd = DeepDream(pretrained_model, cnn_layer, filter_pos, im_path)
#     # This operation can also be done without Pytorch hooks
#     # See layer visualisation for the implementation without hooks
#     dd.dream()
