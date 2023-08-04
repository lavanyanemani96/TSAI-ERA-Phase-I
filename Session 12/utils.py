# utils file

import matplotlib.pyplot as plt
import torch
from torchsummary import summary
from torchvision import transforms
import torchvision
import numpy as np 
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class Cifar10SearchDataset(torchvision.datasets.CIFAR10):

    def __init__(self, root="./data", train=True, download=True, transform=None):
      super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
      image, label = self.data[index], self.targets[index]
      
      if self.transform is not None:
        transformed = self.transform(image=image)
        image = transformed["image"]

        return image, label
        
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import numpy as np 
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def augmentation_custom_resnet(data, mu=(0.49139968, 0.48215827, 0.44653124), sigma=(0.24703233, 0.24348505, 0.26158768), pad=4):

  if data == 'Train':
    transform = A.Compose([A.PadIfNeeded(min_height=32+pad,
                                        min_width=32+pad,
                                        border_mode=cv2.BORDER_CONSTANT,
                                        value=np.mean(mu)),
                            A.RandomCrop(32, 32),
                            A.HorizontalFlip(p=0.5),
                            A.Cutout(max_h_size=8, max_w_size=8),
                            A.Normalize(mean=mu, std=sigma),
                            ToTensorV2()])
  else:
    transform = A.Compose([A.Normalize(mean=mu, std=sigma),
                           ToTensorV2()])

  return transform

def classfication_result(predictions, labels, device, b=True):
    # for misclassified images, b = False
    return torch.where((predictions.argmax(dim=1) == labels) == b)[0]


def device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device, use_cuda
    
def classfication_result(predictions, labels, device, b=True):
    # for misclassified images, b = False
    return torch.where((predictions.argmax(dim=1) == labels) == b)[0]
    

def plot_grid(image, label, UnNorm=None, predictions=[]):

    nrows = 2
    ncols = 5

    fig, ax = plt.subplots(nrows, ncols, figsize=(8, 4))
    if len(predictions):
        for i in range(nrows):
            for j in range(ncols):
                index = i * ncols + j
                ax[i, j].axis("off")
                ax[i, j].set_title('Label: %s, \nPred: %s' %(classes[label[index].cpu()],classes[predictions[index].cpu().argmax()]))
                ax[i, j].imshow(np.transpose(UnNorm(image[index].cpu()), (1, 2, 0)))
    else:
        for i in range(nrows):
            for j in range(ncols):
                index = i * ncols + j
                ax[i, j].axis("off")
                ax[i, j].set_title("Label: %s" %(classes[label[index]]))
                ax[i, j].imshow(np.transpose(image[index], (1, 2, 0)))
                
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def mean_std_cifar10(dataset):

  train_data = dataset.data
  train_data = train_data / 255.0

  mean = np.mean(train_data, axis=tuple(range(train_data.ndim-1)))
  std = np.std(train_data, axis=tuple(range(train_data.ndim-1)))

  return mean, std