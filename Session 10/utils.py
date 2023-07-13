# utils file

import matplotlib.pyplot as plt
import torch
from torchsummary import summary
from torchvision import transforms
import torchvision


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
                
def classfication_result(predictions, labels, device, b=True):
    # for misclassified images, b = False
    return torch.where((predictions.argmax(dim=1) == labels) == b)[0]
    
def plot_loss_accuracy(results, image_path=None):
    train_losses, test_losses, train_acc, test_acc = results
    fig, axs = plt.subplots(1,2,figsize=(15,5))

    axs[0].plot(train_losses, label ='Train')    
    axs[0].plot(test_losses, label ='Test')
    axs[0].set_title("Loss")

    axs[1].plot(train_acc, label ='Train')    
    axs[1].plot(test_acc, label ='Test')
    axs[1].set_title("Accuracy")
    
    plt.legend()
    plt.show()
    
    if image_path is not None:
        fig.savefig(image_path) 
        
def device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device, use_cuda

def show_summary(model, input_size):
  print(summary(model, input_size=input_size))
  
import numpy as np 

def show_examples_dataset_cifar10(batch_data, batch_label, n=12): 
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    fig = plt.figure()
    for i in range(n):
      plt.subplot(3,4,i+1)
      plt.tight_layout()
      plt.imshow(np.transpose(batch_data[i].cpu().squeeze(0).numpy(), (1, 2, 0)).astype('uint8'))
      plt.title(classes[batch_label[i].item()])
      plt.xticks([])
      plt.yticks([])
      
import numpy as np 

def plot_grid(image, label, UnNorm=None, predictions=[], image_path=None):
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
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

    if image_path is not None:
        fig.savefig(image_path)
      
      
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

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

def augmentation(data, mu, sigma):
    if data == 'Train':
        transform = A.Compose([A.HorizontalFlip(),
                           A.ShiftScaleRotate(),
                           A.CoarseDropout(max_holes=1, 
                                           max_height=16, 
                                           max_width=16, 
                                           min_holes=1, 
                                           min_height=16,
                                           min_width=16,
                                           fill_value=np.mean(mu)),
                           A.ToGray(),
                           A.Normalize(mean=mu, std=sigma), 
                           ToTensorV2()])
    elif data == 'Test':
        transform = A.Compose([A.Normalize(mean=mu, std=sigma), 
                           ToTensorV2()])
    else: 
        transform = A.Compose([ToTensorV2()])
   
    return transform

def augmentation_custom_resnet(data, mu, sigma, pad=4):

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
