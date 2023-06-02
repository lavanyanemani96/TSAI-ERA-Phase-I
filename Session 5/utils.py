# utils file

import matplotlib.pyplot as plt
import torch
from torchsummary import summary

def plot_loss_accuracy(results):
    train_losses, test_losses, train_acc, test_acc = results
    fig, axs = plt.subplots(2,2,figsize=(15,10))

    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")

    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")

    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")

    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    plt.show()
    
def device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device, use_cuda

def show_summary(model, input_size):
  print(summary(model, input_size=input_size))
  
def show_examples_dataset(batch_data, batch_label, n=12): 
    fig = plt.figure()
    for i in range(n):
      plt.subplot(3,4,i+1)
      plt.tight_layout()
      plt.imshow(batch_data[i].squeeze(0), cmap='gray')
      plt.title(batch_label[i].item())
      plt.xticks([])
      plt.yticks([])