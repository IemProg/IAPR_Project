import torch
from torch import optim, nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
import numpy as np
from data_augmentation import *
from sklearn.model_selection import train_test_split
import random

import imageio
from skimage import color
from PIL import Image
import math
import numpy as np
from skimage import morphology
from skimage.transform import resize
import scipy
import gzip
import random
from tensorflow.keras.datasets import mnist


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 14)
    
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.leaky_relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def train_(self, train_images_norm, train_labels, n_epochs, learning_rate, batch_size):
        self.train()
        opt = optim.SGD(self.parameters(), lr=learning_rate)
        losses = []

        for n in range(n_epochs):
            sum_loss = 0
            for b in range(0, len(train_images_norm), batch_size):
                predictions = self(torch.Tensor(train_images_norm).narrow(0, b, batch_size).view(-1, 1, 28, 28))
                #print(predictions)
                y = torch.LongTensor(train_labels).narrow(0, b, batch_size).squeeze(1)
                loss = F.nll_loss(predictions, y)
                sum_loss = sum_loss + loss.item()
                self.zero_grad()
                loss.backward()
                opt.step()
            losses.append(sum_loss)
            print("Epoch {}, loss is {} ".format(n, sum_loss))
        return losses
            
    def test_(self, batch_size, test_images_norm, test_labels):
        nb_errors = 0
        for b in range(0, len(test_images_norm), batch_size):
            predictions = self(torch.Tensor(test_images_norm).view(-1, 1, 28, 28).narrow(0, b, batch_size))
            predictions_classes = torch.argmax(predictions, dim = 1)
            for k in range(batch_size):
                if torch.Tensor(test_labels)[b+k].item() != predictions_classes[k].item():
                    nb_errors += 1
        return 1 - nb_errors*1.0/len(test_images_norm)


def model(batch_size = 100, n_epochs = 10, learning_rate = 0.2):
    accuracies = []
    for iter_ in range(1):
        cnn = CNN()
        train_losses = cnn.train_(train_images_norm, train_labels, n_epochs, learning_rate, batch_size)
        accuracy = cnn.test_(batch_size, test_images_norm, test_labels)
        accuracies.append(accuracy)
        print("------- Completed run: {} --------".format(iter_))
    mean_accuracy = sum(accuracies)/1.0
    print("Accuracy for CNN on test set with 10 epochs averaged over 1 runs : " + str(mean_accuracy))
    return cnn



# symbols_train ,digits_train = generate_data(n_augmentation = 5000)
# symbols_test, digits_test = generate_data(n_augmentation = 500)

# symbols_train_labeled = data_labeled(symbols_train, False)
# symbols_test_labeled = data_labeled(symbols_test, False)
# digits_train_labeled = data_labeled(digits_train, True)
# digits_test_labeled = data_labeled(digits_test, True)

# _mnist_ = [digits_train_labeled[0],digits_train_labeled[1], digits_test_labeled[0],digits_test_labeled[1]]
# _operators_ = [symbols_train_labeled[0],symbols_train_labeled[1],symbols_test_labeled[0],symbols_test_labeled[1]]
# train_imgs, train_labels, test_imgs, test_labels = concatenate_dataset(_mnist_,_operators_)
    
# scaler = StandardScaler()
# train_imgs = (255 - torch.IntTensor(train_imgs*255)).type(torch.FloatTensor)
# c = list(zip(train_imgs, train_labels))
# random.shuffle(c)
# train_imgs, train_labels = zip(*c)
# test_imgs = (255 - torch.IntTensor(test_imgs*255)).type(torch.FloatTensor)
# c = list(zip(test_imgs, test_labels))
# random.shuffle(c)
# test_imgs, test_labels = zip(*c)
# train_imgs = torch.cat(train_imgs)
# train_labels = torch.LongTensor(train_labels)
# test_imgs = torch.cat(test_imgs)
# test_labels = torch.FloatTensor(test_labels)
# train_images_norm = scaler.fit_transform(train_imgs.reshape(-1, 28*28)).reshape(-1, 28, 28)
# test_images_norm = scaler.transform(test_imgs.reshape(-1, 28*28)).reshape(-1, 28, 28)


# network = model(batch_size = 50, n_epochs = 10, learning_rate = 0.2)