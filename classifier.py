import torch
from torch import optim, nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
import numpy as np
from data_augmentation import *
from sklearn.model_selection import train_test_split

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 15)
    
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
                loss = F.nll_loss(predictions, torch.LongTensor(train_labels).narrow(0, b, batch_size))
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
        return 1 - nb_errors*1.0/len(test_images)


def model(batch_size = 100, n_epochs = 10, learning_rate = 0.2):
	accuracies = []
	for iter_ in range(5):
	    cnn = CNN()
	    train_losses = cnn.train_(train_images_norm, train_labels, n_epochs, learning_rate, batch_size)
	    accuracy = cnn.test_(batch_size, test_norm, test_labels)
	    accuracies.append(accuracy)
	    print("------- Completed run: {} --------".format(iter_))
	mean_accuracy = sum(accuracies)/5.0
	print("Accuracy for CNN on test set with 10 epochs averaged over 5 runs : " + str(mean_accuracy))


#Prepare data
data_operators, data_digits = generate_data(path = "operators/", image_len = 28, image_wid = 28, n_augmentation = 2000)
data_oper, labels_oper = data_labeled(data_operators)

#mnist[0]: train_data,  mnist[1]:train_labels , mnist[2]: test_data,  mnist[3]:test_labels 
X_train, X_test, y_train, y_test = train_test_split(data_oper, labels_oper, test_size=0.2, random_state=42)
operators = [X_train, y_train, X_test, y_test]

#Loading mnist
"""
    Your folder should contain : - 'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 
                                 - 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'
"""
mnist = load_mnist(data_folder = "./mnist")

train_imgs, train_labels, test_imgs, test_labels = concatenate_dataset(mnist, operators)

scaler = StandardScaler()
train_images_norm = scaler.fit_transform(train_imgs.reshape(-1, 28*28)).reshape(-1, 28, 28)
test_images_norm = scaler.transform(test_imgs.reshape(-1, 28*28)).reshape(-1, 28, 28)


model(batch_size = 100, n_epochs = 25, learning_rate = 0.2)

torch.save(cnn.state_dict(), "CNN_weights1")


# Loading weights 
#model = CNN()
#model.load_state_dict(torch.load("cnn1_weight"))
#model.eval()
