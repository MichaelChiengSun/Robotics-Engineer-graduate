#!/usr/bin/env python
# # Artificial Intelligence: PyTorch CNN

# ## Data
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
train_set = torchvision.datasets.FashionMNIST(root = ".", train=True,
download=True, transform=transforms.ToTensor())
test_set = torchvision.datasets.FashionMNIST(root = ".", train=False,
download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
# Fix the seed to be able to get the same randomness across runs and hence reproducible outcomes
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# If using CuDNN, otherwise ignore
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

input_data, label = next(iter(train_loader))
plt.imshow(input_data[0,:,:,:].numpy().reshape(28,28), cmap="gray_r");
print("Label is: {}".format(label[0]))
print("Dimension of input data: {}".format(input_data.size()))
print("Dimension of labels: {}".format(label.size()))


# ## Convolutional Neural Network Implementation
# ### CNN with Xavier Uniform initialisation weight, multiple activation function, learning rate of 0.1 with SGD optimiser
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()

        self.conv = nn.Conv2d(1, 32, kernel_size = 5)
        
        self.act_conv = nn.ReLU()
        #self.act_conv = nn.Tanh()
        #self.act_conv = nn.Sigmoid()
        #self.act_conv = nn.ELU()
        
        self.max_pool = nn.AvgPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(32, 64, kernel_size = 5)
        
        self.act_conv1 = nn.ReLU()
        #self.act_conv1 = nn.Tanh()
        #self.act_conv1 = nn.Sigmoid()
        #self.act_conv1 = nn.ELU()
        
        self.max_pool1 = nn.AvgPool2d(2, stride=2)

        # Alternatively use the Sequential container to run layers sequentially

        # self.cnn_model = nn.Sequential(nn.Conv2d(1, 6, kernel_size = 5), nn.Tanh(), nn.AvgPool2d(2, stride=2), nn.Conv2d(6, 16, kernel_size = 5), nn.Tanh(), nn.AvgPool2d(2, stride = 2))

        self.fc = nn.Linear(1024, 1024)
        
        self.act = nn.ReLU()
        #self.act = nn.Tanh()
        #self.act = nn.Sigmoid()
        #self.act = nn.ELU()
        
        self.fc1 = nn.Linear(1024, 256)
        # self.dropout = nn.Dropout(p=0.3)    # Dropout rate on second fully connected layer
        
        self.act1 = nn.ReLU()
        #self.act1 = nn.Tanh()
        #self.act1 = nn.Sigmoid()
        #self.act1 = nn.ELU()
        
        self.fc2 = nn.Linear(256, 10)
        
        # Alternatively use the Sequential container to run layers sequentially

        # self.fc_model = nn.Sequential(nn.Linear(256, 120), nn.Tanh(), nn.Linear(120,84), nn.Tanh(), nn.Linear(84, 10))

    def forward(self, x):

        x = self.conv(x)
        x = self.act_conv(x)
        x = self.max_pool(x)

        x = self.conv1(x)
        x = self.act_conv1(x)
        x = self.max_pool1(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = self.act(x)
        x = self.fc1(x)
        x = self.act1(x)
        #x = self.dropout(x)
        x = self.fc2(x)

        # Alternatively use the Sequential container to run layers sequentially

        # x = self.cnn_model(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc_model(x)

        return x

def weights_init(layer):
    if isinstance(layer, nn.Linear):
        # This is a Xavier Uniform initialization
        nn.init.xavier_uniform_(layer.weight)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net = MyCNN().to(device)
net.apply(weights_init)
loss_fn = nn.CrossEntropyLoss()

#opt = torch.optim.SGD(list(net.parameters()), lr = 0.001)
opt = torch.optim.SGD(list(net.parameters()), lr = 0.1)
#opt = torch.optim.SGD(list(net.parameters()), lr = 0.5)
#opt = torch.optim.SGD(list(net.parameters()), lr = 1)
#opt = torch.optim.SGD(list(net.parameters()), lr = 10)

def evaluation(dataloader):
    total, correct = 0,0
    net.eval()
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return 100 * correct / total

loss_epoch_array = []
max_epochs = 30
loss_epoch = 0
train_accuracy = []
test_accuracy = []
for epoch in range(max_epochs):
    loss_epoch = 0
    for i, data in enumerate(train_loader, 0):
        net.train()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        opt.zero_grad()
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        opt.step()
        loss_epoch += loss.item()
        
    loss_epoch_array.append(loss_epoch)
    train_accuracy.append(evaluation(train_loader))
    test_accuracy.append(evaluation(test_loader))
    print("Epoch {}: loss: {}, train accuracy: {}, test accuracy:{}".format(epoch + 1, loss_epoch_array[-1], train_accuracy[-1], test_accuracy[-1]))

# Plot accuracy on training and test sets per each epoch  
fig, ax = plt.subplots()
ax.plot(range(max_epochs), train_accuracy, label='train_accuracy')
ax.plot(range(max_epochs), test_accuracy, label='test_accuracy')
ax.legend()
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()    

# Plot train loss per epoch
plt.plot(range(max_epochs), loss_epoch_array)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

