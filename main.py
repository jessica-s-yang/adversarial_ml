from model import Net
from train import train
from test import test
from DeviceDataLoader import DeviceDataLoader, get_default_device, to_device

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import time

#parameters
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

def main():     
    #gpu
    device = get_default_device()
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

    # train set and data loader
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)
    # test set and data loader
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)
    #wrap data loader for gpu
    trainloader = DeviceDataLoader(trainloader, device)
    testloader = DeviceDataLoader(testloader, device)

    # model
    input_size = 320
    num_classes = 10
    network = Net(input_size, hidden_size=50, out_size=num_classes)
    # model on GPU
    to_device(network, device)

    # initialize optimizer
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

    # print model's state dict
    print("Model's state_dict:")
    for param_tensor in network.state_dict():
        print(param_tensor, "\t", network.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(trainloader.dl.dataset) for i in range(n_epochs + 1)]
    
    start_time = time.time()
    for epoch in range(1, n_epochs + 1):
        train(trainloader, network, optimizer, n_epochs, log_interval, train_losses, train_counter)
        test(network, test_losses, testloader)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()



    