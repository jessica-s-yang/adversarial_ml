import torch.nn.functional as F
import torch
import os

def train(train_loader, network, optimizer, epoch, log_interval, train_losses, train_counter):
      for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                  print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * len(data), len(train_loader.dl.dataset),
                  100. * batch_idx / len(train_loader), loss.item()))
                  train_losses.append(loss.item())
                  train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dl.dataset)))
            
                  torch.save(network.state_dict(), os.getcwd()+'/results/model.pth')
                  torch.save(optimizer.state_dict(), os.getcwd()+'/results/optimizer.pth')
      print('Finished Training')

      
