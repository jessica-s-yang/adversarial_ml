import torch
#https://medium.com/jovianml/training-deep-neural-networks-on-a-gpu-with-pytorch-11079d89805

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

# select GPU as target device if it is available
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        print("cuda")
        return torch.device('cuda')
    else:
        print("cpu")
        return torch.device('cpu')

# function to move data to chosen device
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)