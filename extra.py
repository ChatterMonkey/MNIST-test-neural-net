train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/Users/mayabasu/pytorchdatasets', train=True, download=True,  #download is directed to /Users/mayabasu/pytorchdatasets
                              transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),   #converts to torch.tensor
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))                 #the global mean and standard deviation of the data set to normalize it
                             ])),
    batch_size=train_batch_size, shuffle=True)    #changes up the order and divides it into sets of 64

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/Users/mayabasu/pytorchdatasets', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=test_batch_size, shuffle=False)

