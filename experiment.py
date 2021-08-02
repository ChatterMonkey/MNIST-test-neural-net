from mnist.loaders import train_loader


for id, (data,target) in enumerate(train_loader):
    print(data.shape)
