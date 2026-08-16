from torchvision import datasets
from torch.utils.data import DataLoader

def create_dataloaders(train_dir,
                       test_dir,
                       train_transform,
                       test_transform,
                       batch_size):

    train_data=datasets.ImageFolder(train_dir,
                                    transform=train_transform)
    
    test_data=datasets.ImageFolder(test_dir,
                                   transform=test_transform)

    train_dataloader=DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True)
    
    test_dataloader=DataLoader(test_data,
                               batch_size=batch_size,
                               shuffle=False)

    class_names=train_data.classes

    return train_dataloader,test_dataloader,class_names
