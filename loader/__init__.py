import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader

from loader.dataloader import VGSDDataset

def build_dataset(opt, split='train'):
    name = opt["dataset"]

    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    if split == "train":
        joint_transform = transforms.Compose([
            transforms.Resize((opt["height"], opt["weidth"])),
            transforms.ToTensor()
        ])
        # TODO Add augmentations (random crop, horizontal flip, color)
        input_transform = transforms.Compose([
            transforms.Normalize(_MEAN, _STD)
        ])
        dataset = VGSDDataset(opt, 
                              split=split, 
                              joint_transform=joint_transform, 
                              input_transform=input_transform)
    elif split == "test":
        joint_transform = transforms.Compose([
            transforms.Resize((opt["height"], opt["weidth"])),
            transforms.ToTensor()
        ])
        input_transform = transforms.Compose([
            transforms.Normalize(_MEAN, _STD)
        ])
        dataset = VGSDDataset(opt, 
                              split=split, 
                              joint_transform=joint_transform, 
                              input_transform=input_transform)
        


    return dataset