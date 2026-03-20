import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class PropertyDataset(Dataset):
    """
    Property risk detection dataset
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((640,640)),
            transforms.ToTensor()
        ])
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        for cls_idx, cls_name in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls_name)
            for img_file in os.listdir(cls_dir):
                if img_file.endswith(('.jpg','.png')):
                    self.samples.append((os.path.join(cls_dir, img_file), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls_idx = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # multi-class one-hot encoding
        label = torch.zeros(len(self.classes))
        label[cls_idx] = 1
        return img, label