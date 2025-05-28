import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import cv2


class ImageRigDataset(Dataset):
    def __init__(self, frame_root, rig_root, transform=None):
        self.frame_root = frame_root
        self.rig_root = rig_root
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # resize for ResNet
            transforms.ToTensor(),          # HWC -> CHW and [0,1]
        ])

        self.samples = []

        for video_name in sorted(os.listdir(frame_root)):
            frame_dir = os.path.join(frame_root, video_name)
            rig_path = os.path.join(rig_root, f"{video_name}.txt")

            print(frame_dir, rig_path)

            # Load rig data
            rig_data = np.loadtxt(rig_path, delimiter=',')

            # Load image paths
            frame_files = sorted(os.listdir(frame_dir))
            frame_files = [f for f in frame_files if f.endswith('.jpg')]

            # Pairing images and rigs
            for idx, frame_file in enumerate(frame_files):
                image_path = os.path.join(frame_dir, frame_file)
                self.samples.append((image_path, rig_data[idx]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, rig_vector = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        rig = torch.tensor(rig_vector, dtype=torch.float32)
        return image, rig


def get_dataloader():
    dataset = ImageRigDataset(
        frame_root='/data2/liuchang/Retarget_data/frames_train',
        rig_root='/data2/liuchang/Retarget_data/rig'
    )
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False,
        num_workers=4, pin_memory=True
    )

    return train_loader, test_loader
