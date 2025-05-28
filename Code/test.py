import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from model.net import RetargetNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ImageDataset(Dataset):
    def __init__(self, frame_dir):
        self._samples = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # resize for ResNet
            transforms.ToTensor(),          # HWC -> CHW and [0,1]
        ])

        frame_files = sorted(os.listdir(frame_dir))
        frame_files = [f for f in frame_files if f.endswith('.jpg')]

        for idx, frame_file in enumerate(frame_files):
            image_path = os.path.join(frame_dir, frame_file)
            self._samples.append(image_path)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        image_path = self._samples[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image


def predict_image(model, image_path, save_path):
    dataset = ImageDataset(image_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    rig_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            rig_predictions.extend(outputs.cpu().numpy())

    rig_predictions = np.array(rig_predictions)
    print(rig_predictions.shape)
    np.savetxt(save_path, rig_predictions, delimiter=',')


if __name__ == '__main__':
    weight_path = "weights/50_model.pth"
    test_root = "data/retarget_test"
    save_root = "data/pred_rig"
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    model = RetargetNet().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    for file in os.listdir(test_root):
        image_path = os.path.join(test_root, file)
        save_file = os.path.join(os.path.join(save_root), file + ".txt")
        print(image_path, save_file)
        predict_image(model, image_path, save_file)
