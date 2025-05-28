import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from model.net import RetargetNet
from torch.optim.lr_scheduler import StepLR
from data import get_dataloader
from tensorboardX import SummaryWriter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir='runs/update_training')
model = RetargetNet()
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.MSELoss()
epoch = 50

train_loader, test_loader = get_dataloader()

save_path = "weights"
if not os.path.exists(save_path):
    os.makedirs(save_path)

iteration = 0
for e in range(epoch + 1):
    loss_log = []
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    for i, (frames, rigs) in pbar:
        iteration += 1
        frames, rigs = frames.to(device), rigs.to(device)
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, rigs)
        loss.backward()
        optimizer.step()
        loss_log.append(loss.item())
        pbar.set_description("(Epoch {}, iteration {}) TRAIN LOSS:{:.7f}"
                             .format(e + 1, iteration, np.mean(loss_log)))
        writer.add_scalar('Loss/train', loss.item(), iteration)
    scheduler.step()

    # Validation after epoch
    valid_loss_log = []
    model.eval()
    with torch.no_grad():
        for frames, rigs in test_loader:
            frames, rigs = frames.to(device), rigs.to(device)
            outputs = model(frames)
            loss = criterion(outputs, rigs)
            valid_loss_log.append(loss.item())
    current_loss = np.mean(valid_loss_log)
    writer.add_scalar('Loss/val', current_loss, e + 1)
    print("Epoch: {}, Validation Loss: {:.7f}".format(e + 1, current_loss))

    # save
    if (e > 0 and e % 25 == 0) or e == epoch:
        torch.save(model.state_dict(), os.path.join(save_path, '{}_model.pth'.format(e)))

