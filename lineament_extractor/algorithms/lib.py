import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np

from .device import identify_device


def random_sample(img, msk, size):
    rows, cols, _ = img.shape
    max_start_row = rows - size
    max_start_col = cols - size

    if max_start_row < 0 or max_start_col < 0:
        raise ValueError("Subarray size is larger than input array dimensions.")

    start_row = np.random.randint(0, max_start_row + 1)
    start_col = np.random.randint(0, max_start_col + 1)

    sub_img = img[start_row:start_row+size, start_col:start_col+size]
    sub_msk = msk[start_row:start_row+size, start_col:start_col+size]

    return sub_img, sub_msk

def view_random_sample(img, msk, size=128):
    sub_img, sub_msk = random_sample(img, msk, size=128)
    fig, (ax_img, ax_msk, ax_sup) = plt.subplots(1, 3, figsize=(8,8))
    ax_img.imshow(sub_img)
    ax_msk.imshow(sub_msk)
    ax_sup.imshow(sub_img)
    ax_sup.imshow(sub_msk, alpha=0.2)

class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            ConvLayer(in_channels,out_channels),
            ConvLayer(out_channels,out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

class UpConvBlock(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(UpConvBlock, self).__init__()
        self.upscale = nn.ConvTranspose2d(in_channels1, in_channels1, kernel_size=2, stride=2)
        self.conv1 = ConvLayer(in_channels1 + in_channels2, out_channels)
        self.conv2 = ConvLayer(out_channels, out_channels)
        
    def forward(self, x1, x2):
        x_up = self.upscale(x1)
        x = torch.cat([x_up, x2], dim=1)
        return self.conv2(self.conv1(x))

class FinalLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

class UNet_1_1_3_64(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, first_hidden_channels=64):
        super(UNet_1_1_3_64, self).__init__()
        c = first_hidden_channels
        self.block1 = ConvBlock(in_channels, c)
        self.block2 = ConvBlock(c, 2*c)
        self.block3 = ConvBlock(2*c, 4*c)
        self.upblock3 = UpConvBlock(4*c, 2*c, 2*c)
        self.upblock2 = UpConvBlock(2*c, c, c)
        self.upblock1 = UpConvBlock(c, in_channels, in_channels)
        self.final = FinalLayer(in_channels, out_channels)
        
    def forward(self, x0):         
        x1 = self.block1(x0)       
        x2 = self.block2(x1)      
        x = self.block3(x2)      
        x  = self.upblock3(x, x2) 
        x  = self.upblock2(x, x1) 
        x  = self.upblock1(x, x0)  
        x = self.final(x)         
        return x   
    
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=3, first_hidden_channels=64):
        super(UNet, self).__init__()
        self.depth = depth
        self.downblocks = nn.ModuleList()
        self.upblocks = nn.ModuleList()
        c_in, c_out = in_channels, first_hidden_channels
        for _ in range(depth):
            self.downblocks.append(ConvBlock(c_in, c_out))
            c_in, c_out = c_out, 2*c_out
        
        for _ in range(depth - 1):
            c_in, c_out = c_in//2, c_out//2
            self.upblocks.append(UpConvBlock(c_out, c_in, c_in))
            
        self.upblocks.append(UpConvBlock(c_in, in_channels, in_channels))
        self.final = FinalLayer(in_channels, out_channels)
        
    def forward(self, x):
        skips = []
        for layer in self.downblocks:
            skips.append(x.clone())
            x = layer(x)            
        for layer in self.upblocks:
            x = layer(x, skips.pop())
        x = self.final(x)
        return(x)

class MinMaxScaler(object):
    def __call__(self, tensor):
        tmin = tensor.min()
        return (tensor - tmin) / (tensor.max() - tmin + 1e-12)
    
class SegmentationDataset(Dataset):
    def __init__(self, image, mask):
        self.image = image
        self.mask = mask
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.transform_mask = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        img_tile = self.transform_img(self.image[idx])
        mask_tile = self.transform_mask(self.mask[idx])

        return img_tile, mask_tile   
    
def load_examples(img, msk, tile_size=128, stride=8, batch_size=32, shuffle=True):
    rows, cols, channels = img.shape
    tiles = [] 
    
    for y in range(0, rows-tile_size, stride):
        for x in range(0, cols-tile_size, stride):
            tiles.append((img[y:y+tile_size, x:x+tile_size], msk[y:y+tile_size, x:x+tile_size])) 
    
    if shuffle:
        np.random.shuffle(tiles)
        
    return DataLoader(SegmentationDataset(*zip(*tiles)), batch_size=batch_size)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        intersection = torch.sum(inputs * targets)
        dice = (2.0 * intersection + self.smooth) / (torch.sum(inputs) + torch.sum(targets) + self.smooth)
        loss = 1.0 - dice
        return loss
    
def iou_score(outputs, targets, smooth=1e-5):
    intersection = torch.sum(outputs * targets)
    union = torch.sum(outputs) + torch.sum(targets) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

# iou_score
    
def train_and_validate(model, train_dataloader, val_dataloader, num_epochs, criterion, optimizer, device=None):    
    if not device:
        device = identify_device()
    
    train_loss_series = []
    val_loss_series = []
    val_iou_series = []
    val_accuracy_series = []
    
    # tensor_type = getattr(torch, device.type)
    # start = tensor_type.Event(enable_timing=True)
    # end = tensor_type.Event(enable_timing=True)
    # start.record()

    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        val_iou = 0.0
        val_accuracy = 0.0

        # Treinamento
        model.train()
        for images, masks in train_dataloader:

            images = images.float().to(device.type)
            masks = masks.float().to(device.type)
            # Forward pass
            outputs = model(images).to(device.type) 
            # Compute loss using Dice Loss
            loss = criterion(outputs, masks)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        model.eval()
        with torch.no_grad():
            for images, masks in val_dataloader:
                images = images.float().to(device.type)
                masks = masks.float().to(device.type)
                # Forward pass
                outputs = model(images).to(device.type)

                # Compute loss using Dice Loss
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                # Compute IoU and accuracy metrics
                predicted_masks = (outputs > 0.5).float()
                iou = iou_score(predicted_masks, masks)
                val_iou += iou.item() * images.size(0)

                predicted_labels = (predicted_masks > 0.5).flatten().cpu().numpy()
                true_labels = masks.flatten().cpu().numpy()
                accuracy = accuracy_score(true_labels, predicted_labels)
                val_accuracy += accuracy * images.size(0)
                
        train_dataset = train_dataloader.dataset
        val_dataset = val_dataloader.dataset

        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        val_iou /= len(val_dataset)
        val_accuracy /= len(val_dataset)

        train_loss_series.append(train_loss)
        val_loss_series.append(val_loss)
        val_iou_series.append(val_iou)
        val_accuracy_series.append(val_accuracy)

        # Print training progress and validation metrics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Val Accuracy: {val_accuracy:.4f}")
        print()
        
    # end.record()
    # s = start.elapsed_time(end) / 1000
    # h = int(s // 3600)
    # s %= 3600
    # m = int(s // 60)
    # s %= 60
    # print(f'elapsed time: {h}h, {m}m, {s:.2f}s')
    
    return train_loss_series, val_loss_series, val_iou_series, val_accuracy_series

def view_metrics(train_loss, val_loss, val_iou, val_accuracy):
    fig, ax = plt.subplots(1,2)

    ax[0].plot(train_loss, label='train loss')
    ax[0].plot(val_loss, label='val loss')
    ax[0].legend()

    ax[1].plot(val_iou, label='val IoU')
    ax[1].plot(val_accuracy, label='val acc')
    ax[1].legend()

def predict(image, model, tile_size=128, stride=32, device=None):
    if not device:
        device = identify_device()

    rows, cols, channels = image.shape
    image_out = np.zeros((rows, cols), dtype=float)
    image = torch.Tensor(image).float().to(device.type)
    
    #transform = transforms.Compose([MinMaxScaler()])
    
    for y in range(0, rows-tile_size, stride):
        for x in range(0, cols-tile_size, stride):
            inp = image[y:y+tile_size, x:x+tile_size].reshape(1,1,tile_size,tile_size) 
            #inp = transform(inp)
            out = model(inp).detach().cpu().numpy().reshape(tile_size,tile_size)
            image_out[y:y+tile_size, x:x+tile_size] += out
    image_out = (image_out - image_out.min()) / (image_out.max() - image_out.min())
    return image_out

