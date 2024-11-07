import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms
import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0

class Generator(nn.Module):
    def __init__(self, zDim=100, imgDim=64*64*3):
        super(Generator, self).__init__()
        self.zDim = zDim
        self.imgDim = imgDim
        
        self.main = nn.Sequential(
            nn.Linear(zDim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, imgDim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 3, 64, 64)
        return x

class Discriminator(nn.Module):
    def __init__(self, imgDim=64*64*3):
        super(Discriminator, self).__init__()
        self.imgDim = imgDim
        
        self.main = nn.Sequential(
            nn.Linear(imgDim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, self.imgDim)
        return self.main(x)

def main():
    config = {
        "data_params": {
            "full_image_size": (64, 64),
            "zDim": 256,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            'imagedim': 64 * 64 * 3,
        },
        "train_params": {
            "batch_size": 32,
            'num_epochs': 1000,
            'logstep': 350,
            "lr": 1e-4,
        }
    }

    transform = transforms.Compose([
        transforms.Resize(config["data_params"]["full_image_size"]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageDataset(root_dir='D:/DL project/archive (2)/train', transform=transform)
    train_loader = DataLoader(
        dataset,
        batch_size=config["train_params"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    for images, labels in train_loader:
        plt.imshow(images[0].permute(1, 2, 0).numpy())
        plt.show()
        break
    
    discriminator = Discriminator(imgDim=config["data_params"]["imagedim"]).to(config["data_params"]["device"])
    generator = Generator(zDim=config["data_params"]["zDim"], imgDim=config["data_params"]["imagedim"]).to(config["data_params"]["device"])

    fixedNoise = torch.randn((config["train_params"]["batch_size"], config["data_params"]["zDim"])).to(config["data_params"]["device"])

    lr_disc = 1e-5  # Lower learning rate for discriminator
    lr_gen = 5e-5   # Keep the learning rate for generator the same

    optDisc = optim.Adam(discriminator.parameters(), lr=lr_disc)
    optGen = optim.Adam(generator.parameters(), lr=lr_gen)

    criterion = nn.BCELoss()

    writerFake = SummaryWriter('D:\\DL project\\summary_sir\\fake1')
    writerReal = SummaryWriter('D:\\DL project\\summary_sir\\real1')

    step = 0

    print(f"\nStarted Training and visualization...")

    for epoch in range(config["train_params"]["num_epochs"]):
        print('-' * 80)
        for batch_idx, (real, _) in enumerate(train_loader):
            real = real.to(config["data_params"]["device"])
            batchsize = real.shape[0]

            noise = torch.randn(batchsize, config["data_params"]["zDim"]).to(config["data_params"]["device"])
            fake = generator(noise).to(config["data_params"]["device"])

            discReal = discriminator(real).view(-1)
            lossDreal = criterion(discReal, torch.ones_like(discReal))
            discFake = discriminator(fake).view(-1)
            lossDfake = criterion(discFake, torch.zeros_like(discFake))
            lossD = (lossDreal + lossDfake) / 2

            discriminator.zero_grad()
            lossD.backward()
            optDisc.step()

            noise = torch.randn(batchsize, config["data_params"]["zDim"]).to(config["data_params"]["device"])
            fake = generator(noise)

            output = discriminator(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            generator.zero_grad()
            lossG.backward()
            optGen.step()

            if batch_idx % config["train_params"]["logstep"] == 0:
                print(
                    f"Epoch [{epoch}/{config['train_params']['num_epochs']}]  Batch {batch_idx}/ {len(train_loader)}\n Loss Disc: {lossD:.4f}, Loss Gen: {lossG:.4f}"
                )
                with torch.no_grad():
                    fake = generator(fixedNoise).reshape(-1, 3, 64, 64)
                    data = real.reshape(-1, 3, 64, 64)
                    imgGridFake = torchvision.utils.make_grid(fake, normalize=True)
                    imgGridReal = torchvision.utils.make_grid(data, normalize=True)
                    writerFake.add_image("Fake Image", imgGridFake, global_step=step)
                    writerReal.add_image("Real Image", imgGridReal, global_step=step)
                    writerFake.flush()
                    writerReal.flush()
                    step += 1

    writerFake.close()
    writerReal.close()

if __name__ == "__main__":
    main()
