from torch.utils.tensorboard import SummaryWriter
from pix2Topix import NestedUResnet, pix2pixD_256
import argparse
from mydatasets import CreateDatasets
import os
from torch.utils.data.dataloader import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from utils import train_one_epoch, val


def train(opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    os.makedirs(opt.savePath, exist_ok=True)

    train_dataset = CreateDatasets(opt.source_train, opt.target_train, img_size=opt.imgsize)
    val_dataset = CreateDatasets(opt.source_val, opt.target_val, img_size=opt.imgsize)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch, shuffle=True,
                              num_workers=opt.numworker, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch, shuffle=False,
                            num_workers=opt.numworker, drop_last=False)


    pix_G = NestedUResnet().to(device)

    pix_D = pix2pixD_256().to(device)

    # ---------- 优化器 & 损失 ----------
    optim_G = optim.Adam(pix_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_D = optim.Adam(pix_D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    start_epoch = 0

    # ---------- 加载预训练权重 ----------
    if opt.weight != '':
        ckpt = torch.load(opt.weight)
        pix_G.load_state_dict(ckpt['G_model'], strict=False)
        pix_D.load_state_dict(ckpt['D_model'], strict=False)
        start_epoch = ckpt['epoch'] + 1

    writer = SummaryWriter('train_logs')
    best_psnr = 0.0
    for epoch in range(start_epoch, opt.epoch):
        loss_G, loss_D = train_one_epoch(G=pix_G, D=pix_D, train_loader=train_loader,
                                         optim_G=optim_G, optim_D=optim_D, writer=writer,
                                         loss=mse_loss, device=device, plot_every=opt.every,
                                         epoch=epoch, l1_loss=l1_loss)

        writer.add_scalars('train_loss', {'loss_G': loss_G, 'loss_D': loss_D}, epoch)

        # 验证集
        metrics = val(G=pix_G, D=pix_D, val_loader=val_loader, device=device, epoch=epoch)

        print(f"Epoch {epoch}: PSNR={metrics['PSNR']:.4f}, SSIM={metrics['SSIM']:.4f}, NMSE={metrics['NMSE']:.6f}")

        # 保存每个 epoch
        torch.save({
            'G_model': pix_G.state_dict(),
            'D_model': pix_D.state_dict(),
            'epoch': epoch
        }, os.path.join(opt.savePath, f'pix2pix_{epoch}.pth'))

        # 用 PSNR 挑选最佳模型
        if metrics['PSNR'] > best_psnr:
            best_psnr = metrics['PSNR']
            torch.save({
                'G_model': pix_G.state_dict(),
                'D_model': pix_D.state_dict(),
                'epoch': epoch
            }, os.path.join(opt.savePath, f'pix2pix_best.pth'))
            print(f"✅ Saved best model at epoch {epoch} (PSNR={best_psnr:.4f})")


# -------------------- 配置 --------------------
def cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--imgsize', type=int, default=256)
    parser.add_argument('--source_train', type=str,
                        default=r'C:\Users\DY\Desktop\pix2pix2\dataset\med\trainA',
                        required=False, help='训练源图像文件夹路径 (A)')
    parser.add_argument('--target_train', type=str,
                        default=r'C:\Users\DY\Desktop\pix2pix2\dataset\med\trainB',
                        required=False, help='训练标签图像文件夹路径 (B)')
    parser.add_argument('--source_val', type=str,
                        default=r'C:\Users\DY\Desktop\pix2pix2\dataset\med\testA',
                        required=False,
                        help='验证源图像文件夹路径 (A)')
    parser.add_argument('--target_val', type=str,
                        default=r'C:\Users\DY\Desktop\pix2pix2\dataset\med\testB',
                        required=False,
                        help='验证标签图像文件夹路径 (B)')
    parser.add_argument('--weight', type=str, default=r'', help='预训练权重路径')
    parser.add_argument('--savePath', type=str, default='./weights', help='权重保存路径')
    parser.add_argument('--numworker', type=int, default=4)
    parser.add_argument('--every', type=int, default=5, help='每多少 epoch 打印一次训练结果')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = cfg()
    print(opt)
    train(opt)
