import numpy as np
import torchvision
from tqdm import tqdm
import torch
import os
from pytorch_msssim import ssim, ms_ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
# 例如 MS-SSIM Loss
def ms_ssim_loss(pred, target):
    return 1 - ms_ssim(pred, target, data_range=1.0, size_average=True)

# def train_one_epoch(G, D, train_loader, optim_G, optim_D, writer, loss, device, plot_every, epoch, l1_loss):
#     pd = tqdm(train_loader)
#     loss_D, loss_G = 0, 0
#     step = 0
#     G.train()
#     D.train()
#     for idx, data in enumerate(pd):
#         in_img = data[0].to(device)
#         real_img = data[1].to(device)
#         # 先训练D
#         fake_img = G(in_img)
#         D_fake_out = D(fake_img.detach(), in_img).squeeze()
#         D_real_out = D(real_img, in_img).squeeze()
#         ls_D1 = loss(D_fake_out, torch.zeros(D_fake_out.size()).cuda())
#         ls_D2 = loss(D_real_out, torch.ones(D_real_out.size()).cuda())
#         ls_D = (ls_D1 + ls_D2) * 0.5
#
#         optim_D.zero_grad()
#         ls_D.backward()
#         optim_D.step()
#
#         # 再训练G
#         fake_img = G(in_img)
#         D_fake_out = D(fake_img, in_img).squeeze()
#         ls_G1 = loss(D_fake_out, torch.ones(D_fake_out.size()).cuda())
#         ls_G3 = ms_ssim_loss((fake_img + 1) * 0.5, (real_img + 1) * 0.5)
#         ls_G2 = l1_loss(fake_img, real_img)
#         ls_G = ls_G1 + ls_G2 * 100 + ls_G3 * 100
#
#         optim_G.zero_grad()
#         ls_G.backward()
#         optim_G.step()
#
#         loss_D += ls_D
#         loss_G += ls_G
#
#         pd.desc = 'train_{} G_loss: {} D_loss: {}'.format(epoch, ls_G.item(), ls_D.item())
#         # 绘制训练结果
#         if idx % plot_every == 0:
#             writer.add_images(tag='train_epoch_{}'.format(epoch), img_tensor=0.5 * (fake_img + 1), global_step=step)
#             step += 1
#     mean_lsG = loss_G / len(train_loader)
#     mean_lsD = loss_D / len(train_loader)
#     return mean_lsG, mean_lsD

def train_one_epoch(G, D, train_loader, optim_G, optim_D, writer, loss, device, plot_every, epoch, l1_loss):
    pd = tqdm(train_loader)
    loss_D, loss_G = 0, 0
    step = 0
    G.train()
    D.train()

    for idx, data in enumerate(pd):
        in_img = data[0].to(device)
        real_img = data[1].to(device)

        # ===================== 训练判别器 D =====================
        fake_imgs = G(in_img)
        if isinstance(fake_imgs, list):   # 深度监督
            fake_img = fake_imgs[-1]      # 用最后一层作为 D 的输入
        else:
            fake_img = fake_imgs

        D_fake_out = D(fake_img.detach(), in_img).squeeze()
        D_real_out = D(real_img, in_img).squeeze()
        ls_D1 = loss(D_fake_out, torch.zeros_like(D_fake_out))
        ls_D2 = loss(D_real_out, torch.ones_like(D_real_out))
        ls_D = (ls_D1 + ls_D2) * 0.5

        optim_D.zero_grad()
        ls_D.backward()
        optim_D.step()

        # weights = [0.1, 0.2, 0.3, 0.4]
        weights = [0.02, 0.08, 0.1, 0.8]


        fake_imgs = G(in_img)
        if isinstance(fake_imgs, list):
            ls_G_total = 0
            for w, fi in zip(weights, fake_imgs):
                D_fake_out = D(fi, in_img).squeeze()
                ls_G1 = loss(D_fake_out, torch.ones_like(D_fake_out))
                ls_G3 = ms_ssim_loss((fi + 1) * 0.5, (real_img + 1) * 0.5)
                ls_G2 = l1_loss(fi, real_img)
                ls_G_total += w * (ls_G1 + ls_G2 * 100 + ls_G3 * 100)
            ls_G = ls_G_total
            fake_img = fake_imgs[-1]  # 用最后一个可视化
        else:
            D_fake_out = D(fake_imgs, in_img).squeeze()
            ls_G1 = loss(D_fake_out, torch.ones_like(D_fake_out))
            ls_G3 = ms_ssim_loss((fake_imgs + 1) * 0.5, (real_img + 1) * 0.5)
            ls_G2 = l1_loss(fake_imgs, real_img)
            ls_G = ls_G1 + ls_G2 * 100 + ls_G3 * 100
            fake_img = fake_imgs

        optim_G.zero_grad()
        ls_G.backward()
        optim_G.step()

        loss_D += ls_D.item()
        loss_G += ls_G.item()

        pd.desc = f'train_{epoch} G_loss: {ls_G.item():.4f} D_loss: {ls_D.item():.4f}'
        # 绘制训练结果
        if idx % plot_every == 0:
            writer.add_images(tag=f'train_epoch_{epoch}',
                              img_tensor=0.5 * (fake_img + 1),
                              global_step=step)
            step += 1

    mean_lsG = loss_G / len(train_loader)
    mean_lsD = loss_D / len(train_loader)
    return mean_lsG, mean_lsD

# def train_one_epoch(G, D, train_loader, optim_G, optim_D, writer, loss, device,
#                              plot_every, epoch, l1_loss):
#     """
#     训练一个 epoch，使用深度监督 + 当前 batch 自适应权重（不平滑）
#     """
#     from tqdm import tqdm
#     pd_iter = tqdm(train_loader)
#     loss_D_total, loss_G_total = 0, 0
#     step = 0
#
#     G.train()
#     D.train()
#
#     for idx, data in enumerate(pd_iter):
#         in_img = data[0].to(device)
#         real_img = data[1].to(device)
#
#         # ===================== 训练判别器 D =====================
#         fake_imgs = G(in_img)
#         if isinstance(fake_imgs, list):
#             fake_img_for_D = fake_imgs[-1]
#         else:
#             fake_img_for_D = fake_imgs
#
#         D_fake_out = D(fake_img_for_D.detach(), in_img).squeeze()
#         D_real_out = D(real_img, in_img).squeeze()
#         ls_D1 = loss(D_fake_out, torch.zeros_like(D_fake_out))
#         ls_D2 = loss(D_real_out, torch.ones_like(D_real_out))
#         ls_D = 0.5 * (ls_D1 + ls_D2)
#
#         optim_D.zero_grad()
#         ls_D.backward()
#         optim_D.step()
#
#         # ===================== 训练生成器 G =====================
#         fake_imgs = G(in_img)
#         if isinstance(fake_imgs, list):
#             # 1. 计算每个中间输出的损失
#             losses = []
#             for fi in fake_imgs:
#                 D_fake_out = D(fi, in_img).squeeze()
#                 ls_G1 = loss(D_fake_out, torch.ones_like(D_fake_out))
#                 ls_G2 = l1_loss(fi, real_img)
#                 ls_G3 = ms_ssim_loss((fi + 1) * 0.5, (real_img + 1) * 0.5)
#                 total_loss = ls_G1 + 100 * ls_G2 + 100 * ls_G3
#                 losses.append(total_loss)
#             loss_tensor = torch.stack(losses)  # shape = [num_outputs]
#
#             # 2. 当前 batch 自适应权重（不平滑）
#             batch_weights = loss_tensor / loss_tensor.sum()
#
#             # 3. 总损失 = 加权和
#             ls_G = (batch_weights * loss_tensor).sum()
#
#             fake_img = fake_imgs[-1]  # 最后一个输出可视化
#         else:
#             # 普通单输出
#             D_fake_out = D(fake_imgs, in_img).squeeze()
#             ls_G1 = loss(D_fake_out, torch.ones_like(D_fake_out))
#             ls_G2 = l1_loss(fake_imgs, real_img)
#             ls_G3 = ms_ssim_loss((fake_imgs + 1) * 0.5, (real_img + 1) * 0.5)
#             ls_G = ls_G1 + 100 * ls_G2 + 100 * ls_G3
#             fake_img = fake_imgs
#
#         optim_G.zero_grad()
#         ls_G.backward()
#         optim_G.step()
#
#         # ===================== 统计损失 =====================
#         loss_D_total += ls_D.item()
#         loss_G_total += ls_G.item()
#
#         pd_iter.desc = f'train_{epoch} G_loss: {ls_G.item():.4f} D_loss: {ls_D.item():.4f}'
#
#         # 绘制训练结果
#         if idx % plot_every == 0:
#             writer.add_images(tag=f'train_epoch_{epoch}',
#                               img_tensor=0.5 * (fake_img + 1),
#                               global_step=step)
#             step += 1
#
#     mean_lsG = loss_G_total / len(train_loader)
#     mean_lsD = loss_D_total / len(train_loader)
#     return mean_lsG, mean_lsD


# def train_one_epoch(G, D, train_loader, optim_G, optim_D, writer, loss, device, plot_every, epoch, l1_loss,
#                     momentum=0.9):
#     pd = tqdm(train_loader)
#     loss_D, loss_G = 0, 0
#     step = 0
#     G.train()
#     D.train()
#
#     # 初始化滑动权重，长度 = 中间输出数量
#     running_weights = None
#
#     for idx, data in enumerate(pd):
#         in_img = data[0].to(device)
#         real_img = data[1].to(device)
#
#         # ===================== 训练判别器 D =====================
#         fake_imgs = G(in_img)
#         if isinstance(fake_imgs, list):  # 深度监督
#             fake_img = fake_imgs[-1]  # 用最后一层作为 D 的输入
#         else:
#             fake_img = fake_imgs
#
#         D_fake_out = D(fake_img.detach(), in_img).squeeze()
#         D_real_out = D(real_img, in_img).squeeze()
#         ls_D1 = loss(D_fake_out, torch.zeros_like(D_fake_out))
#         ls_D2 = loss(D_real_out, torch.ones_like(D_real_out))
#         ls_D = (ls_D1 + ls_D2) * 0.5
#
#         optim_D.zero_grad()
#         ls_D.backward()
#         optim_D.step()
#
#         # ===================== 训练生成器 G =====================
#         fake_imgs = G(in_img)
#         if isinstance(fake_imgs, list):
#             # 1. 计算每个中间输出的损失
#             losses = []
#             for fi in fake_imgs:
#                 D_fake_out = D(fi, in_img).squeeze()
#                 ls_G1 = loss(D_fake_out, torch.ones_like(D_fake_out))
#                 ls_G2 = l1_loss(fi, real_img)
#                 ls_G3 = ms_ssim_loss((fi + 1) * 0.5, (real_img + 1) * 0.5)
#                 total_loss = ls_G1 + ls_G2 * 100 + ls_G3 * 100
#                 losses.append(total_loss)
#             loss_tensor = torch.stack(losses)  # shape = [num_outputs]
#
#             # 2. 计算当前 batch 自适应权重
#             batch_weights = loss_tensor / loss_tensor.sum()
#
#             # 3. 初始化 running_weights
#             if running_weights is None:
#                 running_weights = batch_weights.detach()
#             else:
#                 # 4. 滑动平均平滑权重
#                 running_weights = momentum * running_weights + (1 - momentum) * batch_weights.detach()
#
#             # 5. 加权求总损失
#             ls_G = (running_weights * loss_tensor).sum()
#             fake_img = fake_imgs[-1]  # 最后一个输出可视化
#         else:
#             D_fake_out = D(fake_imgs, in_img).squeeze()
#             ls_G1 = loss(D_fake_out, torch.ones_like(D_fake_out))
#             ls_G2 = l1_loss(fake_imgs, real_img)
#             ls_G3 = ms_ssim_loss((fake_imgs + 1) * 0.5, (real_img + 1) * 0.5)
#             ls_G = ls_G1 + ls_G2 * 100 + ls_G3 * 100
#             fake_img = fake_imgs
#
#         optim_G.zero_grad()
#         ls_G.backward()
#         optim_G.step()
#
#         loss_D += ls_D.item()
#         loss_G += ls_G.item()
#
#         pd.desc = f'train_{epoch} G_loss: {ls_G.item():.4f} D_loss: {ls_D.item():.4f}'
#
#         # 绘制训练结果
#         if idx % plot_every == 0:
#             writer.add_images(tag=f'train_epoch_{epoch}',
#                               img_tensor=0.5 * (fake_img + 1),
#                               global_step=step)
#             step += 1
#
#     mean_lsG = loss_G / len(train_loader)
#     mean_lsD = loss_D / len(train_loader)
#     return mean_lsG, mean_lsD


@torch.no_grad()
def tensor_to_img(tensor):
    """把 [-1,1] 的Tensor转换为 numpy uint8 图像"""
    img = tensor.detach().cpu().numpy()
    img = (img + 1) / 2.0  # [-1,1] -> [0,1]
    img = np.transpose(img[0], (1, 2, 0))  # [C,H,W] -> [H,W,C]
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def calculate_nmse(ref_img, gen_img):
    ref_img = ref_img.astype(np.float32)
    gen_img = gen_img.astype(np.float32)
    mse = np.mean((ref_img - gen_img) ** 2)
    nmse = mse / np.mean(ref_img ** 2)
    return nmse


# def val(G, D, val_loader, device, epoch, save_best=True):
#     pd = tqdm(val_loader)
#     G.eval()
#     D.eval()
#
#     all_metrics = []
#     best_score = -1  # 用 PSNR+SSIM-(NMSE) 作为综合指标挑最优
#
#     for idx, item in enumerate(pd):
#         in_img = item[0].to(device)
#         real_img = item[1].to(device)
#
#         with torch.no_grad():
#             fake_img = G(in_img)
#
#         # 转换为 numpy 图像
#         real_np = tensor_to_img(real_img)
#         fake_np = tensor_to_img(fake_img)
#
#         # 计算指标
#         psnr = peak_signal_noise_ratio(real_np, fake_np, data_range=255)
#         ssim = structural_similarity(real_np, fake_np, channel_axis=2, data_range=255)
#         nmse = calculate_nmse(real_np, fake_np)
#
#         metrics = {"PSNR": psnr, "SSIM": ssim, "NMSE": nmse}
#         all_metrics.append(metrics)
#
#         pd.desc = f'val_{epoch}: PSNR:{psnr:.2f} SSIM:{ssim:.4f} NMSE:{nmse:.4f}'
#
#         # 选最优模型
#         score = psnr + ssim - nmse
#         if save_best and score > best_score:
#             best_score = score
#             best_image = fake_img
#
#     # 保存最优生成结果
#     if save_best and 'best_image' in locals():
#         result_img = (best_image + 1) * 0.5
#         if not os.path.exists('results'):
#             os.mkdir('results')
#         torchvision.utils.save_image(result_img, f'results/val_epoch{epoch}.jpg')
#
#     # 计算平均指标
#     avg_metrics = {
#         "PSNR": np.mean([m["PSNR"] for m in all_metrics]),
#         "SSIM": np.mean([m["SSIM"] for m in all_metrics]),
#         "NMSE": np.mean([m["NMSE"] for m in all_metrics]),
#     }
#
#     return avg_metrics
#
def val(G, D, val_loader, device, epoch, save_best=True):
    pd = tqdm(val_loader)
    G.eval()
    D.eval()

    all_metrics = []
    best_score = -1  # 用 PSNR+SSIM-(NMSE) 作为综合指标挑最优

    for idx, item in enumerate(pd):
        in_img = item[0].to(device)
        real_img = item[1].to(device)

        with torch.no_grad():
            fake_imgs = G(in_img)
            if isinstance(fake_imgs, list):
                fake_img = fake_imgs[-1]  # 只取最后输出
            else:
                fake_img = fake_imgs

        # 转换为 numpy 图像
        real_np = tensor_to_img(real_img)
        fake_np = tensor_to_img(fake_img)

        # 计算指标
        psnr = peak_signal_noise_ratio(real_np, fake_np, data_range=255)
        ssim_val = structural_similarity(real_np, fake_np, channel_axis=2, data_range=255)
        nmse = calculate_nmse(real_np, fake_np)

        metrics = {"PSNR": psnr, "SSIM": ssim_val, "NMSE": nmse}
        all_metrics.append(metrics)

        pd.desc = f'val_{epoch}: PSNR:{psnr:.2f} SSIM:{ssim_val:.4f} NMSE:{nmse:.4f}'

        # 选最优模型
        score = psnr + ssim_val - nmse
        if save_best and score > best_score:
            best_score = score
            best_image = fake_img

    # 保存最优生成结果
    if save_best and 'best_image' in locals():
        result_img = (best_image + 1) * 0.5
        if not os.path.exists('results'):
            os.mkdir('results')
        torchvision.utils.save_image(result_img, f'results/val_epoch{epoch}.jpg')

    # 计算平均指标
    avg_metrics = {
        "PSNR": np.mean([m["PSNR"] for m in all_metrics]),
        "SSIM": np.mean([m["SSIM"] for m in all_metrics]),
        "NMSE": np.mean([m["NMSE"] for m in all_metrics]),
    }

    return avg_metrics
