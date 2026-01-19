
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
from pix2Topix import NestedUResnet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def preprocess_image(img_path):
    """读取并预处理图像"""
    if img_path.endswith('.png') or img_path.endswith('.jpg') or img_path.endswith('.jpeg'):
        img = cv2.imread(img_path)
        if img is not None:
            img = img[:, :, ::-1]  # BGR -> RGB
            img = Image.fromarray(img)
        else:
            img = Image.open(img_path)
    else:
        img = Image.open(img_path)

    transform_ops = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = transform_ops(img).unsqueeze(0).to(device)  # [1,3,H,W]
    return img

def inference(G, img_tensor, out_size=(512, 512)):
    """推理单张图像"""
    with torch.no_grad():
        out = G(img_tensor)

        # 如果是 list，就取最后一个输出
        if isinstance(out, list):
            out = out[-1]

        # 确保是 (N, C, H, W)
        if out.dim() == 3:
            out = out.unsqueeze(0)

        # 插值缩放
        out = F.interpolate(out, size=out_size, mode='bilinear', align_corners=False)

        # [-1,1] -> [0,1]
        out = (out + 1) * 0.5
        out = out.clamp(0, 1)

        # 转换成 numpy (H, W, C)
        out = (out * 255).byte()
        out = out[0].permute(1, 2, 0).cpu().numpy()

    return out


def test_folder(img_dir, save_dir='./results', out_size=(128, 128)):
    """批量处理文件夹"""
    os.makedirs(save_dir, exist_ok=True)

    # 实例化网络并加载权重
    G = NestedUResnet().to(device)
    ckpt = torch.load(r'C:\Users\DY\Desktop\pix2pix2\weights\pix2pix_best.pth')
    G.load_state_dict(ckpt['G_model'], strict=False)
    G.eval()

    # 遍历目录
    for root, _, files in os.walk(img_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                img_tensor = preprocess_image(img_path)
                out = inference(G, img_tensor, out_size)

                # 保存结果
                save_path = os.path.join(save_dir, file)
                Image.fromarray(out).save(save_path)
                print(f'Prediction saved to {save_path}')

if __name__ == '__main__':
    test_folder(
        r'C:\Users\DY\Desktop\pix2pix2\dataset\med\testA',
        save_dir=r'C:\Users\DY\Desktop\pix2pix2\test_results',
        out_size=(512, 512)
    )



# import torch
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# from PIL import Image
# import cv2
# import os
# from pix2Topix import NestedUResnet
#
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
# def load_image(img_path, size=(256, 256)):
#     """读取图像并转换为 tensor"""
#     if img_path.endswith('.png') or img_path.endswith('.jpg') or img_path.endswith('.jpeg'):
#         img = cv2.imread(img_path)
#         if img is None:
#             raise ValueError(f"Cannot read image: {img_path}")
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(img)
#     else:
#         img = Image.open(img_path)
#
#     transform_ops = transforms.Compose([
#         transforms.Resize(size),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#     img_tensor = transform_ops(img).unsqueeze(0).to(device)
#     return img_tensor
#
# def test(img_paths, save_dir='./results', out_size=(512, 512), weight_paths=[]):
#     """对单张或多张图像使用多权重模型进行推理"""
#     # 模型实例化一次
#     G = NestedUResnet().to(device)
#     G.eval()  # 推理模式
#
#     for img_path in img_paths:
#         img_tensor = load_image(img_path)
#         base_name = os.path.basename(img_path)
#
#         for weight_path in weight_paths:
#             # 加载权重
#             ckpt = torch.load(weight_path, map_location=device)
#             if 'G_model' in ckpt:
#                 G.load_state_dict(ckpt['G_model'], strict=False)
#             else:
#                 G.load_state_dict(ckpt, strict=False)
#
#             with torch.no_grad():
#                 out = G(img_tensor)
#
#                 # 处理深度监督输出
#                 if isinstance(out, list):
#                     out = out[-1]
#
#                 if out.dim() == 3:
#                     out = out.unsqueeze(0)
#
#                 # 插值到目标尺寸
#                 out = F.interpolate(out, size=out_size, mode='bilinear', align_corners=False)
#                 # [-1,1] -> [0,1]
#                 out = (out + 1) * 0.5
#                 out = out.clamp(0, 1)
#                 # 转换为 numpy
#                 out = (out * 255).byte()
#                 out = out[0].permute(1, 2, 0).cpu().numpy()
#
#             # 保存结果
#             os.makedirs(save_dir, exist_ok=True)
#             weight_name = os.path.splitext(os.path.basename(weight_path))[0]
#             save_path = os.path.join(save_dir, f"{weight_name}_{base_name}")
#             Image.fromarray(out).save(save_path)
#             print(f'Prediction saved to {save_path} (size={out_size})')
#
# if __name__ == '__main__':
#     # 可以一次处理多张图像
#     img_paths = [
#         r'C:\Users\DY\Desktop\pix2pix2\dataset\med\testA\BraTS-GLI-00001-000_T1.png',
#         # r'其他图像路径'
#     ]
#
#     # 所有权重文件
#     weights = [fr"C:\Users\DY\Desktop\pix2pix2\weights\pix2pix_{i}.pth" for i in range(200)]
#
#     test(
#         img_paths=img_paths,
#         save_dir=r'C:\Users\DY\Desktop\pix2pix2\test_results',
#         out_size=(512, 512),
#         weight_paths=weights
#     )
#
#
# import torch
# import lpips
# import numpy as np
# from PIL import Image
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
# from pytorch_msssim import ms_ssim
# import os
# import glob
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# loss_fn = lpips.LPIPS(net='alex').to(device)
#
#
# def load_image_as_tensor(path):
#     img = Image.open(path).convert("RGB")
#     return np.array(img)
#
# def calculate_nmse(ref_img, gen_img):
#     ref_img = ref_img.astype(np.float32)
#     gen_img = gen_img.astype(np.float32)
#     mse = np.mean((ref_img - gen_img) ** 2)
#     nmse = mse / np.mean(ref_img ** 2)
#     return nmse
#
# def to_tensor_rgb(img_np):
#     tensor = torch.tensor(img_np.transpose(2, 0, 1) / 255.0).unsqueeze(0).float()
#     return tensor.to(device)
#
# def evaluate_images(ref_path, gen_path):
#     ref_img = load_image_as_tensor(ref_path)
#     gen_img = load_image_as_tensor(gen_path)
#
#     if ref_img.shape != gen_img.shape:
#         gen_img = np.array(Image.fromarray(gen_img).resize((ref_img.shape[1], ref_img.shape[0])))
#
#     h, w = ref_img.shape[:2]
#     win_size = min(7, h, w)
#
#     psnr = peak_signal_noise_ratio(ref_img, gen_img, data_range=255)
#     ssim = structural_similarity(ref_img, gen_img, data_range=255, win_size=win_size, channel_axis=2)
#     mse = mean_squared_error(ref_img, gen_img)
#     nmse = calculate_nmse(ref_img, gen_img)
#
#     ref_tensor = to_tensor_rgb(ref_img)
#     gen_tensor = to_tensor_rgb(gen_img)
#
#     lpips_score = loss_fn(ref_tensor, gen_tensor).item()
#     ms_ssim_score = ms_ssim(ref_tensor, gen_tensor, data_range=1.0, size_average=True).item()
#
#     return {
#         "PSNR": psnr,
#         "SSIM": ssim,
#         "LPIPS": lpips_score,
#         "MS-SSIM": ms_ssim_score,
#         "MSE": mse,
#         "NMSE": nmse,
#     }
#
#
# if __name__ == "__main__":
#     # 参考图
#     ref_image = r"C:\Users\DY\Desktop\pix2pix2\dataset\med\testB\BraTS-GLI-00001-000_T2.png"
#
#     # 遍历生成图
#     gen_dir = r"C:\Users\DY\Desktop\pix2pix2\test_results"
#     gen_images = sorted(glob.glob(os.path.join(gen_dir, "*.png")))
#
#     best_scores = {
#         "PSNR": (-1, ""),      # (值, 文件名)
#         "SSIM": (-1, ""),
#         "MS-SSIM": (-1, ""),
#         "LPIPS": (1e9, ""),
#         "MSE": (1e9, ""),
#         "NMSE": (1e9, ""),
#     }
#
#     for gen_image in gen_images:
#         score = evaluate_images(ref_image, gen_image)
#         fname = os.path.basename(gen_image)
#
#         # 越大越好
#         for metric in ["PSNR", "SSIM", "MS-SSIM"]:
#             if score[metric] > best_scores[metric][0]:
#                 best_scores[metric] = (score[metric], fname)
#
#         # 越小越好
#         for metric in ["LPIPS", "MSE", "NMSE"]:
#             if score[metric] < best_scores[metric][0]:
#                 best_scores[metric] = (score[metric], fname)
#
#     print("\n==== 最佳结果 ====")
#     for metric, (value, fname) in best_scores.items():
#         print(f"{metric}: {value:.4f} @ {fname}")

