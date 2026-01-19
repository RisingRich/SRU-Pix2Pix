import torch
import lpips
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from pytorch_msssim import ms_ssim
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = lpips.LPIPS(net='alex').to(device)


def load_image_as_tensor(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)


def calculate_nmse(ref_img, gen_img):
    ref_img = ref_img.astype(np.float32)
    gen_img = gen_img.astype(np.float32)
    mse = np.mean((ref_img - gen_img) ** 2)
    nmse = mse / np.mean(ref_img ** 2)
    return nmse


def to_tensor_rgb(img_np):
    tensor = torch.tensor(img_np.transpose(2, 0, 1) / 255.0).unsqueeze(0).float()
    return tensor.to(device)


def evaluate_images(ref_img, gen_img):
    if ref_img.shape != gen_img.shape:
        gen_img = np.array(Image.fromarray(gen_img).resize((ref_img.shape[1], ref_img.shape[0])))

    h, w = ref_img.shape[:2]
    win_size = min(7, h, w)

    psnr = peak_signal_noise_ratio(ref_img, gen_img, data_range=255)
    ssim = structural_similarity(ref_img, gen_img, data_range=255, win_size=win_size, channel_axis=2)
    mse = mean_squared_error(ref_img, gen_img)
    nmse = calculate_nmse(ref_img, gen_img)

    ref_tensor = to_tensor_rgb(ref_img)
    gen_tensor = to_tensor_rgb(gen_img)

    lpips_score = loss_fn(ref_tensor, gen_tensor).item()
    ms_ssim_score = ms_ssim(ref_tensor, gen_tensor, data_range=1.0, size_average=True, win_size=7).item()

    return {
        "PSNR": psnr,
        "SSIM": ssim,
        "LPIPS": lpips_score,
        "MS-SSIM": ms_ssim_score,
        "MSE": mse,
        "NMSE": nmse,
    }


def evaluate_folder(ref_dir, gen_dir):
    metrics_sum = {
        "PSNR": 0,
        "SSIM": 0,
        "LPIPS": 0,
        "MS-SSIM": 0,
        "MSE": 0,
        "NMSE": 0,
    }
    count = 0

    ref_files = [f for f in os.listdir(ref_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for ref_file in ref_files:
        ref_path = os.path.join(ref_dir, ref_file)

        # 自动匹配生成图：_T2 -> _T1
        gen_file = ref_file.replace("_T2", "_T1")
        gen_path = os.path.join(gen_dir, gen_file)

        if not os.path.exists(gen_path):
            print(f"⚠️ 生成图像不存在: {gen_path}")
            continue

        ref_img = load_image_as_tensor(ref_path)
        gen_img = load_image_as_tensor(gen_path)

        scores = evaluate_images(ref_img, gen_img)
        for k in metrics_sum:
            metrics_sum[k] += scores[k]
        count += 1

    metrics_avg = {k: (v / count if count > 0 else 0) for k, v in metrics_sum.items()}
    return metrics_avg


# def evaluate_folder(ref_dir, gen_dir):
#     metrics_sum = {
#         "PSNR": 0,
#         "SSIM": 0,
#         "LPIPS": 0,
#         "MS-SSIM": 0,
#         "MSE": 0,
#         "NMSE": 0,
#     }
#     count = 0
#
#     ref_files = [f for f in os.listdir(ref_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#
#     for ref_file in ref_files:
#         ref_path = os.path.join(ref_dir, ref_file)
#
#         # 自动匹配生成图：B_xxxx.png → A_xxxx.png
#         if ref_file.startswith("B_"):
#             gen_file = ref_file.replace("B_", "A_", 1)
#         else:
#             print(f"⚠️ 文件名格式不符合预期: {ref_file}")
#             continue
#
#         gen_path = os.path.join(gen_dir, gen_file)
#
#         if not os.path.exists(gen_path):
#             print(f"⚠️ 生成图像不存在: {gen_path}")
#             continue
#
#         ref_img = load_image_as_tensor(ref_path)
#         gen_img = load_image_as_tensor(gen_path)
#
#         scores = evaluate_images(ref_img, gen_img)
#         for k in metrics_sum:
#             metrics_sum[k] += scores[k]
#         count += 1
#
#     metrics_avg = {k: (v / count if count > 0 else 0) for k, v in metrics_sum.items()}
#     return metrics_avg

if __name__ == "__main__":

    ref_dir = r"C:\Users\DY\Desktop\pix2pix1\t1_2_t2\test\B"
    # ref_dir = r"C:\Users\DY\Desktop\pix2pix2\synthRAD2025\AB_D\mr_ct\test\B"
    #
    # gen_dir = r"C:\Users\DY\Desktop\pix2pix2\test_results_AB_1"
    # ref_dir = r"C:\Users\DY\Desktop\pix2pix2\dataset\med\testD"
    # ref_dir = r"C:\Users\DY\Desktop\BBDM-main\results\dataset_name\BrownianBridge\sample_to_eval\ground_truth"
    gen_dir = r"C:\Users\DY\Desktop\pix2pix2\test_results"
    # gen_dir = r"C:\Users\DY\Desktop\pix2pix2\t\test_results\test_results"


    avg_scores = evaluate_folder(ref_dir, gen_dir)
    print("Average metrics:", avg_scores)
#
#
# # 1
# # Average metrics: {'PSNR': np.float64(21.975042270030052), 'SSIM': np.float64(0.6880022462843243), 'LPIPS': 0.27837320651326863, 'MS-SSIM': 0.7843701907566616, 'MSE': np.float64(623.562851642427), 'NMSE': np.float32(0.17688157)}
#
# # 2
# # Average metrics: {'PSNR': np.float64(25.262925268083524), 'SSIM': np.float64(0.832261390345884), 'LPIPS': 0.17624703321892482, 'MS-SSIM': 0.8680731058120728, 'MSE': np.float64(260.8233247903676), 'NMSE': np.float32(0.06762213)}

# import torch
# import torch_fidelity
#
#
# def calculate_fid(ref_dir, gen_dir):
#     metrics = torch_fidelity.calculate_metrics(
#         input1=ref_dir,  # 真实图文件夹
#         input2=gen_dir,  # 生成图文件夹
#         cuda=torch.cuda.is_available(),
#         isc=False, fid=False, kid=True, kid_subset_size=16,  # 👈 改这里，必须小于 52
#         kid_subset_count=100
#     )
#     # fid = metrics['frechet_inception_distance']
#     kid_mean = metrics['kernel_inception_distance_mean']
#     # kid_std = metrics['kernel_inception_distance_std']
#     # return fid, kid_mean, kid_std
#     return kid_mean
#
# if __name__ == "__main__":
#     # ref_dir = r"C:\Users\DY\Desktop\pix2pix2\synthRAD2025\AB_D\mr_ct\test\B"
#     # gen_dir = r"C:\Users\DY\Desktop\pix2pix2\test_results_AB_1"
#     # ref_dir = r"C:\Users\DY\Desktop\pix2pix2\synthRAD2025\AB_D\cbct_ct\test\B"
#     # gen_dir = r"C:\Users\DY\Desktop\pix2pix2\test_results_AB_2"
#     ref_dir = r"C:\Users\DY\Desktop\pix2pix2\dataset\med\testD"
#     gen_dir = r"C:\Users\DY\Desktop\pix2pix2\test_results_1_4"
#
#     fid_score = calculate_fid(ref_dir, gen_dir)
#     print("FID:", fid_score)
