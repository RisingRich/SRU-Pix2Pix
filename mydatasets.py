# from torch.utils.data.dataset import Dataset
# import torchvision.transforms as transform
# from PIL import Image
# import cv2
#
#
# class CreateDatasets(Dataset):
#     def __init__(self, ori_imglist,img_size):
#         self.ori_imglist = ori_imglist
#         self.transform = transform.Compose([
#             transform.ToTensor(),
#             transform.Resize((img_size, img_size)),
#             transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
#
#     def __len__(self):
#         return len(self.ori_imglist)
#
#     def __getitem__(self, item):
#         ori_img = cv2.imread(self.ori_imglist[item])
#         ori_img = ori_img[:, :, ::-1]
#         real_img = Image.open(self.ori_imglist[item].replace('.png', '.jpg'))
#         ori_img = self.transform(ori_img.copy())
#         real_img = self.transform(real_img)
#         return ori_img, real_img

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os


class CreateDatasets(Dataset):
    def __init__(self, source_dir, target_dir, img_size=256):
        """
        source_dir: 源图像文件夹路径 (A)
        target_dir: 标签图像文件夹路径 (B)
        img_size: 图像统一大小
        """
        self.source_paths = sorted([os.path.join(source_dir, f) for f in os.listdir(source_dir)
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.target_paths = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir)
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        assert len(self.source_paths) == len(self.target_paths), "源图像和标签图像数量不一致"

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        source_img = Image.open(self.source_paths[index]).convert('RGB')
        target_img = Image.open(self.target_paths[index]).convert('RGB')

        source_img = self.transform(source_img)
        target_img = self.transform(target_img)
        return source_img, target_img
