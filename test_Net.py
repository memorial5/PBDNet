import torch
import torch.nn.functional as F
import sys
import warnings
from tqdm import tqdm
import numpy as np
import os, argparse
import cv2
from model.PBDNet import PBDNet
from data import test_dataset

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--trainsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
# parser.add_argument('--test_path', type=str, default='', help='test dataset path')
parser.add_argument('--save_path', type=str, default='', help='save path')
parser.add_argument('--pth_path', type=str, default='./ckps/PBDNet/PBDNet_mae_best.pth', help='checkpoint path')
opt = parser.parse_args()

# dataset_path = opt.test_path

# set device for test
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIB_LEDEVICES"] = "1"
    print('USE GPU 1')
# load the model
model = PBDNet().eval().cuda()


model.load_state_dict(torch.load(opt.pth_path))


save_path = ""
if not os.path.exists(save_path):
    os.makedirs(save_path)
image_root = "./dataset/test/S0/"
gt_root = "./dataset/test/gtmask/"
depth_root = "./dataset/test/DOP_Gray/"

# depth_root = dataset_path + dataset + '/T/'
test_loader = test_dataset(image_root, gt_root, depth_root, opt.trainsize)
for i in tqdm(range(test_loader.size), file=sys.stdout):
    image, gt, depth, name, image_for_post = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    image = image.cuda()
    depth = depth.repeat(1, 3, 1, 1).cuda()
    res, pred_2, pred_3, pred_4 = model(image, depth)
    res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    cv2.imwrite(save_path + name, res * 255)
print('Test Done!')
