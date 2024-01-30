from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.morphology import dilation, square
import cv2   
import torch

from .file_util import node_path,checkpoints_path
from .utils import *

# 指定本地分割模型文件夹的路径
model_folder_path = checkpoints_path("ComfyUI_Seg_VITON","segformer_b2_clothes")

processor = SegformerImageProcessor.from_pretrained(model_folder_path)
model = AutoModelForSemanticSegmentation.from_pretrained(model_folder_path)


# 切割服装
def get_segmentation(tensor_image):
    cloth = tensor2pil(tensor_image)
    # 预处理和预测
    inputs = processor(images=cloth, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(logits, size=cloth.size[::-1], mode="bilinear", align_corners=False)
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    return pred_seg,cloth

    
# 生成mask
def seg_mask(tensor_image):
    pred_seg,cloth = get_segmentation(tensor_image)
    # 非零像素被标记为1，而零像素被标记为0 处理二值图像
    mask = (pred_seg != 0).astype(np.uint8)
    mask = np.where(mask == 1, 0, 255)  # 保留区域为白色，其他区域为黑色
    # mask = np.where(mask == 1, 255, 0)  
    # Create the cloth-mask image using the mask
    cloth_mask = Image.fromarray(np.uint8(mask))
    cloth_mask = cloth_mask.convert("RGB")
    return cloth_mask,cloth

# 生成mask(seg_mask) 只保留背景mask
def seg_reverse_mask(file):
    return seg_mask_by_label(file,[0])

# Labels: 0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"
def seg_mask_by_label(file,array_label):
    pred_seg,cloth = get_segmentation(file)
    # 选择保留的标签
    labels_to_keep = array_label
    mask = np.isin(pred_seg, labels_to_keep).astype(np.uint8)
    mask = np.where(mask == 1, 0, 255)  # 保留区域为白色，其他区域为黑色
    cloth_mask = Image.fromarray(np.uint8(mask))
    cloth_mask = cloth_mask.convert("RGB")
    return cloth_mask,cloth



def seg_show(pred_seg):
    # 显示原始分割结果
    plt.imshow(pred_seg, cmap="viridis")
    plt.axis('off')
    plt.show()
    
