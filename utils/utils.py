import torch
import numpy as np
import io
from PIL import Image,ImageOps, ImageFilter
from urllib.request import urlopen
import datetime
import os
import cv2
import requests
from io import BytesIO

#comfyui自己的图像就是tensor
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# PIL to Mask
def pil2mask(image):
    image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np)
    return 1.0 - mask

# tensor to cv2
def tensor2cv2(img):
    tmp = ((img.cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    return cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)



# convert PIL image to numpy array
def piltonumpy_array(binary_mask_image):
    tensor_bw = binary_mask_image.convert("RGB")
    tensor_bw = np.array(tensor_bw).astype(np.float32) / 255.0
    tensor_bw = torch.from_numpy(tensor_bw)[None,]
    tensor_bw = tensor_bw.squeeze(0)[..., 0]
    return tensor_bw

# 保存tensor图片
def save_tensor_image(img,path):
    pil_image = tensor2pil(img)
    pil_image.save(path)
    
def save_image(img, filename):
    tmp = ((img.detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))
    
def ioBytes_to_tensor(bytes):
    image = Image.open(bytes)
    return pil2tensor(image)
    
# Flutter ImageFilter 所使用的模糊算法高斯模糊
def gaussian_region(image, radius=5.0):
            image = ImageOps.invert(image.convert("L"))
            image = image.filter(ImageFilter.GaussianBlur(radius=int(radius)))
            return image.convert("RGB")

# 网络下载图片
def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def open_pil_image(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    image = img.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image

def img_from_url(url):
    img = io.BytesIO(urlopen(url).read())
    return open_pil_image(img)

def save_images(img_list, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    folder_path = os.path.join(folder, date_str)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    time_str = now.strftime("%H_%M_%S")
    for idx, img in enumerate(img_list):
        image_number = idx + 1
        filename = f"{time_str}_{image_number}.jpg"
        save_path = os.path.join(folder_path, filename)
        cv2.imwrite(save_path, img[..., ::-1])


# 未测试
def save_tensors_images(img_tensors, img_names, save_dir):
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = (img_tensor.clone()+1)*0.5 * 255
        tensor = tensor.cpu().clamp(0,255)

        try:
            array = tensor.numpy().astype('uint8')
        except:
            array = tensor.detach().numpy().astype('uint8')

        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)

        im = Image.fromarray(array)
        im.save(os.path.join(save_dir, img_name), format='JPEG')

def check_channels(image):
    channels = image.shape[2] if len(image.shape) == 3 else 1
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif channels > 3:
        image = image[:, :, :3]
    return image


def resize_image(img, max_length=768):
    height, width = img.shape[:2]
    max_dimension = max(height, width)

    if max_dimension > max_length:
        scale_factor = max_length / max_dimension
        new_width = int(round(width * scale_factor))
        new_height = int(round(height * scale_factor))
        new_size = (new_width, new_height)
        img = cv2.resize(img, new_size)
    height, width = img.shape[:2]
    img = cv2.resize(img, (width-(width % 64), height-(height % 64)))
    return img

# 未测试
def gen_noise(shape):
    noise = np.zeros(shape, dtype=np.uint8)
    ### noise
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    noise = torch.tensor(noise, dtype=torch.float32)
    return noise