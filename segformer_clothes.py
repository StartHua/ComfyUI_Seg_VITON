import os
import numpy as np
from urllib.request import urlopen
import torchvision.transforms as transforms  

from .utils.file_util import node_path
from .utils.seg_clothes import *
from rembg import remove

class segformer_remove_bg:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {     
                 "source":("IMAGE", {"default": "","multiline": False})
                }
        }
    RETURN_TYPES = ("IMAGE","BOOLEAN")
    RETURN_NAMES = ("image","open")
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "CXH" 
    def sample(self,source):
        pil_image = tensor2pil(source)

        o_image = remove(pil_image)

        r=  pil2tensor(o_image)
        
        return r,True
    
        
class segformer_agnostic:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {     
                 "source":("IMAGE", {"default": "","multiline": False}),
                 "mask":("MASK", {"default": "","multiline": False}),
                }
        }

    RETURN_TYPES = ("IMAGE","BOOLEAN")
    RETURN_NAMES = ("mark_image","open")
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "CXH"

    def sample(self,source,mask):
        # 将source和mask从tensor转换为PIL Image  
        pil_image = tensor2pil(source)  
        mask_pil = tensor2pil(mask)  
    
        # 将mask转换为灰度图并二值化  
        mask_np = np.array(mask_pil.convert('L'))  
        _, binary_mask = cv2.threshold(mask_np, 240, 255, cv2.THRESH_BINARY)  
    
        # 使用膨胀和腐蚀操作填充区域  
        kernel = np.ones((5, 5), np.uint8)  
        dilation = cv2.dilate(binary_mask, kernel, iterations=1)  
        erosion = cv2.erode(dilation, kernel, iterations=1)  
    
        # 创建一个与原图大小相同的灰色图像  
        # gray_image = np.full(pil_image.size, (128, 128, 128), dtype=np.uint8)  
    
        # 将PIL Image转换为NumPy数组以进行操作  
        pil_image_np = np.array(pil_image)  
    
        # 将mask应用到原始图像上，将指定区域替换为灰色  
        pil_image_np[erosion == 255] = [128, 128, 128] 
    
        # 将NumPy数组转回PIL Image  
        result_pil = Image.fromarray(pil_image_np).convert("RGB")  
    
        # 返回tensor形式的处理后的图像  
        return pil2tensor(result_pil),True 
    

class segformer_clothes:
   
    def __init__(self):
        pass
    
    # Labels: 0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {     
                 "image":("IMAGE", {"default": "","multiline": False}),
                 "Face": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Hat": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Hair": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Upper_clothes": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Skirt": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Pants": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Dress": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Belt": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "shoe": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "leg": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "arm": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Bag": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Scarf": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                }
        }

    RETURN_TYPES = ("IMAGE","BOOLEAN")
    RETURN_NAMES = ("mask_image","open")
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "CXH"

    def sample(self,image,Face,Hat,Hair,Upper_clothes,Skirt,Pants,Dress,Belt,shoe,leg,arm,Bag,Scarf):
        # seg切割结果，衣服pil
        pred_seg,cloth = get_segmentation(image)
        labels_to_keep = [0]
        # if background :
        #     labels_to_keep.append(0)
        if not Hat:
            labels_to_keep.append(1)
        if not Hair:
            labels_to_keep.append(2)
        if not Upper_clothes:
            labels_to_keep.append(4)
        if not Skirt:
            labels_to_keep.append(5)
        if not Pants:
            labels_to_keep.append(6)
        if not Dress:
            labels_to_keep.append(7)
        if not Belt:
            labels_to_keep.append(8)
        if not shoe:
            labels_to_keep.append(9)
            labels_to_keep.append(10)
        if not Face:
            labels_to_keep.append(11)
        if not leg:
            labels_to_keep.append(12)
            labels_to_keep.append(13)
        if not arm:
            labels_to_keep.append(14) 
            labels_to_keep.append(15) 
        if not Bag:
            labels_to_keep.append(16)
        if not Scarf:
            labels_to_keep.append(17)
            
        mask = np.isin(pred_seg, labels_to_keep).astype(np.uint8)
        
        # 创建agnostic-mask图像
        mask_image = Image.fromarray(mask * 255)
        mask_image = mask_image.convert("RGB")
        mask_image = pil2tensor(mask_image)
        # mask_r = pil2mask(mask_image)
        # mask_tensor = torch.from_numpy(mask).clone()

        return (mask_image,True)