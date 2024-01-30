import os
import shutil
import numpy as np
import torchvision.transforms as transforms  
import cv2
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import torch
from importlib import import_module
from .cldm.model import create_model
from .cldm.plms_hacked import PLMSSampler
from .utils.utils import * 
from .utils.file_util import * 

vition_path = node_path("ComfyUI_Seg_VITON")
cache_dir = os.path.join(vition_path,"cache")

model_load_path = os.path.join( vition_path,"checkpoints/VITONHD.ckpt")
yaml_path = os.path.join(vition_path,"configs/VITON512_COMFYUI.yaml")

def tensor2img_seg(x):
    '''
    x : [BS x c x H x W] or [c x H x W]
    '''
    if x.ndim == 3:
        x = x.unsqueeze(0)
    BS, C, H, W = x.shape
    x = x.permute(0,2,3,1).reshape(-1, W, C).detach().cpu().numpy()
    x = np.clip(x, -1, 1)
    x = (x+1)/2
    x = np.uint8(x*255.0)
    if x.shape[-1] == 1:
        x = np.concatenate([x,x,x], axis=-1)
    return x

def imread(p, h, w, is_mask=False, in_inverse_mask=False, img=None):
    if img is None:
        img = cv2.imread(p)
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w,h))
        img = (img.astype(np.float32) / 127.5) - 1.0  # [-1, 1]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (w,h))
        img = (img >= 128).astype(np.float32)  # 0 or 1
        img = img[:,:,None]
        if in_inverse_mask:
            img = 1-img
    return img

  

class stabel_vition:
    def __init__(self):
        self.model = None
        self.sampler = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {     
                    "agn":("IMAGE", {"default": "","multiline": False}),
                    "agn_mask":("MASK", {"default": "","multiline": False}),
                    "cloth":("IMAGE", {"default": "","multiline": False}),
                    "image":("IMAGE", {"default": "","multiline": False}),
                    "image_densepose":("IMAGE", {"default": "","multiline": False}),
                    "img_H": ("INT", {"default": 512, "min": 268, "max": 2048}),
                    "img_W": ("INT", {"default": 384, "min": 268, "max": 2048}),
                    "denoise_steps": ("INT", {"default": 20, "min": 5, "max": 200}),
                    "batch_size": ("INT", {"default": 16, "min": 0, "max": 32, "step": 16}),
                    "eta": ("INT", {"default": 0, "min": 0, "max": 200}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "cache": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                    "repaint": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    
                }
        }
    RETURN_TYPES = ("IMAGE","BOOLEAN")
    RETURN_NAMES = ("image","open")
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "CXH" 
    def sample(self,agn,agn_mask,cloth,image,image_densepose,img_H,img_W,denoise_steps,batch_size,eta,seed,cache,repaint):
        seed = str(seed)
        img_fn = seed+"_img.jpg"
        cloth_fn = seed+"_cloth.jpg"
        #创建缓存文件夹 +缓存本地（待优化直接tensor转cv2）
        mkdir(cache_dir)
        agnostic_v3_2_dir = os.path.join(cache_dir,seed,"agnostic_v3_2")
        mkdir(agnostic_v3_2_dir)
        agnostic_v3_2_img_path =  os.path.join(agnostic_v3_2_dir,img_fn)
        save_tensor_image(agn,agnostic_v3_2_img_path)
        
        agnostic_mask_dir = os.path.join(cache_dir,seed,"agnostic_mask")
        mkdir(agnostic_mask_dir)
        agnostic_mask_img_path =  os.path.join(agnostic_mask_dir,img_fn)
        save_tensor_image(agn_mask,agnostic_mask_img_path)
        
        cloth_dir = os.path.join(cache_dir,seed,"cloth")
        mkdir(cloth_dir)
        cloth_img_path =  os.path.join(cloth_dir,img_fn)
        save_tensor_image(cloth,cloth_img_path)
        
        image_dir = os.path.join(cache_dir,seed,"image")
        mkdir(image_dir)
        image_img_path =  os.path.join(image_dir,img_fn)
        save_tensor_image(image,image_img_path)
        
        image_densepose_dir = os.path.join(cache_dir,seed,"image_densepose")
        mkdir(image_densepose_dir)
        image_densepose_img_path =  os.path.join(image_densepose_dir,img_fn)
        save_tensor_image(image_densepose,image_densepose_img_path)
        
        agn = imread(agnostic_v3_2_img_path, img_H, img_W)
        agn_mask = imread(agnostic_mask_img_path, img_H, img_W, is_mask=True, in_inverse_mask=True)
        cloth = imread(cloth_img_path, img_H, img_W)
        image = imread(image_img_path, img_H, img_W)
        image_densepose = imread(image_densepose_img_path, img_H, img_W)
    
        
        config = OmegaConf.load(yaml_path)
        config.model.params.img_H = img_H
        config.model.params.img_W = img_W
        params = config.model.params
        
        if  self.model == None:        
            self.model = create_model(config_path=None, config=config)
            self.model.load_state_dict(torch.load(model_load_path, map_location="cpu"))
            self.model = self.model.cuda()
            self.model.eval()
        
        if self.sampler == None:
            self.sampler = PLMSSampler(self.model)
            
        dataset = getattr(import_module("comyui_dataset"), config.dataset_name)(
            img_fn,
            cloth_fn,
            agn,
            agn_mask,
            cloth,
            image,
            image_densepose,
        )
        dataloader = DataLoader(dataset, num_workers=4, shuffle=False, batch_size=batch_size, pin_memory=True)
        
        shape = (4, img_H//8, img_W//8)
        x_sample_list =[] 
        
        for batch_idx, batch in enumerate(dataloader):
            print(f"{batch_idx}/{len(dataloader)}")
            z, c = self.model.get_input(batch, params.first_stage_key)
            bs = z.shape[0]
            c_crossattn = c["c_crossattn"][0][:bs]
            if c_crossattn.ndim == 4:
                c_crossattn = self.model.get_learned_conditioning(c_crossattn)
                c["c_crossattn"] = [c_crossattn]
            uc_cross = self.model.get_unconditional_conditioning(bs)
            uc_full = {"c_concat": c["c_concat"], "c_crossattn": [uc_cross]}
            uc_full["first_stage_cond"] = c["first_stage_cond"]
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            self.sampler.model.batch = batch

            ts = torch.full((1,), 999, device=z.device, dtype=torch.long)
            start_code = self.model.q_sample(z, ts)     

            samples, _, _ = self.sampler.sample(
                denoise_steps,
                bs,
                shape, 
                c,
                x_T=start_code,
                verbose=False,
                eta=eta,
                unconditional_conditioning=uc_full,
            )

            x_samples = self.model.decode_first_stage(samples)
            for sample_idx, (x_sample, fn,  cloth_fn) in enumerate(zip(x_samples, batch['img_fn'], batch["cloth_fn"])):
                x_sample_img = tensor2img_seg(x_sample)  
                x_sample_list.append(x_sample_img)
                if repaint:
                    repaint_agn_img = np.uint8((batch["image"][sample_idx].cpu().numpy()+1)/2 * 255)   # [0,255]
                    repaint_agn_mask_img = batch["agn_mask"][sample_idx].cpu().numpy()  # 0 or 1
                    x_sample_img = repaint_agn_img * repaint_agn_mask_img + x_sample_img * (1-repaint_agn_mask_img)
                    x_sample_img = np.uint8(x_sample_img)
                to_path =  os.path.join(cache_dir,seed,"result_"+str(sample_idx)+".jpg")
                cv2.imwrite(to_path, x_sample_img[:,:,::-1])
                      
        if not cache:
            shutil.rmtree(os.path.join(cache_dir,seed))
            
        return pil2tensor(x_sample_list[0]),True
    
