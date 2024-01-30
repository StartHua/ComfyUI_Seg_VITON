# from .mine_nodes import *

from .utils.file_util import * 

import folder_paths
import os
import sys

# comfy_path = os.path.dirname(folder_paths.__file__)
# custom_nodes_path = os.path.join(comfy_path, "custom_nodes")
# import sys
vition_path = node_path("ComfyUI_Seg_VITON")
sys.path.append(vition_path)
# sys.path.append(os.path.join(vition_path,"cldm"))
# sys.path.append(os.path.join(vition_path,"ldm"))
# sys.path.append(os.path.join(vition_path,"ldm","data"))
sys.path.append(os.path.join(vition_path,"ldm","models"))
# sys.path.append(os.path.join(vition_path,"ldm","models","diffusion"))
# sys.path.append(os.path.join(vition_path,"ldm","models","diffusion","dpm_solver"))
# sys.path.append(os.path.join(vition_path,"ldm","modules"))
# sys.path.append(os.path.join(vition_path,"ldm","modules","diffusionmodules"))
# sys.path.append(os.path.join(vition_path,"ldm","modules","distributions"))
# sys.path.append(os.path.join(vition_path,"ldm","modules","encoders"))
# sys.path.append(os.path.join(vition_path,"ldm","modules","image_degradation"))
# sys.path.append(os.path.join(vition_path,"ldm","modules","image_degradation","utils"))
# sys.path.append(os.path.join(vition_path,"ldm","modules","image_encoders"))
# sys.path.append(os.path.join(vition_path,"ldm","modules","midas"))
# sys.path.append(os.path.join(vition_path,"ldm","modules","midas","midas"))
# sys.path.append(os.path.join(vition_path,"utils"))

from .segformer_clothes import *
from .stabel_vition import *

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "segformer_clothes":segformer_clothes,
    "segformer_agnostic":segformer_agnostic,
    "segformer_remove_bg":segformer_remove_bg,
    "stabel_vition":stabel_vition
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "segformer_clothes":"segformer_clothes",
    "segformer_agnostic":"segformer_agnostic",
    "segformer_remove_bg":"segformer_remove_bg",
    "stabel_vition":"stabel_vition"
}
