import folder_paths
import os

comfy_path = os.path.dirname(folder_paths.__file__)
custom_nodes_path = os.path.join(comfy_path, "custom_nodes")

# 节点路径
def node_path(node_name):
    return os.path.join(custom_nodes_path,node_name)

def checkpoints_path(node,mode):
    return os.path.join(node_path(node),"checkpoints",mode)


def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
