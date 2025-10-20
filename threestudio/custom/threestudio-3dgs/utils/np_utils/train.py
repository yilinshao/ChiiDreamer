import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
print(__name__)
from .dataset import Dataset
from .utils import CAPUDFNetwork
import argparse
import os
from shutil import copyfile
import numpy as np
import trimesh
from .extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import math
from pytorch3d.ops import knn_points
from scipy.spatial import cKDTree
import mcubes
import warnings
import torch.nn as nn

warnings.filterwarnings('ignore')


class Runner(nn.Module):
    def __init__(self, path, pointcloud=None, part_num=4, iteration=9000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device('cuda')
        self.part_num = part_num
        if pointcloud is not None:
            for i in range(1, self.part_num + 1):
                os.makedirs(os.path.join(path, 'query_data'), exist_ok=True)
                dataset_path = os.path.join(path, 'query_data', 'dataset_iter{}_part{}.pt'.format(iteration, i))
                if os.path.exists(dataset_path):
                    dataset = torch.load(dataset_path)
                else:
                    raise NotImplementedError
                self.__setattr__('dataset' + str(i), dataset)
        self.ChamferDisL1 = ChamferDistanceL1().cuda()

        # Networks
        for i in range(1, part_num+1):
            self.__setattr__('sdf_network'+str(i), CAPUDFNetwork().to(self.device))

    def reset_datasets(self, path, pointcloud, obj_name, iteration=9000, scene_name='Barn', save_dataset=True):
        for i in range(1, self.part_num + 1):
            os.makedirs(os.path.join(path, 'query_data'), exist_ok=True)
            dataset_path = os.path.join(path, 'query_data', 'dataset_{}_iter{}_part{}.pt'.format(obj_name, iteration, i))

            if os.path.exists(dataset_path):
                dataset = torch.load(dataset_path)
                print(f"Stored dataset found in {dataset_path}.")
            else:
                if not isinstance(pointcloud, np.ndarray):
                    pointcloud = pointcloud.clone().detach().cpu().numpy()
                if hasattr(self, 'dataset' + str(i)):
                    old_datset = getattr(self, 'dataset' + str(i))
                    pc_bbx = old_datset.pc_bbx
                    shape_center = old_datset.shape_center
                    shape_scale = old_datset.shape_scale
                else:
                    pc_bbx = shape_center = shape_scale = None
                dataset = Dataset(pointcloud, part=i, scene_name=scene_name, old_pc_bbx=pc_bbx,
                                        old_shape_center=shape_center, old_shape_scale=shape_scale)
                if save_dataset:
                    torch.save(dataset, dataset_path)
            self.__setattr__('dataset' + str(i), dataset)


    def get_learning_rate_at_iteration(self, iter_step, max_iter=60050):
        warn_up = 100
        init_lr = 0.001
        lr = (iter_step / warn_up) if iter_step < warn_up else 0.5 * (
                math.cos((iter_step - warn_up) / (max_iter - warn_up) * math.pi) + 1)
        lr = lr * init_lr
        return lr

    def load_checkpoint(self, path, checkpoint_name):
        checkpoint = torch.load(os.path.join(path, 'checkpoints', checkpoint_name),
                                map_location=self.device)
        print(os.path.join(path, 'checkpoints', checkpoint_name))
        for i in range(1, self.part_num+1):
            sdf_network = self.__getattribute__('sdf_network'+str(i))
            sdf_network.load_state_dict(checkpoint['sdf_network'+str(i)])

    def load_from_state_dict(self, state_dict):
        for i in range(1, self.part_num + 1):
            sdf_network_i_state = {}
            for name, param in state_dict.items():
                if f'sdf_network{i}' in name:
                    param_name = name.split(f'sdf_network{i}.')[-1]
                    sdf_network_i_state[param_name] = param
            sdf_network = getattr(self, 'sdf_network' + str(i))
            sdf_network.load_state_dict(sdf_network_i_state)

    def save_checkpoint(self, path, iter_step):
        checkpoint = {}
        for i in range(1, self.part_num+1):
            sdf_network = self.__getattribute__('sdf_network'+str(i))
            checkpoint.update({f'sdf_network{i}': sdf_network.state_dict()})

        os.makedirs(os.path.join(path, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(path, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(iter_step)))


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/ndf.conf')
    parser.add_argument('--mcube_resolution', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dir', type=str, default='test')
    parser.add_argument('--dataname', type=str, default='demo')
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args, args.conf)
    # runner.load_checkpoint('ckpt_100000.pth')
    # runner.iter_step = 100000-1

    runner.train()
    # runner.extract_mesh_meshudf(resolution=512, dist_threshold_ratio=1.0)
    # runner.extract_mesh(resolution=512)
    for threshold in [0.001, 0.002, 0.003, 0.004, 0.005]:
        runner.extract_mesh_sdf(resolution=512, threshold=threshold)
