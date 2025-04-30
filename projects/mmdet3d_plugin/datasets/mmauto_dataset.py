import copy

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import Custom3DDataset
import mmcv
import os
from os import path as osp
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .nuscnes_eval import NuScenesEval_custom
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.parallel import DataContainer as DC
import random

from .nuscenes_dataset import CustomNuScenesDataset
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString
from mmdet.datasets.pipelines import to_tensor
import json
import cv2

@DATASETS.register_module()
class MMautoDataset(Custom3DDataset):
    r"""MMauto Dataset.

    use all eight cams.
    """
    MAPCLASSES = ('divider',)
    def __init__(self,
                 queue_length=4, 
                 bev_size=(200, 200), 
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 overlap_test=False, 
                 fixed_ptsnum_per_line=-1,
                 eval_use_same_gt_sample_num_flag=False,
                 padding_value=-10000,
                 map_classes=None,
                 noise='None',
                 noise_std=0,
                 aux_seg = dict(
                    use_aux_seg=False,
                    bev_seg=False,
                    pv_seg=False,
                    seg_classes=1,
                    feat_down_sample=32,
                 ),
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size

        self.MAPCLASSES = self.get_map_classes(map_classes)
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        self.pc_range = pc_range
        patch_h = pc_range[4]-pc_range[1]
        patch_w = pc_range[3]-pc_range[0]
        self.patch_size = (patch_h, patch_w)
        self.padding_value = padding_value
        self.fixed_num = fixed_ptsnum_per_line
        self.eval_use_same_gt_sample_num_flag = eval_use_same_gt_sample_num_flag
        self.aux_seg = aux_seg

        self.is_vis_on_test = False

    def load_annotations(self, ann_file):
        """
        加载多相机标注数据并按时间戳对齐（支持复杂文件夹命名和新的文件命名规则）
        
        参数:
            ann_file: 相机内参JSON文件路径
        
        返回:
            List[Dict]: 每个字典包含一个样本的所有相机信息，结构为：
            {
                'camera2ego': List[np.array(4,4)],  # 各相机的变换矩阵
                'camera_intrinsics': List[np.array(3,3)],  # 各相机的内参矩阵
                'img_filename': List[str]  # 各相机图片路径
            }
        """
        # 1. 定义相机类型及顺序（必须按此顺序输出）
        cam_keywords_order = [
            'front_mid', 'left_front', 'left_mid', 'left_rear',
            'rear_mid', 'right_front', 'right_mid', 'right_rear'
        ]
        num_cams = len(cam_keywords_order)

        # 2. 加载相机内参配置文件
        with open(ann_file, 'r') as f:
            cam_intrinsics_data = json.load(f)['camera_imx490_group_intrinsics']
        
        # 构建相机名称到内参的映射（去掉"_cam"后缀匹配）
        cam_intrinsics_map = {}
        for cam_data in cam_intrinsics_data:
            name = cam_data['device_name'].replace('_cam', '')
            intrinsics = cam_data['intrinsics']
            K = np.array([
                [intrinsics['fx'], 0, intrinsics['cx']],
                [0, intrinsics['fy'], intrinsics['cy']],
                [0, 0, 1]
            ], dtype=np.float32)
            cam_intrinsics_map[name] = K

        # 3. 扫描并匹配相机文件夹
        data_root = os.path.dirname(ann_file) + '/' 
        base_cam_path = osp.join(data_root, 'camera')
        all_dirs = [d for d in os.listdir(base_cam_path) 
                if osp.isdir(osp.join(base_cam_path, d))]
        
        cam_dir_map = {}
        for dir_name in all_dirs:
            lower_name = dir_name.lower()
            for keyword in cam_keywords_order:
                # 支持多种分隔符（下划线/连字符）的匹配
                if f'_{keyword}_' in lower_name or f'-{keyword}-' in lower_name:
                    cam_dir_map[keyword] = dir_name
                    break

        # 验证是否找到所有相机
        missing_cams = [k for k in cam_keywords_order if k not in cam_dir_map]
        if missing_cams:
            raise ValueError(f"缺少以下相机文件夹: {missing_cams}")

        # 4. 收集各相机文件并按时间戳排序
        all_cam_files = {}
        for keyword in cam_keywords_order:
            actual_dir = cam_dir_map[keyword]
            cam_path = osp.join(base_cam_path, actual_dir)
            
            # 获取所有pose_info.json文件
            json_files = [f for f in os.listdir(cam_path) 
                        if f.endswith('_pose_info.json') and not f.startswith('.')]
            
            # 提取时间戳并排序
            files_with_ts = []
            for f in json_files:
                try:
                    # 新命名规则：timestamp_pose_info.json
                    ts = int(f.split('_')[0])  # 提取时间戳数字部分
                    files_with_ts.append((ts, f))
                except (ValueError, IndexError):
                    continue
            
            # 按时间戳排序
            files_with_ts.sort(key=lambda x: x[0])
            all_cam_files[keyword] = {
                'dir': actual_dir,
                'files': [f[1] for f in files_with_ts]  # 保存排序后的json文件名
            }

        # 5. 确定最小公共样本数
        min_samples = min(len(v['files']) for v in all_cam_files.values())
        if min_samples == 0:
            raise ValueError("某些相机文件夹没有有效的样本文件")

        # 6. 构建样本列表
        samples = []
        for sample_idx in range(min_samples):
            sample_data = {
                'camera2ego': [],
                'camera_intrinsics': [],
                'img_filename': [],
                'lidar2ego': np.eye(4, dtype=np.float32), 
                # encoder.py中geom需要用到这个，暂且假定lidar坐标系与自车坐标系重合
                'lidar2img': [],
                'scene_token': 'mmauto',
                'can_bus': np.zeros((4, 1), dtype=np.float32) #随便写的，实际上没用上这个
            }

            for keyword in cam_keywords_order:
                cam_info = all_cam_files[keyword]
                json_file = cam_info['files'][sample_idx]
                
                # 从pose_info.json文件名构建undistorted.jpg文件名
                timestamp = json_file.split('_')[0]
                img_file = f"{timestamp}_undistorted.jpg"
                
                # 完整路径
                json_path = osp.join(base_cam_path, cam_info['dir'], json_file)
                img_path = osp.join(base_cam_path, cam_info['dir'], img_file)
                
                # 验证图片文件是否存在
                if not osp.exists(img_path):
                    raise FileNotFoundError(f"img not exist: {img_path}")

                # 加载JSON数据
                with open(json_path, 'r') as f:
                    cam_data = json.load(f)
                
                # 提取并转换外参矩阵（assum gnss=ego）
                gnss2cam = np.array(cam_data['gnss2cam_extrinsic'], 
                                dtype=np.float32).reshape(4, 4).T
                cam2ego = np.linalg.inv(gnss2cam)  # 求逆得到cam2ego
                
                # 获取内参矩阵
                K = cam_intrinsics_map[keyword]

                # 添加到样本数据（严格按cam_keywords_order顺序）
                sample_data['camera2ego'].append(cam2ego)
                sample_data['camera_intrinsics'].append(K)
                sample_data['img_filename'].append(img_path)

                # 这个数据暂时填充为全零, 实际没有用上
                sample_data['lidar2img'].append(np.zeros((4, 4), dtype=np.float32))

            
            samples.append(sample_data)
        
        return samples
    
    @classmethod
    def get_map_classes(cls, map_classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            list[str]: A list of class names.
        """
        if map_classes is None:
            return cls.MAPCLASSES

        if isinstance(map_classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(map_classes)
        elif isinstance(map_classes, (tuple, list)):
            class_names = map_classes
        else:
            raise ValueError(f'Unsupported type {type(map_classes)} of map classes.')

        return class_names

    def get_data_info(self, index):

        info = self.data_infos[index]

        return info

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)

        #这里面有用的：'camera2ego':len6的4*4array列表;'cam_intrinsic';'img_filename'等
        example = self.pipeline(input_dict) 
        #example:{'img_metas':[],'img':[]}  value都是DataContainer类构成的列表
        #img.shape=[6,3,480,800], 原本是900*1600，缩放为原来的1/2后，再将450填充至能整除32的480
        #相当于是self.pipeline将input_dict中的img_filename载出了图片，其它信息保存为'img_metas'

        return example

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data


@DATASETS.register_module()
class MMautoDataset_6cams(Custom3DDataset):
    r"""MMauto Dataset.

    just use 6 cams (ignore left_mid and right_mid)
    """
    MAPCLASSES = ('divider',)
    def __init__(self,
                 queue_length=4, 
                 bev_size=(200, 200), 
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 overlap_test=False, 
                 fixed_ptsnum_per_line=-1,
                 eval_use_same_gt_sample_num_flag=False,
                 padding_value=-10000,
                 map_classes=None,
                 noise='None',
                 noise_std=0,
                 aux_seg = dict(
                    use_aux_seg=False,
                    bev_seg=False,
                    pv_seg=False,
                    seg_classes=1,
                    feat_down_sample=32,
                 ),
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size

        self.MAPCLASSES = self.get_map_classes(map_classes)
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        self.pc_range = pc_range
        patch_h = pc_range[4]-pc_range[1]
        patch_w = pc_range[3]-pc_range[0]
        self.patch_size = (patch_h, patch_w)
        self.padding_value = padding_value
        self.fixed_num = fixed_ptsnum_per_line
        self.eval_use_same_gt_sample_num_flag = eval_use_same_gt_sample_num_flag
        self.aux_seg = aux_seg

        self.is_vis_on_test = False

    def load_annotations(self, ann_file):
        """
        加载多相机标注数据并按时间戳对齐（支持复杂文件夹命名和新的文件命名规则）
        
        参数:
            ann_file: 相机内参JSON文件路径
        
        返回:
            List[Dict]: 每个字典包含一个样本的所有相机信息，结构为：
            {
                'camera2ego': List[np.array(4,4)],  # 各相机的变换矩阵
                'camera_intrinsics': List[np.array(3,3)],  # 各相机的内参矩阵
                'img_filename': List[str]  # 各相机图片路径
            }
        """
        # 1. 定义相机类型及顺序（必须按此顺序输出）
        cam_keywords_order = [
            'front_mid', 'right_front', 'left_front',
            'rear_mid', 'left_rear', 'right_rear'
        ]
        num_cams = len(cam_keywords_order)

        # 2. 加载相机内参配置文件
        with open(ann_file, 'r') as f:
            cam_intrinsics_data = json.load(f)['camera_imx490_group_intrinsics']
        
        # 构建相机名称到内参的映射（去掉"_cam"后缀匹配）
        cam_intrinsics_map = {}
        for cam_data in cam_intrinsics_data:
            name = cam_data['device_name'].replace('_cam', '')
            intrinsics = cam_data['intrinsics']
            K = np.array([
                [intrinsics['fx'], 0, intrinsics['cx']],
                [0, intrinsics['fy'], intrinsics['cy']],
                [0, 0, 1]
            ], dtype=np.float32)
            cam_intrinsics_map[name] = K

        # 3. 扫描并匹配相机文件夹
        data_root = os.path.dirname(ann_file) + '/' 
        base_cam_path = osp.join(data_root, 'camera')
        all_dirs = [d for d in os.listdir(base_cam_path) 
                if osp.isdir(osp.join(base_cam_path, d))]
        
        cam_dir_map = {}
        for dir_name in all_dirs:
            lower_name = dir_name.lower()
            for keyword in cam_keywords_order:
                # 支持多种分隔符（下划线/连字符）的匹配
                if f'_{keyword}_' in lower_name or f'-{keyword}-' in lower_name:
                    cam_dir_map[keyword] = dir_name
                    break

        # 验证是否找到所有相机
        missing_cams = [k for k in cam_keywords_order if k not in cam_dir_map]
        if missing_cams:
            raise ValueError(f"缺少以下相机文件夹: {missing_cams}")

        # 4. 收集各相机文件并按时间戳排序
        all_cam_files = {}
        for keyword in cam_keywords_order:
            actual_dir = cam_dir_map[keyword]
            cam_path = osp.join(base_cam_path, actual_dir)
            
            # 获取所有pose_info.json文件
            json_files = [f for f in os.listdir(cam_path) 
                        if f.endswith('_pose_info.json') and not f.startswith('.')]
            
            # 提取时间戳并排序
            files_with_ts = []
            for f in json_files:
                try:
                    # 新命名规则：timestamp_pose_info.json
                    ts = int(f.split('_')[0])  # 提取时间戳数字部分
                    files_with_ts.append((ts, f))
                except (ValueError, IndexError):
                    continue
            
            # 按时间戳排序
            files_with_ts.sort(key=lambda x: x[0])
            all_cam_files[keyword] = {
                'dir': actual_dir,
                'files': [f[1] for f in files_with_ts]  # 保存排序后的json文件名
            }

        # 5. 确定最小公共样本数
        min_samples = min(len(v['files']) for v in all_cam_files.values())
        if min_samples == 0:
            raise ValueError("某些相机文件夹没有有效的样本文件")

        # 6. 构建样本列表
        samples = []
        for sample_idx in range(min_samples):
            sample_data = {
                'camera2ego': [],
                'camera_intrinsics': [],
                'img_filename': [],
                'lidar2ego': np.eye(4, dtype=np.float32), 
                # encoder.py中geom需要用到这个，暂且假定lidar坐标系与自车坐标系重合
                'lidar2img': [],
                'scene_token': 'mmauto',
                'can_bus': np.zeros((4, 1), dtype=np.float32) #随便写的，实际上没用上这个
            }

            for keyword in cam_keywords_order:
                cam_info = all_cam_files[keyword]
                json_file = cam_info['files'][sample_idx]
                
                # 从pose_info.json文件名构建undistorted.jpg文件名
                timestamp = json_file.split('_')[0]
                img_file = f"{timestamp}_undistorted.jpg"
                
                # 完整路径
                json_path = osp.join(base_cam_path, cam_info['dir'], json_file)
                img_path = osp.join(base_cam_path, cam_info['dir'], img_file)
                
                # 验证图片文件是否存在
                if not osp.exists(img_path):
                    raise FileNotFoundError(f"图片文件不存在: {img_path}")

                # 加载JSON数据
                with open(json_path, 'r') as f:
                    cam_data = json.load(f)
                
                # 提取并转换外参矩阵（gnss=ego）
                gnss2cam = np.array(cam_data['gnss2cam_extrinsic'], 
                                dtype=np.float32).reshape(4, 4).T
                cam2ego = np.linalg.inv(gnss2cam)  # 求逆得到cam2ego
                
                # 获取内参矩阵
                K = cam_intrinsics_map[keyword]

                # 添加到样本数据（严格按cam_keywords_order顺序）
                sample_data['camera2ego'].append(cam2ego)
                sample_data['camera_intrinsics'].append(K)
                sample_data['img_filename'].append(img_path)

                # 这个数据暂时填充为全零
                sample_data['lidar2img'].append(np.zeros((4, 4), dtype=np.float32))

            
            samples.append(sample_data)
        
        return samples
    
    @classmethod
    def get_map_classes(cls, map_classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            list[str]: A list of class names.
        """
        if map_classes is None:
            return cls.MAPCLASSES

        if isinstance(map_classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(map_classes)
        elif isinstance(map_classes, (tuple, list)):
            class_names = map_classes
        else:
            raise ValueError(f'Unsupported type {type(map_classes)} of map classes.')

        return class_names

    def get_data_info(self, index):

        info = self.data_infos[index]

        return info

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        example = self.pipeline(input_dict) 

        return example

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

