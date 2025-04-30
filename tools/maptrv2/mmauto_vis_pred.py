import argparse
import mmcv
import os
import shutil
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet3d.utils import collect_env, get_root_logger
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
import sys
sys.path.append('')
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Rectangle
import cv2
import re

CAMS = ['left_front','front_mid','right_front',
        'left_mid',              'right_mid',
        'left_rear', 'rear_mid', 'right_rear']

def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords

def parse_args():
    parser = argparse.ArgumentParser(description='vis hdmaptr map gt label')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--score-thresh', default=0.4, type=float, help='samples to visualize')
    parser.add_argument(
        '--show-dir', help='directory where visualizations will be saved')
    parser.add_argument('--show-cam', action='store_true', help='show camera pic')
    parser.add_argument(
        '--gt-format',
        type=str,
        nargs='+',
        default=['fixed_num_pts',],
        help='vis format, default should be "points",'
        'support ["se_pts","bbox","fixed_num_pts","polyline_pts"]')
    args = parser.parse_args()
    return args

def extract_camera_name(filepath):
    """
    从文件路径中提取cam_和_cam中间的字符串
    
    参数:
        filepath: 包含相机名称的路径字符串
        
    返回:
        cam_和_cam中间的字符串，如果找不到则返回None
    """
    # 使用正则表达式匹配模式
    pattern = r'cam_(.*?)_cam'
    match = re.search(pattern, filepath)
    
    if match:
        return match.group(1)
    else:
        return None
    
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # cfg.xxx 返回 cfg._cfg_dict 中的内容

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):  # 会进入这个循环
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.show_dir is None:
        args.show_dir = osp.join('./work_dirs', 
                                osp.splitext(osp.basename(args.config))[0],
                                'vis_pred')
    # create vis_label dir
    mmcv.mkdir_or_exist(osp.abspath(args.show_dir))
    cfg.dump(osp.join(args.show_dir, osp.basename(args.config)))
    logger = get_root_logger()
    logger.info(f'DONE create vis_pred dir: {args.show_dir}')


    dataset = build_dataset(cfg.data.test) #plugin/datasets/nucenes_offlinemap_dataset.py/CustomN..Off
    #这里是我们要改的

    dataset.is_vis_on_test = True #TODO, this is a hack
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        # workers_per_gpu=cfg.data.workers_per_gpu,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    logger.info('Done build test data set')

    # build the model and load checkpoint
    # import pdb;pdb.set_trace()
    cfg.model.train_cfg = None
    # cfg.model.pts_bbox_head.bbox_coder.max_num=15 # TODO this is a hack
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # 这个会跳到/MapTR/projects/mmdet3d_plugin/maptr/detectors/maptrv2.py

    print(cfg.get('test_cfg')) # None

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    logger.info('loading check point')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    logger.info('DONE load check point')
    model = MMDataParallel(model, device_ids=[6])
    model.eval()

    img_norm_cfg = cfg.img_norm_cfg

    # get denormalized param
    mean = np.array(img_norm_cfg['mean'],dtype=np.float32)
    std = np.array(img_norm_cfg['std'],dtype=np.float32)
    to_bgr = img_norm_cfg['to_rgb']

    # get pc_range
    pc_range = cfg.point_cloud_range
    #pc_range=[-15,-30,-10,15,30,10],实际上-10与10并没有用上

    # get car icon
    car_img = Image.open('./figs/lidar_car.png')
    #这个是地图中间的红色小车

    # get color map: divider->r, ped->b, boundary->g
    colors_plt = ['orange', 'b', 'r', 'g']


    logger.info('BEGIN vis test dataset samples gt label & pred')

    dataset = data_loader.dataset

    # prog_bar = mmcv.ProgressBar(len(CANDIDATE))
    prog_bar = mmcv.ProgressBar(len(dataset))
    # import pdb;pdb.set_trace()
    for i, data in enumerate(data_loader):   #后面的全在这个for循环里面
       
        img = data['img'][0].data[0] #[1,6,3,480,800]
        img_metas = data['img_metas'][0].data[0]
        #data字典里面比较有用的就这两个，尝试将这两个写好。还有两个是'gt_labels_3d','gt_bboxes_3d'.
        #img_metas信息见思源

        # 选择front_mid_cam中的图片命名
        pts_filename = img_metas[0]['filename'][0]
        pts_filename = osp.basename(pts_filename)
        pts_filename = pts_filename.replace('_undistorted', '').split('.')[0]

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # 这里得到了推理结果
        sample_dir = osp.join(args.show_dir, pts_filename)
        mmcv.mkdir_or_exist(osp.abspath(sample_dir))

        filename_list = img_metas[0]['filename']

        # save cam img for sample
        for filepath in filename_list:

            filename = extract_camera_name(filepath)
            img_name = filename + '.jpg'
            img_path = osp.join(sample_dir,img_name)
            shutil.copyfile(filepath,img_path)
         
        # surrounding view
        cam_images = []
        for cam in CAMS:
            cam_img_name = cam + '.jpg'
            cam_img = cv2.imread(osp.join(sample_dir, cam_img_name))
            if cam_img is None:
                raise FileNotFoundError(f"can't find img: {osp.join(sample_dir, cam_img_name)}")
            cam_images.append(cam_img)
        h, w = cam_images[0].shape[:2]
        black_img = np.zeros((h, w, 3), dtype=np.uint8)
        row1 = cv2.hconcat([cam_images[0], cam_images[1], cam_images[2]])  # 上排
        row2 = cv2.hconcat([cam_images[3], black_img, cam_images[4]])     # 中排（中间留黑）
        row3 = cv2.hconcat([cam_images[5], cam_images[6], cam_images[7]])  # 下排
        surround_view = cv2.vconcat([row1, row2, row3])
        output_path = osp.join(sample_dir, 'SURROUND_VIEW.jpg')
        cv2.imwrite(output_path, surround_view, [cv2.IMWRITE_JPEG_QUALITY, 70])
        
        
        # import pdb;pdb.set_trace()
        plt.figure(figsize=(2, 4))
        plt.xlim(pc_range[0], pc_range[3])
        plt.ylim(pc_range[1], pc_range[4])
        plt.axis('off')

        # visualize pred
        # import pdb;pdb.set_trace()
        result_dic = result[0]['pts_bbox'] # 这个是maptrv2中的simple_test_pts中的bbox_results
        boxes_3d = result_dic['boxes_3d'] # bbox: xmin, ymin, xmax, ymax, shape=[50,4]
        scores_3d = result_dic['scores_3d'] # shape=[50]
        labels_3d = result_dic['labels_3d'] # shape=[50]
        pts_3d = result_dic['pts_3d'] # shape=[50,30,2]
        keep = scores_3d > args.score_thresh 

        plt.figure(figsize=(2, 4))
        plt.xlim(pc_range[0], pc_range[3])
        plt.ylim(pc_range[1], pc_range[4])
        plt.axis('off')
        for pred_score_3d, pred_bbox_3d, pred_label_3d, pred_pts_3d in zip(scores_3d[keep], boxes_3d[keep],labels_3d[keep], pts_3d[keep]):
            # []               [4]            []            [20,2]
            pred_pts_3d = pred_pts_3d.numpy()
            pts_x = pred_pts_3d[:,0]
            pts_y = pred_pts_3d[:,1]
            plt.plot(pts_x, pts_y, color=colors_plt[pred_label_3d],linewidth=1,alpha=0.8,zorder=-1)
            plt.scatter(pts_x, pts_y, color=colors_plt[pred_label_3d],s=1,alpha=0.8,zorder=-1)

            # 这里的bbox根本就没用上？
            pred_bbox_3d = pred_bbox_3d.numpy()
            xy = (pred_bbox_3d[0],pred_bbox_3d[1])
            width = pred_bbox_3d[2] - pred_bbox_3d[0]
            height = pred_bbox_3d[3] - pred_bbox_3d[1]


            pred_score_3d = float(pred_score_3d)
            pred_score_3d = round(pred_score_3d, 2)
            s = str(pred_score_3d)

        plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
        # 显示中间的红色小车

        map_path = osp.join(sample_dir, 'PRED_MAP.png')
        plt.savefig(map_path, bbox_inches='tight', format='png',dpi=1200)
        plt.close()

        prog_bar.update()

    logger.info('\n DONE vis test dataset samples gt label & pred')
if __name__ == '__main__':
    main()
