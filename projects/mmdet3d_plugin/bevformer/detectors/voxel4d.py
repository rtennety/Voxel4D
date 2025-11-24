# Voxel4D: Vision-Centric 4D Occupancy Forecasting and Planning via World Models for Autonomous Driving
# Author: Rohan Tennety

import mmcv
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
import copy
import numpy as np
import os
import torch.nn.functional as F
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.bevformer.losses.plan_reg_loss_lidar import plan_reg_loss
from projects.mmdet3d_plugin.bevformer.utils.metric_stp3 import PlanningMetric
from projects.mmdet3d_plugin.bevformer.utils.planning_metrics import PlanningMetric_v2
from torchvision.transforms.functional import rotate

from .bevformer import BEVFormer
from mmdet3d.models import builder
from ..utils import e2e_predictor_utils


@DETECTORS.register_module()
class Voxel4D(BEVFormer):
    def __init__(self,
                 # Future predictions.
                 future_pred_head,
                 turn_on_flow,
                 future_pred_frame_num,  # number of future prediction frames.
                 test_future_frame_num,  # number of future prediction frames when testing.

                 # BEV configurations.
                 point_cloud_range,
                 bev_h,
                 bev_w,

                 # Plan Head configurations.
                 turn_on_plan=False,
                 plan_head=None,

                 # Memory Queue configurations.
                 memory_queue_len=1,

                 # Augmentations.
                 # A1. randomly drop current image (to enhance temporal feature.)
                 random_drop_image_rate=0.0,
                 # A2. add noise to previous_bev_queue.
                 random_drop_prev_rate=0.0,
                 random_drop_prev_start_idx=1,
                 random_drop_prev_end_idx=None,
                 # A3. grid mask augmentation.
                 grid_mask_image=True,
                 grid_mask_backbone_feat=False,
                 grid_mask_fpn_feat=False,
                 grid_mask_prev=False,
                 grid_mask_cfg=dict(
                     use_h=True,
                     use_w=True,
                     rotate=1,
                     offset=False,
                     ratio=0.5,
                     mode=1,
                     prob=0.7
                 ),

                 # Supervision.
                 only_generate_dataset=False,
                 supervise_all_future=True,

                 _viz_pcd_flag=False,
                 _viz_pcd_path='dbg/pred_pcd',  # root/{prefix}

                 *args,
                 **kwargs,):

        super().__init__(*args, **kwargs)
        # occ head
        self.future_pred_head = builder.build_head(future_pred_head)
        # flow head
        self.turn_on_flow = turn_on_flow
        if self.turn_on_flow:
            future_pred_head_flow = future_pred_head
            future_pred_head_flow['num_classes'] = 3
            future_pred_head_flow['turn_on_flow'] = True
            future_pred_head_flow['prev_render_neck']['occ_flow'] = 'flow'
            self.future_pred_head_flow = builder.build_head(future_pred_head_flow)
            if self.future_pred_head.num_classes == 2:      # GMO
                self.vehicles_id = [1]
            elif self.future_pred_head.num_classes == 9:    # nus_MMO
                self.vehicles_id = [1,2,3,4,5,6,7,8]
            elif self.future_pred_head.num_classes == 17:   # nus_finegrained
                self.vehicles_id = [2,3,4,5,6,7,9,10]
            self.gmo_id = 1 # sem_clsses -> GMO
            self.iou_thresh_for_vpq = 0.2
        
        # plan head
        self.turn_on_plan = turn_on_plan
        if turn_on_plan:
            self.plan_head = builder.build_head(plan_head)
            self.plan_head_type = plan_head.type
            self.planning_metric = None
            self.planning_metric_v2 = PlanningMetric_v2(n_future=future_pred_frame_num+1)
        
        # memory queue
        self.memory_queue_len = memory_queue_len


        self.future_pred_frame_num = future_pred_frame_num
        self.test_future_frame_num = test_future_frame_num
        # if not predict any future,
        #  then only predict current frame.
        self.only_train_cur_frame = (future_pred_frame_num == 0)

        self.point_cloud_range = point_cloud_range
        self.bev_h = bev_h
        self.bev_w = bev_w

        # Augmentations.
        self.random_drop_image_rate = random_drop_image_rate
        self.random_drop_prev_rate = random_drop_prev_rate
        self.random_drop_prev_start_idx = random_drop_prev_start_idx
        self.random_drop_prev_end_idx = random_drop_prev_end_idx

        # Grid mask.
        self.grid_mask_image = grid_mask_image
        self.grid_mask_backbone_feat = grid_mask_backbone_feat
        self.grid_mask_fpn_feat = grid_mask_fpn_feat
        self.grid_mask_prev = grid_mask_prev
        self.grid_mask = GridMask(**grid_mask_cfg)

        # Training configurations.
        # randomly sample one future for loss computation?
        self.only_generate_dataset = only_generate_dataset
        self.supervise_all_future = supervise_all_future

        self._viz_pcd_flag = _viz_pcd_flag
        self._viz_pcd_path = _viz_pcd_path

        # remove the useless modules in pts_bbox_head
        #  * box/cls prediction head; decoder transformer.
        del self.pts_bbox_head.cls_branches, self.pts_bbox_head.reg_branches
        del self.pts_bbox_head.query_embedding
        del self.pts_bbox_head.transformer.decoder

        if self.only_train_cur_frame:
            # remove useless parameters.
            del self.future_pred_head.transformer
            del self.future_pred_head.bev_embedding
            del self.future_pred_head.prev_frame_embedding
            del self.future_pred_head.can_bus_mlp
            del self.future_pred_head.positional_encoding
            del self.future_pred_head_flow.transformer
            del self.future_pred_head_flow.bev_embedding
            del self.future_pred_head_flow.prev_frame_embedding
            del self.future_pred_head_flow.can_bus_mlp
            del self.future_pred_head_flow.positional_encoding

    def set_epoch(self, epoch):
        self.training_epoch = epoch

    ####################### Image Feature Extraction. #######################
    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        if ('aug_param' in img_metas[0] and
                img_metas[0]['aug_param'] is not None and
                img_metas[0]['aug_param']['CropResizeFlipImage_param'][-1] is True):
            img_feats = [torch.flip(x, dims=[-1, ]) for x in img_feats]

        return img_feats

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask and self.grid_mask_image:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
            if self.use_grid_mask and self.grid_mask_backbone_feat:
                new_img_feats = []
                for img_feat in img_feats:
                    img_feat = self.grid_mask(img_feat)
                    new_img_feats.append(img_feat)
                img_feats = new_img_feats
        else:
            return None

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
            if self.use_grid_mask and self.grid_mask_fpn_feat:
                new_img_feats = []
                for img_feat in img_feats:
                    img_feat = self.grid_mask(img_feat)
                    new_img_feats.append(img_feat)
                img_feats = new_img_feats

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped


    ############# Align coordinates between reference (current frame) to other frames. #############
    def _get_history_ref_to_previous_transform(self, tensor, num_frames, prev_img_metas, ref_img_metas):
        """Get transformation matrix from reference frame to all previous frames.

        Args:
            tensor: to convert {ref_to_prev_transform} to device and dtype.
            num_frames: total num of available history frames.
            img_metas_list: a list of batch_size items.
                In each item, there is {num_prev_frames} img_meta for transformation alignment.

        Return:
            ref_to_history_list (torch.Tensor): with shape as [bs, num_prev_frames, 4, 4]
        """
        ref_num_frames = 1
        history_num_frames = num_frames - ref_num_frames
        # history
        ref_to_history_list = []
        for img_metas in prev_img_metas:
            img_metas_len = len(img_metas)
            cur_ref_to_prev = [img_metas[i]['ref_lidar_to_cur_lidar'] for i in range(img_metas_len-history_num_frames, img_metas_len)]   # [-3,-2,-1]
            ref_to_history_list.append(cur_ref_to_prev)
        ref_to_history_list = tensor.new_tensor(np.array(ref_to_history_list))
        # ref
        ref_to_ref_list = []
        for img_metas in ref_img_metas:
            cur_ref_to_prev = [img_metas[i]['ref_lidar_to_cur_lidar'] for i in range(ref_num_frames)]
            ref_to_ref_list.append(cur_ref_to_prev)
        ref_to_ref_list = tensor.new_tensor(np.array(ref_to_ref_list))
        # concat
        if ref_to_history_list.shape[1] == 0:   # not use history
            ref_to_history_list = ref_to_ref_list
        else:
            ref_to_history_list = torch.cat([ref_to_history_list, ref_to_ref_list], dim=1)
        return ref_to_history_list

    def _align_bev_coordnates(self, frame_idx, ref_to_history_list, img_metas, plan_traj):
        """Align the bev_coordinates of frame_idx to each of history_frames.

        Args:
            frame_idx: the index of target frame.
            ref_to_history_list (torch.Tensor): a tensor with shape as [bs, num_prev_frames, 4, 4]
                indicating the transformation metric from reference to each history frames.
            img_metas: a list of batch_size items.
                In each item, there is one img_meta (reference frame)
                whose {future2ref_lidar_transform} & {ref2future_lidar_transform} are for
                transformation alignment.
        """
        bs, num_frame = ref_to_history_list.shape[:2]
        translation_xy = torch.cumsum(plan_traj, dim=1)[:, -1, :2].float()

        # 1. get future2ref and ref2future_matrix of frame_idx.
        future2ref = [img_meta['future2ref_lidar_transform'][frame_idx] for img_meta in img_metas]
        future2ref = ref_to_history_list.new_tensor(np.array(future2ref))
        # use translation_xy
        if self.future_pred_head.use_plan_traj:
            future2ref = future2ref.transpose(-1, -2)
            future2ref[:, :2, 3] = translation_xy
            future2ref = future2ref.transpose(-1, -2)
            future2ref = future2ref.detach().clone()

        ref2future = [img_meta['ref2future_lidar_transform'][frame_idx] for img_meta in img_metas]
        ref2future = ref_to_history_list.new_tensor(np.array(ref2future))
        # use translation_xy
        if self.future_pred_head.use_plan_traj:
            ref2future = ref2future.transpose(-1, -2)
            rot = ref2future[:, :3, :3]
            translation_xyz = future2ref[:, 3, :3].unsqueeze(2)     # cur2ref
            translation_xyz = -(rot @ translation_xyz).squeeze(2)   # ref2cur
            ref2future[:, :3, 3] = translation_xyz
            ref2future = ref2future.transpose(-1, -2)
            ref2future = ref2future.detach().clone()

        # 2. compute the transformation matrix from current frame to all previous frames.
        future2ref = future2ref.unsqueeze(1).repeat(1, num_frame, 1, 1).contiguous()
        future_to_history_list = torch.matmul(future2ref, ref_to_history_list)

        # 3. compute coordinates of future frame.
        bev_grids = e2e_predictor_utils.get_bev_grids(
            self.bev_h, self.bev_w, bs * num_frame)
        bev_grids = bev_grids.view(bs, num_frame, -1, 2)
        bev_coords = e2e_predictor_utils.bev_grids_to_coordinates(
            bev_grids, self.point_cloud_range)

        # 4. align target coordinates of future frame to each of previous frames.
        aligned_bev_coords = torch.cat([
            bev_coords, torch.ones_like(bev_coords[..., :2])], -1)
        aligned_bev_coords = torch.matmul(aligned_bev_coords, future_to_history_list)
        aligned_bev_coords = aligned_bev_coords[..., :2]
        aligned_bev_grids, _ = e2e_predictor_utils.bev_coords_to_grids(
            aligned_bev_coords, self.bev_h, self.bev_w, self.point_cloud_range)
        aligned_bev_grids = (aligned_bev_grids + 1) / 2.  # range of [0, 1]
        # b, h*w, num_frame, 2
        aligned_bev_grids = aligned_bev_grids.permute(0, 2, 1, 3).contiguous()

        # 5. get target bev_grids at target future frame.
        tgt_grids = bev_grids[:, -1].contiguous()
        return tgt_grids, aligned_bev_grids, ref2future, future_to_history_list.transpose(-1, -2)
    

    def obtain_ref_bev(self, img, img_metas, prev_bev):
        # Extract current BEV features.
        # C1. Forward.
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None

        # C3. BEVFormer Encoder Forward.
        # ref_bev: bs, bev_h * bev_w, c
        ref_bev = self.pts_bbox_head(img_feats, img_metas, prev_bev, only_bev=True)
        return ref_bev
    
    def obtain_ref_bev_with_plan(self, img, img_metas, prev_bev, ref_sample_traj, ref_sem_occupancy, ref_command, ref_real_traj=None):
        # Extract current BEV features.
        # C1. Forward.
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None

        # C2. BEVFormer Encoder Forward.
        # ref_bev: bs, bev_h * bev_w, c
        ref_bev = self.pts_bbox_head(img_feats, img_metas, prev_bev, only_bev=True)

        # C3. PlanHead 
        if 'v1' in self.plan_head_type:
            if ref_sem_occupancy is None:   # use pred_occupancy to calculate sample_traj cost during inference, GT_occupancy during training
                ref_sem_occupancy = self.future_pred_head.forward_head(ref_bev.unsqueeze(0).unsqueeze(0))[-1, -1, 0].argmax(-1).detach()
                bs, hw, d = ref_sem_occupancy.shape
                ref_sem_occupancy = ref_sem_occupancy.view(bs, self.bev_w, self.bev_h, d).transpose(1,2)
            ref_pose_pred, ref_pose_loss = self.plan_head(ref_bev, ref_sample_traj, ref_sem_occupancy, ref_command, ref_real_traj)
        elif 'v2' in self.plan_head_type:
            ref_pose_pred = self.plan_head(ref_bev, ref_command)
            ref_pose_loss = None

        return ref_bev, ref_pose_pred, ref_pose_loss

    
    def future_pred(self, prev_bev_input, action_condition_dict, cond_norm_dict, plan_dict, 
                    valid_frames, img_metas, prev_img_metas, num_frames, occ_flow='occ'):
        if occ_flow == 'occ':
            future_pred_head = self.future_pred_head
        elif occ_flow == 'flow':
            future_pred_head = self.future_pred_head_flow
        else:
            AssertionError('Not Implemented')
        
        # D1. preparations.
        # prev_bev_input: B,memory_queue_len,HW,C
        ref_bev = prev_bev_input[:, -1].unsqueeze(0).repeat(
                len(self.future_pred_head.bev_pred_head), 1, 1, 1).contiguous()

        next_bev_feats, next_bev_sem, next_pose_loss = [ref_bev], [], []
        next_pose_preds = plan_dict['ref_pose_pred'] # B,Lout,2


        # D2. Align previous frames to the reference coordinates.
        ref_img_metas = [[each[num_frames-1]] for each in prev_img_metas]
        prev_img_metas = [[each[i] for i in range(num_frames-1)] for each in prev_img_metas]
        ref_to_history_list = self._get_history_ref_to_previous_transform(
            prev_bev_input, prev_bev_input.shape[1], prev_img_metas, ref_img_metas)


        # D3. future decoder forward.
        if self.training:
            future_frame_num = self.future_pred_frame_num
        else:
            future_frame_num = self.test_future_frame_num

        for future_frame_index in range(1, future_frame_num + 1):
            if (not self.turn_on_plan) or (self.turn_on_plan and self.training and self.training_epoch < 12):   # use GT planning during training
                plan_traj = plan_dict['gt_traj'][:, :future_frame_index, :2]
            else:
                plan_traj = next_pose_preds
            action_condition_dict['plan_traj'] = plan_traj

            # 1. obtain the coordinates of future BEV query to previous frames.
            tgt_grids, aligned_prev_grids, ref2future, future2history = self._align_bev_coordnates(
                future_frame_index, ref_to_history_list, img_metas, plan_traj)
            cond_norm_dict['future2history'] = future2history


            # 2. transform for generating freespace of future frame.
            # pred_feat: inter_num, bs, bev_h * bev_w, c
            if future_frame_index in valid_frames:  # compute loss if it is a valid frame.
                pred_feat, bev_sem_pred = future_pred_head(
                    prev_bev_input, img_metas, future_frame_index, action_condition_dict, cond_norm_dict,
                    tgt_points=tgt_grids, bev_h=self.bev_h, bev_w=self.bev_w, ref_points=aligned_prev_grids)
                next_bev_feats.append(pred_feat)
                next_bev_sem.append(bev_sem_pred)
            else:
                with torch.no_grad():
                    pred_feat, bev_sem_pred = future_pred_head(
                        prev_bev_input, img_metas, future_frame_index, action_condition_dict, cond_norm_dict,
                        tgt_points=tgt_grids, bev_h=self.bev_h, bev_w=self.bev_w, ref_points=aligned_prev_grids)
                    next_bev_feats.append(pred_feat)


            # 3. Planning based on semantic occupancy.
            if self.turn_on_plan and occ_flow == 'occ':
                # sample_traj  gt_traj
                sample_traj_i,  gt_traj_i = plan_dict['sample_traj'][:,:,future_frame_index], plan_dict['gt_traj'][:,future_frame_index]
                # command
                command_i = action_condition_dict['command'][:,future_frame_index]
                # forward plan_head 
                if 'v1' in self.plan_head_type:    # used for fine-grained_MMO when sem_occupancy distinguish categories in MMO
                    # sem_occupancy
                    if plan_dict['sem_occupancy'] is None:   # use_pred
                        sem_occupancy_i = future_pred_head.forward_head(pred_feat.unsqueeze(0))[-1, -1, 0].argmax(-1).detach()
                        bs, hw, d = sem_occupancy_i.shape
                        sem_occupancy_i = sem_occupancy_i.view(bs, self.bev_w, self.bev_h, d).transpose(1,2)
                    else:   # use_gt  traning_epoch < 12
                        sem_occupancy_i = plan_dict['sem_occupancy'][:,future_frame_index]
                    pose_pred, pose_loss = self.plan_head(pred_feat[-1], sample_traj_i, sem_occupancy_i, command_i, gt_traj_i)
                    # update prev_pose and store pred
                    next_pose_preds = torch.cat([next_pose_preds, pose_pred], dim=1)
                    next_pose_loss.append(pose_loss)
                elif 'v2' in self.plan_head_type:   # used for inflated_GMO when sem_occupancy does not distinguish categories in GMO
                    pose_pred = self.plan_head(pred_feat[-1], command_i)
                    next_pose_preds = torch.cat([next_pose_preds, pose_pred], dim=1)


            # 4. update pred_feat to prev_bev_input and update ref_to_history_list.
            prev_bev_input = torch.cat([prev_bev_input, pred_feat[-1].unsqueeze(1)], 1)
            prev_bev_input = prev_bev_input[:, 1:, ...].contiguous()
            # update ref2future to ref_to_history_list.
            ref_to_history_list = torch.cat([ref_to_history_list, ref2future.unsqueeze(1)], 1)
            ref_to_history_list = ref_to_history_list[:, 1:].contiguous()
            # update occ_gts
            if cond_norm_dict['occ_gts'] is not None:
                cond_norm_dict['occ_gts'] = cond_norm_dict['occ_gts'][:, 1:, ...].contiguous()


        # D4. forward head.
        next_bev_feats = torch.stack(next_bev_feats, 0)
        # forward head
        next_bev_preds = future_pred_head.forward_head(next_bev_feats)

        return next_bev_preds, next_bev_sem, next_pose_preds, next_pose_loss


    def compute_occ_loss(self, occ_preds, occ_gts):
        # preds
        occ_preds = occ_preds.permute(1, 0, 3, 2, 6, 4, 5).squeeze(3)
        inter_num, select_frames, bs, num_cls, hw, d = occ_preds.shape
        occ_preds = occ_preds.view(inter_num, select_frames*bs, num_cls, self.bev_w, self.bev_h, d).transpose(3,4)
        # gts
        occ_gts = occ_gts[0][self.future_pred_head.history_queue_length:]
        occ_gts = occ_gts.view(select_frames*bs, *occ_gts.shape[-3:])
        
        # occ loss
        losses_occupancy = self.future_pred_head.loss_occ(occ_preds, occ_gts)
        return losses_occupancy
    
    def compute_sem_norm_loss(self, bev_sem_preds, occ_gts):
        # gts
        occ_gts = occ_gts[0][self.future_pred_head.history_queue_length:-1]

        loss_dict = {}
        # loss sem
        if bev_sem_preds[0] is not None:
            bev_sem_preds = torch.stack(bev_sem_preds, dim=0).transpose(0,1)
            loss_sem_norm = self.future_pred_head.loss_sem_norm(bev_sem_preds, occ_gts)
        return loss_sem_norm

    def compute_sem_norm(self, bev_sem_preds, occ_gts):
        # gts
        occ_gts = occ_gts[0][self.future_pred_head.history_queue_length:]

        # loss sem
        if bev_sem_preds[0] is not None:
            bev_sem_preds = torch.stack(bev_sem_preds, dim=0).permute(1,2,0,3,4,5,6).flatten(1,2)
            loss_sem_norm = self.future_pred_head.loss_sem_norm(bev_sem_preds, occ_gts)
        return loss_sem_norm

    def compute_obj_motion_norm(self, flow_preds, flow_gts):
        # gts
        flow_gts = flow_gts[0][self.future_pred_head_flow.history_queue_length:]

        # preds
        if flow_preds[0] is not None:
            flow_preds = torch.stack(flow_preds, dim=0).permute(1,2,0,3,4,5,6).flatten(1,2)
            losses_flow = self.future_pred_head_flow.loss_obj_motion_norm(flow_preds, flow_gts)
        return losses_flow

    def get_one_hot(self, label, N):
        size = list(label.size())
        label = label.view(-1)
        ones = torch.sparse.torch.eye(N).to(label)
        ones = ones.index_select(0, label.long())
        size.append(N)
        ones = ones.view(*size)
        ones = ones.transpose(2, 3)
        ones = ones.transpose(1, 2)
        return ones
    
    def compute_flow_loss(self, flow_preds, flow_gts):
        # preds
        flow_preds = flow_preds.permute(1, 0, 3, 2, 6, 4, 5).squeeze(3)
        inter_num, select_frames, bs, num_cls, hw, d = flow_preds.shape
        flow_preds = flow_preds.view(inter_num, select_frames*bs, num_cls, self.bev_w, self.bev_h, d).transpose(3,4)
        # gts
        flow_gts = flow_gts[0][self.future_pred_head_flow.history_queue_length:]
        flow_gts = flow_gts.view(select_frames*bs, *flow_gts.shape[-4:])
        # flow loss
        losses_flow = self.future_pred_head_flow.loss_flow(flow_preds, flow_gts)
        return losses_flow
    
    def compute_plan_loss(self, outs_planning, sdc_planning, sdc_planning_mask, gt_future_boxes):
        ## outs_planning, sdc_planning: under ref_lidar coord
        pred_under_ref = torch.cumsum(outs_planning, dim=1)
        gt_under_ref = torch.cumsum(sdc_planning, dim=1)

        losses_plan = self.plan_head.loss(pred_under_ref, gt_under_ref, sdc_planning_mask, gt_future_boxes)
        return losses_plan
    
    def evaluate_occ(self, occ_preds, occ_gts, img_metas):
        # preds
        occ_preds = occ_preds.permute(1, 0, 3, 2, 6, 4, 5).squeeze(3)
        inter_num, select_frames, bs, num_cls, hw, d = occ_preds.shape
        occ_preds = occ_preds.view(inter_num, select_frames*bs, num_cls, self.bev_w, self.bev_h, d).transpose(3,4)
        # gts
        occ_gts = occ_gts[0][self.future_pred_head.history_queue_length:]
        occ_gts = occ_gts.view(select_frames*bs, *occ_gts.shape[-3:])

        hist_for_iou = self.evaluate_occupancy_forecasting(occ_preds[-1], occ_gts, img_metas=img_metas, save_pred=self._viz_pcd_flag, save_path=self._viz_pcd_path)
        hist_for_iou_current = self.evaluate_occupancy_forecasting(occ_preds[-1][0:1], occ_gts[0:1], img_metas=img_metas, save_pred=False)
        hist_for_iou_future = self.evaluate_occupancy_forecasting(occ_preds[-1][1:], occ_gts[1:], img_metas=img_metas, save_pred=False)
        hist_for_iout_future_time_weighting = self.evaluate_occupancy_forecasting(occ_preds[-1][1:], occ_gts[1:], img_metas=img_metas, time_weighting=True)
        return hist_for_iou, hist_for_iou_current, hist_for_iou_future, hist_for_iout_future_time_weighting

    def evaluate_instance(self, occ_preds, flow_preds, occ_gts, instance_gts):
        # occ_preds
        occ_preds = occ_preds.permute(1, 0, 3, 2, 6, 4, 5).squeeze(3)
        inter_num, select_frames, bs, num_cls, hw, d = occ_preds.shape
        occ_preds = occ_preds.view(inter_num, select_frames*bs, num_cls, self.bev_w, self.bev_h, d).transpose(3,4)
        # occ_gts
        occ_gts = occ_gts[0][self.future_pred_head.history_queue_length:]
        occ_gts = occ_gts.view(select_frames*bs, *occ_gts.shape[-3:])
        # flow_preds
        flow_preds = flow_preds.permute(1, 0, 3, 2, 6, 4, 5).squeeze(3)
        inter_num, select_frames, bs, num_cls, hw, d = flow_preds.shape
        flow_preds = flow_preds.view(inter_num, select_frames*bs, num_cls, self.bev_w, self.bev_h, d).transpose(3,4)
        # instance_gts
        instance_gts = instance_gts[0][self.future_pred_head.history_queue_length:]
        instance_gts = instance_gts.view(select_frames*bs, *instance_gts.shape[-3:])

        vpq = self.evaluate_instance_prediction(occ_preds[-1], flow_preds[-1], occ_gts, instance_gts)
        return vpq

    def evaluate_plan(self, next_pose_preds, sdc_planning, sdc_planning_mask, segmentation_bev, img_metas):
        """
            pred_ego_fut_trajs: B,Lout,2
            gt_ego_fut_trajs:   B,Lout,2
            segmentation_bev:   B,Lout,h,w
        """
        next_pose_gts = sdc_planning

        # pred, gt: under ref_lidar coord
        pred_under_ref = torch.cumsum(next_pose_preds[..., :2], dim=1)
        gt_under_ref = torch.cumsum(next_pose_gts[..., :2], dim=1).float()

        if self._viz_pcd_flag:
            save_data = np.load(os.path.join(self._viz_pcd_path, img_metas[0]["scene_token"]+'_'+img_metas[0]["lidar_token"]+'.npz'), allow_pickle=True)
            np.savez(os.path.join(self._viz_pcd_path, img_metas[0]["scene_token"]+'_'+img_metas[0]["lidar_token"]), 
                                occ_pred=save_data['occ_pred'], pose_pred=pred_under_ref[0].detach().cpu().numpy())

        self.planning_metric_v2(pred_under_ref, gt_under_ref, sdc_planning_mask, segmentation_bev)


    @auto_fp16(apply_to=('img', 'segmentation', 'flow', 'sdc_planning'))
    def forward_train(self,
                      img_metas=None,
                      img=None,
                      # occ_flow
                      segmentation=None,
                      instance=None, 
                      flow=None,
                      # sdc-plan
                      sdc_planning=None,
                      sdc_planning_mask=None,
                      command=None,
                      gt_future_boxes=None,
                      # sample_traj
                      sample_traj=None,
                      # vel_sterring
                      vel_steering=None,
                      ):
        """Forward training function.
        Args:
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            segmentation (list[torch.Tensor])
            flow (list[torch.Tensor])
            sample_traj
        Returns:
            dict: Losses of different branches.
        """

        # manually stop forward
        if self.only_generate_dataset:
            return {"pseudo_loss": torch.tensor(0.0, device=img.device, requires_grad=True)}


        # Augmentations.
        # A1. Randomly drop cur image input.
        if np.random.rand() < self.random_drop_image_rate:
            img[:, -1:, ...] = torch.zeros_like(img[:, -1:, ...])
        # A2. Randomly drop previous image inputs.
        num_frames = img.size(1)
        if np.random.rand() < self.random_drop_prev_rate:
            random_drop_prev_v2_end_idx = (
                self.random_drop_prev_end_idx if self.random_drop_prev_end_idx is not None
                else num_frames)
            drop_prev_index = np.random.randint(
                self.random_drop_prev_start_idx, random_drop_prev_v2_end_idx)
        else:
            drop_prev_index = -1


        # Extract history BEV features.
        # B1. Forward previous frames.
        prev_img = img[:, :-1, ...]
        prev_img_metas = copy.deepcopy(img_metas)
        # B2. Randomly grid-mask prev_bev.
        prev_bev, prev_bev_list = self.obtain_history_bev(prev_img, prev_img_metas, drop_prev_index=drop_prev_index)
        # B2. Randomly grid-mask prev_bev.
        if self.grid_mask_prev and prev_bev is not None:
            b, n, c = prev_bev.shape
            assert n == self.bev_h * self.bev_w
            prev_bev = prev_bev.view(b, self.bev_h, self.bev_w, c)
            prev_bev = prev_bev.permute(0, 3, 1, 2).contiguous()
            prev_bev = self.grid_mask(prev_bev)
            prev_bev = prev_bev.view(b, c, n).permute(0, 2, 1).contiguous()


        # C. Extract current BEV features.
        img = img[:, -1, ...]
        img_metas = [each[num_frames-1] for each in img_metas]
        if self.turn_on_plan:
            ref_sample_traj = sample_traj[:, :, 0]
            ref_real_traj = sdc_planning[:, 0]
            ref_command = command[:, 0]
            sem_occupancy = segmentation[0][self.future_pred_head.history_queue_length:].unsqueeze(0)   # using GT occupancy to calculate sample_traj cost during training
            sem_occupancy = F.interpolate(sem_occupancy, size=(self.bev_h, self.bev_w, self.future_pred_head.num_pred_height), mode='nearest')
            ref_sem_occupancy = sem_occupancy[:, 0]
            ref_bev, ref_pose_pred, ref_pose_loss = self.obtain_ref_bev_with_plan(img, img_metas, prev_bev, ref_sample_traj, ref_sem_occupancy, ref_command, ref_real_traj)
        else:
            ref_bev = self.obtain_ref_bev(img, img_metas, prev_bev)
            sem_occupancy, ref_pose_pred, ref_pose_loss = None, None, None


        # D. Extract future BEV features.
        valid_frames = [0]
        if not self.only_train_cur_frame:
            if self.supervise_all_future:
                valid_frames.extend(list(range(1, self.future_pred_frame_num + 1)))
            else:  # randomly select one future frame for computing loss to save memory cost.
                train_frame = np.random.choice(np.arange(1, self.future_pred_frame_num + 1), 1)[0]
                valid_frames.append(train_frame)
            # D1. prepare memory_queue
            prev_bev_list = torch.stack(prev_bev_list, dim=1)
            prev_bev_list = torch.cat([prev_bev_list, ref_bev.unsqueeze(1)], dim=1)[:, -self.memory_queue_len:, ...]
            # D2. prepare conditional-normalization dict
            if self.future_pred_head.prev_render_neck.sem_norm and self.future_pred_head.prev_render_neck.sem_gt_train and self.training_epoch < 12:
                occ_gts = segmentation[0][self.future_pred_head.history_queue_length+1-self.memory_queue_len:-1]
                occ_gts = F.interpolate(occ_gts.unsqueeze(1), size=(self.bev_h, self.bev_w, self.future_pred_head.prev_render_neck.pred_height), mode='nearest').transpose(0,1)
            else:
                occ_gts = None
            cond_norm_dict = {'occ_gts': occ_gts}
            # D3. prepare action condition dict
            action_condition_dict = {'command':command, 'vel_steering': vel_steering}
            # D4. prepare planning dict
            plan_dict = {'sem_occupancy': sem_occupancy, 'sample_traj': sample_traj, 'gt_traj': sdc_planning, 'ref_pose_pred': ref_pose_pred}

            # D5. predict future occ in auto-regressive manner
            next_bev_preds, next_bev_sem, next_pose_preds, next_pose_loss = self.future_pred(prev_bev_list, action_condition_dict, cond_norm_dict, plan_dict, 
                                                                            valid_frames, img_metas, prev_img_metas, num_frames, occ_flow='occ')


            # D6. predict future flow in auto-regressive manner
            if self.turn_on_flow:
                next_bev_preds_flow, _, _, _ = self.future_pred(prev_bev_list, action_condition_dict, cond_norm_dict, plan_dict, 
                                                                valid_frames, img_metas, prev_img_metas, num_frames, occ_flow='flow')


        # E. Compute Loss
        losses = dict()
        # E1. Compute loss for occ predictions.
        losses_occupancy = self.compute_occ_loss(next_bev_preds, segmentation)
        losses.update(losses_occupancy)
        # E2. Compute loss for flow predictions.
        if self.turn_on_flow:
            losses_flow = self.compute_flow_loss(next_bev_preds_flow, flow)
            losses.update(losses_flow)
        # E3. Compute loss for plan regression.
        if self.turn_on_plan:
            if 'v1' in self.plan_head_type: # used for fine-grained_MMO when sem_occupancy distinguish categories in MMO
                gt_future_boxes = gt_future_boxes[0]   # Lout,[boxes]  NOTE: Current Support bs=1
                losses_plan = self.compute_plan_loss(next_pose_preds, sdc_planning, sdc_planning_mask, gt_future_boxes)
                losses_plan_cost = ref_pose_loss + sum(next_pose_loss)
                losses_plan.update(losses_plan_cost = 0.1 * losses_plan_cost)
            elif 'v2' in self.plan_head_type:   # used for inflated_GMO when sem_occupancy does not distinguish categories in GMO
                gt_future_boxes = gt_future_boxes[0]   # Lout,[boxes]  NOTE: Current Support bs=1
                losses_plan = self.compute_plan_loss(next_pose_preds, sdc_planning, sdc_planning_mask, gt_future_boxes)
            losses.update(losses_plan)
        # E4. Compute loss for bev rendering
        if self.future_pred_head.prev_render_neck.sem_norm:
            losses_bev_render = self.compute_sem_norm_loss(next_bev_sem, segmentation)
            losses.update(losses_bev_render)
        # if self.future_pred_head.sem_norm:
        #     losses_sem_norm = self.compute_sem_norm(next_bev_sem, segmentation)
        #     losses.update(losses_sem_norm)
        # if self.turn_on_flow and self.future_pred_head_flow.obj_motion_norm:
        #     losses_obj_motion_norm = self.compute_obj_motion_norm(next_bev_flow, flow)
        #     losses.update(losses_obj_motion_norm)

        return losses



    def forward_test(self, 
                     img_metas, 
                     img=None,
                     # occ_flow
                     segmentation=None, 
                     instance=None, 
                     flow=None, 
                     # sdc-plan
                     sdc_planning=None,
                     sdc_planning_mask=None,
                     command=None,
                     segmentation_bev=None,
                     # sample_traj
                     sample_traj=None,
                     # vel_sterring
                     vel_steering=None,
                     **kwargs):
        """has similar implementation with train forward."""

        # manually stop forward
        if self.only_generate_dataset:
            return {'hist_for_iou': 0, 'pred_c': 0, 'vpq':0}


        self.eval()
        # Extract history BEV features.
        # B. Forward previous frames.
        num_frames = img.size(1)
        prev_img = img[:, :-1, ...]
        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev, prev_bev_list = self.obtain_history_bev(prev_img, prev_img_metas)


        # C. Extract current BEV features.
        img = img[:, -1, ...]
        img_metas = [each[num_frames-1] for each in img_metas]
        if self.turn_on_plan:
            ref_sample_traj = sample_traj[:, :, 0]
            ref_command = command[:, 0]
            ref_sem_occupancy = None
            ref_bev, ref_pose_pred, _ = self.obtain_ref_bev_with_plan(img, img_metas, prev_bev, ref_sample_traj, ref_sem_occupancy, ref_command)
        else:
            ref_bev = self.obtain_ref_bev(img, img_metas, prev_bev)
            ref_pose_pred = None


        # D. Predict future BEV.
        valid_frames = [] # no frame have grad
        # D1. prepare memory_queue
        prev_bev_list = torch.stack(prev_bev_list, dim=1)
        prev_bev_list = torch.cat([prev_bev_list, ref_bev.unsqueeze(1)], dim=1)[:, -self.memory_queue_len:, ...]
        # D2. prepare conditional-normalization dict
        cond_norm_dict = {'occ_gts': None}
        # D3. prepare action condition dict
        action_condition_dict = {'command':command, 'vel_steering': vel_steering}
        # D4. prepare planning dict
        plan_dict = {'sem_occupancy': None, 'sample_traj': sample_traj, 'gt_traj': sdc_planning, 'ref_pose_pred': ref_pose_pred}

        # D5. predict future occ in auto-regressive manner
        next_bev_preds, _, next_pose_preds, _ = self.future_pred(prev_bev_list, action_condition_dict, cond_norm_dict, plan_dict,
                                                                valid_frames, img_metas, prev_img_metas, num_frames, occ_flow='occ')

        # D6. predict future flow in auto-regressive manner
        if self.turn_on_flow:
            next_bev_preds_flow, _, _, _ = self.future_pred(prev_bev_list, action_condition_dict, cond_norm_dict, plan_dict,
                                                            valid_frames, img_metas, prev_img_metas, num_frames, occ_flow='flow')


        # E. Evaluate
        test_output = {}
        # evaluate occ
        occ_iou, occ_iou_current, occ_iou_future, occ_iou_future_time_weighting = self.evaluate_occ(next_bev_preds, segmentation, img_metas)
        test_output.update(hist_for_iou=occ_iou, hist_for_iou_current=occ_iou_current, 
                           hist_for_iou_future=occ_iou_future, hist_for_iou_future_time_weighting=occ_iou_future_time_weighting)
        # evaluate flow(instance)
        if self.turn_on_flow:
            vpq = self.evaluate_instance(next_bev_preds, next_bev_preds_flow, segmentation, instance)
            test_output.update(vpq=vpq)
        else:
            test_output.update(vpq=0.1)
        # evluate plan
        if self.turn_on_plan:
            self.evaluate_plan(next_pose_preds, sdc_planning, sdc_planning_mask, segmentation_bev, img_metas)

        return test_output



    def evaluate_occupancy_forecasting(self, pred, gt, img_metas=None, save_pred=False, save_path=None, time_weighting=False):

        B, H, W, D = gt.shape
        pred = F.interpolate(pred, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()

        hist_all = 0
        iou_per_pred_list = []
        pred_list = []
        gt_list = []
        for i in range(B):
            pred_cur = pred[i,...]
            pred_cur = torch.argmax(pred_cur, dim=0).cpu().numpy()
            gt_cur = gt[i, ...].cpu().numpy()
            gt_cur = gt_cur.astype(np.int)

            pred_list.append(pred_cur)
            gt_list.append(gt_cur)

            # ignore noise
            noise_mask = gt_cur != 255

            # GMO and others for max_label=2
            # multiple movable objects for max_label=9
            hist_cur, iou_per_pred = fast_hist(pred_cur[noise_mask], gt_cur[noise_mask], max_label=self.future_pred_head.num_classes)
            if time_weighting:
                hist_all = hist_all + 1 / (i+1) * hist_cur
            else:
                hist_all = hist_all + hist_cur
            iou_per_pred_list.append(iou_per_pred)

        # whether save prediction results
        if save_pred:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            pred_for_save_list = []
            for k in range(B):
                pred_for_save = torch.argmax(pred[k], dim=0).cpu()
                x_grid = torch.linspace(0, H-1, H, dtype=torch.long)
                x_grid = x_grid.view(H, 1, 1).expand(H, W, D)
                y_grid = torch.linspace(0, W-1, W, dtype=torch.long)
                y_grid = y_grid.view(1, W, 1).expand(H, W, D)
                z_grid = torch.linspace(0, D-1, D, dtype=torch.long)
                z_grid = z_grid.view(1, 1, D).expand(H, W, D)
                segmentation_for_save = torch.stack((x_grid, y_grid, z_grid), -1)
                segmentation_for_save = segmentation_for_save.view(-1, 3)
                segmentation_label = pred_for_save.squeeze(0).view(-1,1)
                segmentation_for_save = torch.cat((segmentation_for_save, segmentation_label), dim=-1) # N,4
                kept = segmentation_for_save[:,-1]!=0
                segmentation_for_save= segmentation_for_save[kept].cpu().numpy()
                pred_for_save_list.append(segmentation_for_save)
            np.savez(os.path.join(save_path, img_metas[0]["scene_token"]+'_'+img_metas[0]["lidar_token"]), occ_pred=pred_for_save_list)

        return hist_all

    def compute_planner_metric_stp3(
        self,
        pred_ego_fut_trajs,
        gt_ego_fut_trajs,
        sdc_planning_mask,
        segmentation_bev
    ):
        """Compute planner metric for one sample same as stp3
            pred_ego_fut_trajs: B,Lout,2      
            gt_ego_fut_trajs: B,Lout,2
            sdc_planning_mask: B,Lout
            segmentation_bev: B,Lout,h,w
        """
        metric_dict = {
            'plan_L2_1s':0,
            'plan_L2_2s':0,
            'plan_L2_3s':0,
            'plan_obj_col_1s':0,
            'plan_obj_col_2s':0,
            'plan_obj_col_3s':0,
            'plan_obj_box_col_1s':0,
            'plan_obj_box_col_2s':0,
            'plan_obj_box_col_3s':0,
            'plan_L2_1s_single':0,
            'plan_L2_2s_single':0,
            'plan_L2_3s_single':0,
            'plan_obj_col_1s_single':0,
            'plan_obj_col_2s_single':0,
            'plan_obj_col_3s_single':0,
            'plan_obj_box_col_1s_single':0,
            'plan_obj_box_col_2s_single':0,
            'plan_obj_box_col_3s_single':0,
            
        }
        future_second = 1
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        for i in range(future_second):
            cur_time = (i+1)*2
            traj_L2 = self.planning_metric.compute_L2(
                pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                gt_ego_fut_trajs[0, :cur_time],
                sdc_planning_mask[0, :cur_time]
            )
            traj_L2_single = self.planning_metric.compute_L2(
                pred_ego_fut_trajs[0, cur_time-2:cur_time].detach().to(gt_ego_fut_trajs.device),
                gt_ego_fut_trajs[0, cur_time-2:cur_time],
                sdc_planning_mask[0, cur_time-2:cur_time]
            )
            obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                pred_ego_fut_trajs[:, :cur_time].detach(),
                gt_ego_fut_trajs[:, :cur_time],
                segmentation_bev[:, :cur_time])
            obj_coll_single, obj_box_coll_single = self.planning_metric.evaluate_coll(
                pred_ego_fut_trajs[:, cur_time-2:cur_time].detach(),
                gt_ego_fut_trajs[:, cur_time-2:cur_time],
                segmentation_bev[:, cur_time-2:cur_time])
            metric_dict['plan_L2_{}s'.format(i+1)] = traj_L2
            metric_dict['plan_L2_{}s_single'.format(i+1)] = traj_L2_single
            metric_dict['plan_obj_col_{}s'.format(i+1)] = obj_coll.mean()
            metric_dict['plan_obj_box_col_{}s'.format(i+1)] = obj_box_coll.mean()
            metric_dict['plan_obj_col_{}s_single'.format(i+1)] = obj_coll_single.mean()
            metric_dict['plan_obj_box_col_{}s_single'.format(i+1)] = obj_box_coll_single.mean()

        return metric_dict

    def evaluate_instance_prediction(self, pred_seg, pred_flow, gt_seg, gt_instance):
        """
            pred_seg:  pred_occ:  B*Lout,C,H,W,D
            pred_flow: pred_flow: B*Lout,3,H,W,D
            gt_seg:    gt_occ:    B*Lout,H,W,D
            gt_instance: gt_instance_id: B*Lout,H,W,D
        """

        B, H, W, D = gt_seg.shape

        pred_consistent_instance_seg = self.predict_instance_segmentation(pred_seg, pred_flow)

        # add one feature dimension for interpolate
        pred_consistent_instance_seg = F.interpolate(pred_consistent_instance_seg.float(), size=[H, W, D], mode='nearest').contiguous()
        pred_consistent_instance_seg = pred_consistent_instance_seg.squeeze(1)

        iou = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0

        # starting from the present frame
        pred_instance = pred_consistent_instance_seg
        gt_instance = gt_instance.long()

        assert gt_instance.min() == 0, 'ID 0 of gt_instance must be background'
        pred_segmentation = (pred_instance > 0).long()
        gt_segmentation = (gt_instance > 0).long()

        unique_id_mapping = {}
        for t in range(pred_segmentation.shape[0]):
            result = self.panoptic_metrics(
                pred_segmentation[t].detach(),
                pred_instance[t].detach(),
                gt_segmentation[t],
                gt_instance[t],
                unique_id_mapping,
            )

            iou += result['iou']
            true_positive += result['true_positive']
            false_positive += result['false_positive']
            false_negative += result['false_negative']

        denominator = torch.maximum(
            (true_positive + false_positive / 2 + false_negative / 2),
            torch.ones_like(true_positive)
        )
        pq = iou / denominator

        return pq.cpu().numpy()

    def find_instance_centers(self, center_prediction, conf_threshold=0.1, nms_kernel_size=3, dist_threshold=2):
        assert len(center_prediction.shape) == 4

        center_prediction = F.threshold(center_prediction, threshold=conf_threshold, value=-1)

        nms_padding = (nms_kernel_size - 1) // 2
        maxpooled_center_prediction = F.max_pool3d(
            center_prediction, kernel_size=nms_kernel_size, stride=1, padding=nms_padding
        )

        # Filter all elements that are not the maximum (i.e. the center of the heatmap instance)
        center_prediction[center_prediction != maxpooled_center_prediction] = -1
        centers = torch.nonzero(center_prediction > 0)[:, 1:].float()

        # distance threshold
        if len(self.vehicles_id) == 2:
            return centers
        else:
            distances = torch.cdist(centers, centers, p=2)
            combine_mask = (distances < dist_threshold).float()
            combine_centers = torch.mm(combine_mask, centers) / combine_mask.sum(-1).unsqueeze(1)
            combine_centers = torch.unique(combine_centers, dim=0).long()

            return combine_centers # Nc,3

    def group_pixels(self, centers, offset_predictions):
        dx, dy, dz = offset_predictions.shape[-3:]
        x_grid = (
            torch.arange(dx, dtype=offset_predictions.dtype, device=offset_predictions.device)
            .view(1, dx, 1, 1)
            .repeat(1, 1, dy, dz)
        )
        y_grid = (
            torch.arange(dy, dtype=offset_predictions.dtype, device=offset_predictions.device)
            .view(1, 1, dy, 1)
            .repeat(1, dx, 1, dz)
        )
        z_grid = (
            torch.arange(dz, dtype=offset_predictions.dtype, device=offset_predictions.device)
            .view(1, 1, 1, dz)
            .repeat(1, dx, dy, 1)
        )

        pixel_grid = torch.cat((x_grid, y_grid, z_grid), dim=0)
        center_locations = (pixel_grid + offset_predictions).view(3, dx*dy*dz, 1).permute(2, 1, 0)
        centers = centers.view(-1, 1, 3)

        distances = torch.norm(centers - center_locations, dim=-1)

        instance_id = torch.argmin(distances, dim=0).reshape(1, dx, dy, dz) + 1
        return instance_id

    def update_instance_ids(self, instance_seg, old_ids, new_ids):
        indices = torch.arange(old_ids.max() + 1, device=instance_seg.device)
        for old_id, new_id in zip(old_ids, new_ids):
            indices[old_id] = new_id

        return indices[instance_seg].long()

    def make_instance_seg_consecutive(self, instance_seg):
        # Make the indices of instance_seg consecutive
        unique_ids = torch.unique(instance_seg)
        new_ids = torch.arange(len(unique_ids), device=instance_seg.device)
        instance_seg = self.update_instance_ids(instance_seg, unique_ids, new_ids)
        return instance_seg

    def get_instance_segmentation_and_centers(self,
        center_predictions,
        offset_predictions,
        foreground_mask,
        conf_threshold=0.1,
        nms_kernel_size=5,
        max_n_instance_centers=100,):

        dx, dy, dz = offset_predictions.shape[-3:]
        center_predictions = center_predictions.view(1, -1, dx, dy, dz) # 1,cls,x,y,z
        offset_predictions = offset_predictions.view(3, dx, dy, dz)
        foreground_mask = foreground_mask.view(1, dx, dy, dz)

        # class-wise center proposal
        cls_kernel_size = [7, 15, 11, 17, 7, 11, 13, 5]
        center_dist_threshold = [2, 10, 4, 8, 2.3, 12, 6.4, 1.6]  # bicycle, bus, car, construction, motorcycle, trailer, truck
        center_dist_threshold = [x * 2 for x in center_dist_threshold]  # voxel-wise 0.5m
        centers = []
        for i in range(center_predictions.shape[1]):
            center_cls = self.find_instance_centers(center_predictions[:,i], conf_threshold=conf_threshold, nms_kernel_size=nms_kernel_size, dist_threshold=center_dist_threshold[i])    # Nc,3
            centers.append(center_cls)
        centers = torch.cat(centers, dim=0)

        if not len(centers):
            return torch.zeros(foreground_mask.shape, dtype=torch.int64, device=center_predictions.device)

        if len(centers) > max_n_instance_centers:
            centers = centers[:max_n_instance_centers].clone()
        
        instance_ids = self.group_pixels(centers, offset_predictions * foreground_mask.float()) 
        instance_seg = (instance_ids * foreground_mask.float()).long()

        # Make the indices of instance_seg consecutive
        instance_seg = self.make_instance_seg_consecutive(instance_seg) 

        return instance_seg.long()  # 1,H,W,D

    def flow_warp(self, occupancy, flow, mode='nearest', padding_mode='zeros'):
        '''
        Warp ground-truth flow-origin occupancies according to predicted flows
        '''

        _, num_waypoints, _, grid_dx_cells, grid_dy_cells, grid_dz_cells = occupancy.size()

        dx = torch.linspace(-1, 1, steps=grid_dx_cells)
        dy = torch.linspace(-1, 1, steps=grid_dy_cells)
        dz = torch.linspace(-1, 1, steps=grid_dz_cells)

        x_idx, y_idx, z_idx = torch.meshgrid(dx, dy, dz)
        identity_indices = torch.stack((x_idx, y_idx, z_idx), dim=0).to(device=occupancy.device)

        warped_occupancy = []
        for k in range(num_waypoints):  # 1
            flow_origin_occupancy = occupancy[:, k]  # B T 1 dx dy dz -> B 1 dx dy dz
            pred_flow = flow[:, k]  # B T 3 dx dy dz -> B 3 dx dy dz
            # Normalize along the width and height direction
            normalize_pred_flow = torch.stack(
                (2.0 * pred_flow[:, 0] / (grid_dx_cells - 1),  
                2.0 * pred_flow[:, 1] / (grid_dy_cells - 1),
                2.0 * pred_flow[:, 2] / (grid_dz_cells - 1),),
                dim=1,
            )

            warped_indices = identity_indices + normalize_pred_flow
            warped_indices = warped_indices.permute(0, 2, 3, 4, 1)

            flow_origin_occupancy = flow_origin_occupancy.permute(0, 1, 4, 3, 2)

            sampled_occupancy = F.grid_sample(
                input=flow_origin_occupancy,
                grid=warped_indices,
                mode=mode,
                padding_mode='zeros',
                align_corners=True,
            )
            warped_occupancy.append(sampled_occupancy)
        return warped_occupancy[0]

    def make_instance_id_temporally_consecutive(self, pred_inst, preds, backward_flow, ignore_index=255.0):

        assert pred_inst.shape[0] == 1, 'Assumes batch size = 1'

        # Initialise instance segmentations with prediction corresponding to the present
        consistent_instance_seg = [pred_inst.unsqueeze(0)]
        backward_flow = backward_flow.clone().detach()
        backward_flow[backward_flow == ignore_index] = 0.0
        seq_len, _, dx, dy, dz = preds.shape

        for t in range(1, seq_len):

            init_warped_instance_seg = self.flow_warp(consistent_instance_seg[-1].unsqueeze(0).float(), backward_flow[t:t+1].unsqueeze(0)).int()

            warped_instance_seg = init_warped_instance_seg * preds[t:t+1, 0]
        
            consistent_instance_seg.append(warped_instance_seg)
        
        consistent_instance_seg = torch.cat(consistent_instance_seg, dim=1)
        return consistent_instance_seg

    def predict_instance_segmentation(self, pred_seg, pred_flow):
        """
            pred_seg:  pred_occ:  B*Lout,C,H,W,D
            pred_flow: pred_flow: B*Lout,3,H,W,D
        """
        pred_seg_sm = pred_seg.detach()
        pred_seg_sm = torch.argmax(pred_seg_sm, dim=1, keepdims=True)
        vehicles_id = torch.tensor(self.vehicles_id).to(pred_seg_sm)
        foreground_masks = torch.isin(pred_seg_sm.squeeze(1), vehicles_id)

        pred_inst_batch = self.get_instance_segmentation_and_centers(
            torch.softmax(pred_seg, dim=1)[0:1, self.vehicles_id].detach(),
            pred_flow[1:2].detach(), 
            foreground_masks[1:2].detach(),
            nms_kernel_size=7,
        )  
        
        pred_seg_sm = torch.tensor(pred_seg_sm.detach() > 0, dtype=torch.int)   # sem_classes -> GMO
        consistent_instance_seg = self.make_instance_id_temporally_consecutive(
                pred_inst_batch,
                pred_seg_sm[1:],
                pred_flow[1:].detach(),
                )

        consistent_instance_seg = torch.cat([pred_inst_batch.unsqueeze(0), consistent_instance_seg], dim=1)

        return consistent_instance_seg.permute(1, 0, 2, 3, 4).long()

    def combine_mask(self, segmentation: torch.Tensor, instance: torch.Tensor, n_classes: int, n_all_things: int):
        '''
        Shift all things ids by num_classes and combine things and stuff into a single mask
        '''
        instance = instance.view(-1)
        instance_mask = instance > 0
        instance = instance - 1 + n_classes

        segmentation = segmentation.clone().view(-1)
        segmentation_mask = torch.bitwise_and(segmentation > 0, segmentation < n_classes+1) # things_mask

        # Build an index from instance id to class id.
        instance_id_to_class_tuples = torch.cat(
            (
                instance[instance_mask & segmentation_mask].unsqueeze(1),
                segmentation[instance_mask & segmentation_mask].unsqueeze(1),
            ),
            dim=1,
        )   # N_ins_points, 2 [ins_id, sem_cls]

        instance_id_to_class = -instance_id_to_class_tuples.new_ones((n_all_things,))
        instance_id_to_class[instance_id_to_class_tuples[:, 0]] = instance_id_to_class_tuples[:, 1] # instance_id -- sem_class
        instance_id_to_class[torch.arange(n_classes, device=segmentation.device)] = torch.arange(
            n_classes, device=segmentation.device
        )

        segmentation[instance_mask] = instance[instance_mask]
        segmentation[~segmentation_mask] = 0

        return segmentation, instance_id_to_class
        # segmentation: ins_id
        # instance_id_to_class[ins_id] = sem_class

    def panoptic_metrics(self, pred_segmentation, pred_instance, gt_segmentation, gt_instance, unique_id_mapping):
        # GMO and others
        n_classes = 1   # numebr of things_class  (GMO=1)
        self.keys = ['iou', 'true_positive', 'false_positive', 'false_negative'] # hard coding
        result = {key: torch.zeros(n_classes, dtype=torch.float32, device=gt_instance.device) for key in self.keys}

        assert pred_segmentation.dim() == 3
        assert pred_segmentation.shape == pred_instance.shape == gt_segmentation.shape == gt_instance.shape

        n_instances = int(torch.cat([pred_instance, gt_instance]).max().item())
        n_all_things = n_instances + n_classes  # Classes + instances.
        n_things_and_void = n_all_things + 1

        pred_segmentation = pred_segmentation.long().detach().cpu()
        pred_instance = pred_instance.long().detach().cpu()
        gt_segmentation = gt_segmentation.long().detach().cpu()
        gt_instance = gt_instance.long().detach().cpu()
        
        prediction, pred_to_cls = self.combine_mask(pred_segmentation, pred_instance, n_classes, n_all_things)
        target, target_to_cls = self.combine_mask(gt_segmentation, gt_instance, n_classes, n_all_things)

        # Compute ious between all stuff and things
        # hack for bincounting 2 arrays together
        x = prediction + n_things_and_void * target  
        bincount_2d = torch.bincount(x.long(), minlength=n_things_and_void ** 2) 
        if bincount_2d.shape[0] != n_things_and_void ** 2:
            raise ValueError('Incorrect bincount size.')
        conf = bincount_2d.reshape((n_things_and_void, n_things_and_void))
        # Drop void class
        conf = conf[1:, 1:]  
        # Confusion matrix contains intersections between all combinations of classes
        union = conf.sum(0).unsqueeze(0) + conf.sum(1).unsqueeze(1) - conf
        iou = torch.where(union > 0, (conf.float() + 1e-9) / (union.float() + 1e-9), torch.zeros_like(union).float())

        mapping = (iou > self.iou_thresh_for_vpq).nonzero(as_tuple=False)
 
        # Check that classes match.
        is_matching = pred_to_cls[mapping[:, 1]] == target_to_cls[mapping[:, 0]]
        mapping = mapping[is_matching.detach().cpu().numpy()]
        tp_mask = torch.zeros_like(conf, dtype=torch.bool)
        tp_mask[mapping[:, 0], mapping[:, 1]] = True

        # First ids correspond to "stuff" i.e. semantic seg.
        # Instance ids are offset accordingly
        for target_id, pred_id in mapping:
            cls_id = pred_to_cls[pred_id]
            if cls_id == 0 or cls_id == -1:
                continue

            self.temporally_consistent = True # hard coding !
            if self.temporally_consistent and cls_id == self.gmo_id:
                if target_id.item() in unique_id_mapping and unique_id_mapping[target_id.item()] != pred_id.item():
                    # Not temporally consistent
                    result['false_negative'][target_to_cls[target_id]-1] += 1
                    result['false_positive'][pred_to_cls[pred_id]-1] += 1
                    unique_id_mapping[target_id.item()] = pred_id.item()
                    continue

            result['true_positive'][cls_id-1] += 1
            result['iou'][cls_id-1] += iou[target_id][pred_id]
            unique_id_mapping[target_id.item()] = pred_id.item()

        for target_id in range(n_classes, n_all_things):
            # If this is a true positive do nothing.
            if tp_mask[target_id, n_classes:].any():
                continue
            # If this target instance didn't match with any predictions and was present set it as false negative.
            if target_to_cls[target_id] != -1:
                result['false_negative'][target_to_cls[target_id]-1] += 1

        for pred_id in range(n_classes, n_all_things):
            # If this is a true positive do nothing.
            if tp_mask[n_classes:, pred_id].any():
                continue
            # If this predicted instance didn't match with any prediction, set that predictions as false positive.
            if pred_to_cls[pred_id] != -1 and (conf[:, pred_id] > 0).any():
                result['false_positive'][pred_to_cls[pred_id]-1] += 1

        return result

    def _viz_pcd(self, pred_pcd, pred_ctr,  output_path, gt_pcd=None):
        """Visualize predicted future point cloud."""
        color_map = np.array([
            [0, 0, 230], [219, 112, 147], [255, 0, 0]
        ])
        pred_label = np.ones_like(pred_pcd)[:, 0].astype(np.int) * 0
        if gt_pcd is not None:
            gt_label = np.ones_like(gt_pcd)[:, 0].astype(np.int)

            pred_label = np.concatenate([pred_label, gt_label], 0)
            pred_pcd = np.concatenate([pred_pcd, gt_pcd], 0)

        e2e_predictor_utils._dbg_draw_pc_function(
            pred_pcd, pred_label, color_map, output_path=output_path,
            ctr=pred_ctr, ctr_labels=np.zeros_like(pred_ctr)[:, 0].astype(np.int)
        )

def fast_hist(pred, label, max_label=18):
    pred = copy.deepcopy(pred.flatten())
    label = copy.deepcopy(label.flatten())
    bin_count = np.bincount(max_label * label.astype(int) + pred, minlength=max_label ** 2) 
    iou_per_pred = (bin_count[-1]/(bin_count[-1]+bin_count[1]+bin_count[2]))
    return bin_count[:max_label ** 2].reshape(max_label, max_label),iou_per_pred