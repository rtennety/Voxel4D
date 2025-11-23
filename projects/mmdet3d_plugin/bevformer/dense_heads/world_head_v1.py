import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from mmdet.models import HEADS, build_loss

from mmcv.runner import force_fp32, auto_fp16
from .world_head_base import WorldHeadBase
from projects.mmdet3d_plugin.bevformer.losses.semkitti_loss import geo_scal_loss, sem_scal_loss, CE_ssc_loss, Smooth_L1_loss, BCE_loss
from projects.mmdet3d_plugin.bevformer.losses.lovasz_softmax import lovasz_softmax


@HEADS.register_module()
class WorldHeadV1(WorldHeadBase):
    def __init__(self,
                 history_queue_length,
                 soft_weight,
                 pred_history_frame_num=0,
                 pred_future_frame_num=0,
                 per_frame_loss_weight=(1.0,),
                 loss_weight_cfg=None,

                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.history_queue_length = history_queue_length    # 4

        self.pred_history_frame_num = pred_history_frame_num    # 0(mem-e)
        self.pred_future_frame_num = pred_future_frame_num      # 0(mem-e)

        self.pred_frame_num = 1 + self.pred_history_frame_num + self.pred_future_frame_num
        self.per_frame_loss_weight = per_frame_loss_weight
        assert len(self.per_frame_loss_weight) == self.pred_frame_num

        self.class_weights = np.ones((self.num_classes,))
        self.class_weights[1:] = 5
        self.class_weights = torch.from_numpy(self.class_weights)

        # voxel sem losses
        if loss_weight_cfg is None:
            self.multi_loss = False
            self.loss_voxel_ce_weight = 1.0
        else:
            self.multi_loss = True
            self.loss_voxel_ce_weight = loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
            self.loss_voxel_sem_scal_weight = loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
            self.loss_voxel_lovasz_weight = loss_weight_cfg.get('loss_voxel_lovasz_weight', 1.0)
            self.loss_voxel_geo_scal_weight = loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)

        self.soft_weight = soft_weight
        self._init_bev_pred_layers()

        self.num_points_sampling_feat = self.transformer.decoder.num_layers
        if self.soft_weight:
            self.bev_soft_weights = nn.Sequential(
                nn.Linear(self.embed_dims//2, self.embed_dims//2),
                nn.LayerNorm(self.embed_dims//2),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims//2, self.num_points_sampling_feat),
            )

            self.occ_pred_conv = nn.Sequential(
                nn.Linear(self.embed_dims//2, self.embed_dims//2),
                nn.LayerNorm(self.embed_dims//2),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims//2, self.num_pred_height * self.num_classes * self.pred_frame_num)
            )
            if self.turn_on_flow:
                self.occ_pred_conv[3].bias = nn.parameter.Parameter(torch.zeros(self.num_pred_height * self.num_classes * self.pred_frame_num).float(), requires_grad=True)

    def _init_bev_pred_layers(self):
        """Overwrite the {self.bev_pred_head} of super()._init_layers()
        """
        bev_pred_branch = []
        mid_dims = self.embed_dims//2 if self.soft_weight else self.embed_dims
        for _ in range(self.num_pred_fcs):
            bev_pred_branch.append(nn.Linear(self.embed_dims, mid_dims))
            bev_pred_branch.append(nn.LayerNorm(mid_dims))
            bev_pred_branch.append(nn.ReLU(inplace=True))

        # not_soft_weight: direct output
        if not self.soft_weight:
            bev_pred_branch.append(nn.Linear(
                mid_dims, self.pred_frame_num * self.num_pred_height * self.num_classes))
            if self.turn_on_flow:
                bev_pred_branch[-1].bias = nn.parameter.Parameter(torch.zeros(self.num_pred_height * self.num_classes * self.pred_frame_num).float(), requires_grad=True)

        bev_pred_head = nn.Sequential(*bev_pred_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # Auxiliary supervision for all intermediate results.
        num_pred = self.transformer.decoder.num_layers if self.transformer.decoder.return_intermediate else 1
        self.bev_pred_head = _get_clones(bev_pred_head, num_pred)

    def forward_head_soft(self, next_bev_feats):
        """Get freespace estimation from multi-frame BEV feature maps.

        Args:
            next_bev_feats (torch.Tensor): with shape as
                [pred_frame_num_set, inter_num, bs, bev_h * bev_w, dims]    pred_frame_num_set = cur + future_select
                self.pred_frame_num: history frames + current frame + future frames.
        """
        next_bev_preds = []
        for lvl in range(next_bev_feats.shape[1]):
            next_bev_preds.append(self.bev_pred_head[lvl](next_bev_feats[:, lvl]))
        
        if self.soft_weight:
            bev_soft_weights = self.bev_soft_weights(next_bev_preds[-1])
            bev_soft_weights = torch.softmax(bev_soft_weights, dim=1)
        else:
            bev_soft_weights = torch.ones([next_bev_preds[-1].shape[0], next_bev_preds[-1].shape[1], 1, self.num_points_sampling_feat], ).to(next_bev_preds[0].device) / self.num_points_sampling_feat
        
        # soft_weight
        out_bev_feats = 0
        for feat, weights in zip(next_bev_preds, torch.unbind(bev_soft_weights, dim=-1)):
            out_bev_feats += feat * weights.unsqueeze(-1)
        
        # out pred
        out_occ = self.occ_pred_conv(out_bev_feats) # Lout,B,hw,c -> Lout,B,hw,d*cls*pred_frame

        # base + pred
        out_occ = out_occ.view(*out_occ.shape[:-1], self.num_pred_height, self.num_classes, self.pred_frame_num)
        base_occ = out_occ[..., self.pred_history_frame_num][..., None]
        out_occ = torch.cat([
            out_occ[..., :self.pred_history_frame_num] + base_occ,
            base_occ,
            out_occ[..., self.pred_history_frame_num + 1:] + base_occ
        ], -1)

        out_occ = out_occ.permute(0, 5, 1, 2, 3, 4).contiguous().unsqueeze(1)

        return out_occ  
    
    def forward_head_layers(self, next_bev_feats):
        """Get freespace estimation from multi-frame BEV feature maps.

        Args:
            next_bev_feats (torch.Tensor): with shape as
                [Lout, inter_num, bs, bev_h * bev_w, dims]    Lout = cur + future_select
                self.pred_frame_num: history frames + current frame + future frames.
        """
        next_bev_preds = []
        for lvl in range(next_bev_feats.shape[1]):
            #  ===> Lout, bs, h*w, d, num_frame
            next_bev_pred = self.bev_pred_head[lvl](next_bev_feats[:, lvl]) # C -> d * num_cls * num_frame 
            next_bev_pred = next_bev_pred.view(
                *next_bev_pred.shape[:-1], self.num_pred_height, self.num_classes, self.pred_frame_num)

            base_bev_pred = next_bev_pred[..., self.pred_history_frame_num][..., None]
            next_bev_pred = torch.cat([
                next_bev_pred[..., :self.pred_history_frame_num] + base_bev_pred,
                base_bev_pred,
                next_bev_pred[..., self.pred_history_frame_num + 1:] + base_bev_pred
            ], -1)

            next_bev_pred = next_bev_pred.permute(0, 5, 1, 2, 3, 4).contiguous()   # Lout, history+1+future, bs, h*w, d, num_cls
            next_bev_preds.append(next_bev_pred)
        next_bev_preds = torch.stack(next_bev_preds, 1)
        return next_bev_preds
    
    def forward_head(self, next_bev_feats):
        if self.soft_weight:
            return self.forward_head_soft(next_bev_feats)   # multi-decoder_layers soft_weight_sum
        else:
            return self.forward_head_layers(next_bev_feats) # multi-decoder_layers

    def loss_voxel(self, output_voxels, target_voxels, tag):
        B, C, pH, pW, pD = output_voxels.shape
        tB, tH, tW, tD = target_voxels.shape

        H, W, D = 256, 256, 20
        # output_voxel align to H,W,D
        if pH != H:
            output_voxels = F.interpolate(output_voxels, size=(H, W, D), mode='trilinear', align_corners=False)
        # target_voxel align to H,W,D
        ratio = tH // H
        if ratio != 1:
            target_voxels = target_voxels.reshape(B, H, ratio, W, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio**3)
            empty_idx = 0
            empty_mask = target_voxels.sum(-1) == empty_idx    # B,H,W,D
            target_voxels = target_voxels.to(torch.int64)
            occ_space = target_voxels[~empty_mask]
            occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
            target_voxels[~empty_mask] = occ_space
            target_voxels = torch.mode(target_voxels, dim=-1)[0]
            target_voxels[target_voxels<0] = 255
            target_voxels = target_voxels.long()

        assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        loss_dict = {}

        loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)

        if self.multi_loss:
            loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
            loss_dict['loss_voxel_lovasz_{}'.format(tag)] = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.softmax(output_voxels, dim=1), target_voxels, ignore=255)
            if self.loss_voxel_geo_scal_weight is not None:
                loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=empty_idx)

        return loss_dict
    
    def loss_occ(self, output_voxels=None, target_voxels=None, **kwargs):
        """
            output_voxels = inter_num, select_frame*bs, cls, h,w,d
            target_voxels =            select_frame*bs,      H,W,D
        """
        loss_dict = {}
        for index, output_voxel in enumerate(output_voxels):
            loss_dict.update(self.loss_voxel(output_voxel, target_voxels,  tag='inter_{}'.format(index)))
            
        return loss_dict
    
    def loss_sem_norm(self, output_voxels=None, target_voxels=None, **kwargs):
        """
            output_voxels = inter_num, select_frame*bs, cls, h,w,d
            target_voxels =            select_frame*bs,      H,W,D
        """
        inter, B, C, pH, pW, pD = output_voxels.shape
        tB, tH, tW, tD = target_voxels.shape

        H, W, D = 256, 256, 20
        # output_voxel align to H,W,D
        if pH != H:
            output_voxels = F.interpolate(output_voxels.flatten(0,1), size=(H, W, D), mode='trilinear', align_corners=False)
            output_voxels = output_voxels.view(inter, B,C,H,W,D)
        
        # target_voxel align to H,W,D
        ratio = tH // H
        if ratio != 1:
            target_voxels = target_voxels.reshape(B, H, ratio, W, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio**3)
            empty_idx = 0
            empty_mask = target_voxels.sum(-1) == empty_idx
            target_voxels = target_voxels.to(torch.int64)
            occ_space = target_voxels[~empty_mask]
            occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
            target_voxels[~empty_mask] = occ_space
            target_voxels = torch.mode(target_voxels, dim=-1)[0]
            target_voxels[target_voxels<0] = 255
            target_voxels = target_voxels.long()
        
        assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        
        loss_dict = {}
        for index, output_voxel in enumerate(output_voxels):
            inter_loss = CE_ssc_loss(output_voxel, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
            loss_dict['loss_sem_norm_{}'.format(index)] = inter_loss

        return loss_dict

    def loss_obj_motion_norm(self, output_voxels=None, target_voxels=None, **kwargs):
        """
            output_voxels = inter_num, select_frame*bs, cls, h,w,d
            target_voxels =            select_frame*bs,      H,W,D
        """
        inter, B, C, H, W, D = output_voxels.shape
        tB, tC, tH, tW, tD = target_voxels.shape
        
        assert torch.isnan(output_voxels).sum().item() == 0

        # output_voxel align to target_voxel
        if H != tH:
            output_voxels = F.interpolate(output_voxels.flatten(0,1), size=(tH, tW, tD), mode='trilinear', align_corners=False)
            output_voxels = output_voxels.view(inter, B,C,tH,tW,tD)

        output_voxels = output_voxels.permute(0,1,3,4,5,2)    # inter,B*L,H,W,D,3
        target_voxels = target_voxels.permute(0,2,3,4,1)      #       B*L,H,W,D,3

        loss_dict = {}
        for index, output_voxel in enumerate(output_voxels):
            inter_loss = (0.5) * Smooth_L1_loss(output_voxel, target_voxels, ignore_index=255)
            loss_dict['loss_obj_motion_norm_{}'.format(index)] = inter_loss

        return loss_dict
    
    def loss_voxel_flow(self, output_voxels, target_voxels, tag):                    
        B, C, H, W, D = output_voxels.shape
        tB, tC, tH, tW, tD = target_voxels.shape
        
        assert torch.isnan(output_voxels).sum().item() == 0

        # output_voxel align to target_voxel
        if H != tH:
            output_voxels = F.interpolate(output_voxels, size=(tH, tW, tD), mode='trilinear', align_corners=False)

        output_voxels = output_voxels.permute(0,2,3,4,1)    # B,tH,tW,tD,3
        target_voxels = target_voxels.permute(0,2,3,4,1)

        loss_dict = {}

        loss_dict['loss_flow_l1_{}'.format(tag)] = (0.5) * Smooth_L1_loss(output_voxels, target_voxels, ignore_index=255)

        return loss_dict
    
    def loss_flow(self, output_voxels=None, target_voxels=None, **kwargs):
        """
            output_voxels = inter_num, select_frame*bs, 3, h,w,d
            target_voxels =            select_frame*bs, 3, H,W,D
        """
        loss_dict = {}
        for index, output_voxel in enumerate(output_voxels):
            loss_dict.update(self.loss_voxel_flow(output_voxel, target_voxels,  tag='inter_{}'.format(index)))
            
        return loss_dict

    def loss_bev_occ(self, output_voxels, target_voxels):
        B, pH, pW, pD = output_voxels.shape
        tB, tH, tW, tD = target_voxels.shape
        
        output_voxels = output_voxels.unsqueeze(1)
        target_voxels = target_voxels.unsqueeze(1)

        H, W, D = 256, 256, 20
        # output_voxel align to H,W,D
        if pH != H:
            output_voxels = F.interpolate(output_voxels, size=(H, W, D), mode='trilinear', align_corners=False).squeeze(1)
        # target_voxel align to H,W,D
        ratio = tH // H
        if ratio != 1:
            target_voxels = target_voxels.reshape(B, H, ratio, W, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio**3)
            empty_idx = 0
            empty_mask = target_voxels.sum(-1) == empty_idx
            target_voxels = target_voxels.to(torch.int64)
            occ_space = target_voxels[~empty_mask]
            occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
            target_voxels[~empty_mask] = occ_space
            target_voxels = torch.mode(target_voxels, dim=-1)[0]
            target_voxels[target_voxels<0] = 0
            target_voxels = target_voxels.long()
        
        loss_bev_occ = BCE_loss(output_voxels, target_voxels)
        return loss_bev_occ
    
    def loss_bev_sem(self, output_voxels, target_voxels):
        B, pC, pH, pW = output_voxels.shape
        tB, tH, tW = target_voxels.shape
        
        output_voxels = F.interpolate(output_voxels, size=(tH, tW), mode='bilinear', align_corners=False)

        loss_bev_sem = CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels))
        return loss_bev_sem

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