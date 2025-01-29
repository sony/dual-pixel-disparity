# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from .costdcnet.encoder3d import Encoder3D
from .costdcnet.encoder2d import Encoder2D
from .costdcnet.unet3d import UNet3D

class CostDCNet(nn.Module):
    def __init__(self, args):
        super(CostDCNet, self).__init__()

        self.res = 16 # number of depth plane
        self.up_scale = 4 # scale factor of upsampling
        self.depth_max = 255

        self.models = {}

        # Networks
        self.enc2d  = Encoder2D(in_ch=4, output_dim=16)  
        self.enc3d  = Encoder3D(1, 16, D= 3, planes=(32, 48, 64)) 
        self.unet3d = UNet3D(32, self.up_scale**2, f_maps=[32, 48, 64, 80], mode="nearest")

        
        self.z_step = self.depth_max /(self.res-1)
            
    def forward(self, input):
     
        outputs = {}
        losses = {}
        
        rgb = input['rgb']
        dep = input['d']

        ##############################################################
        ## [step 1] RGB-D Feature Volume Construction
        in_2d = torch.cat([rgb, dep],1)
        in_3d = self.depth2MDP(dep)
        feat2d = self.enc2d(in_2d)
        feat3d = self.enc3d(in_3d)
        rgbd_feat_vol = self.fusion(feat3d, feat2d)

        ## [step 2] Cost Volume Prediction
        cost_vol, _ = self.unet3d(rgbd_feat_vol)
        
        ## [step 3] Depth Regression
        pred = self.upsampling(cost_vol, res = self.res, up_scale=self.up_scale) * self.z_step

        ###############################################################
            
        return pred

    def depth2MDP(self, dep):
        # Depth to sparse tensor in MDP (multiple-depth-plane)        
        idx = torch.round(dep / self.z_step).type(torch.int64)
        idx[idx>(self.res-1)] = self.res - 1
        idx[idx<0] = 0
        inv_dep = (idx * self.z_step)
        res_map = (dep - inv_dep) /self.z_step

        if not torch.any(idx):
            idx[0] = 1

        B, C, H, W = dep.size()
        ones = (idx !=0).float()
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        grid_ = torch.stack((grid_y, grid_x), 2).to(dep.device)
        # grid_ = self.grid.clone().detach()
        grid_ = grid_.unsqueeze(0).repeat((B,1,1,1))
        points_yx = grid_.reshape(-1,2)
        point_z = idx.reshape(-1, 1)
        m = (idx != 0).reshape(-1)
        points3d = torch.cat([point_z, points_yx], dim=1)[m]
        split_list = torch.sum(ones, dim=[1,2,3], dtype=torch.int).tolist()
        coords = points3d.split(split_list)
        # feat = torch.ones_like(points3d)[:,0].reshape(-1,1)       ## if occ to feat
        feat = res_map
        feat = feat.permute(0,2,3,1).reshape(-1, feat.size(1))[m]   ## if res to feat
        
        # Convert to a sparse tensor
        in_field = ME.TensorField(
            features = feat, 
            coordinates=ME.utils.batched_coordinates(coords, dtype=torch.float32),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=dep.device,
        )
        return in_field.sparse() 

    def fusion(self, sout, feat2d):
        # sparse tensor to dense tensor
        B0,C0,H0,W0 = feat2d.size()
        dense_output_, min_coord, tensor_stride = sout.dense(min_coordinate=torch.IntTensor([0, 0, 0]))
        dense_output = dense_output_[:, :, :self.res, :H0, :W0]
        B,C,D,H,W = dense_output.size() 
        feat3d_ = torch.zeros((B0, C0, self.res, H0, W0), device = feat2d.device)
        feat3d_[:B,:,:D,:H,:W] += dense_output
        
        # construct type C feat vol
        mask = (torch.sum((feat3d_ != 0), dim=1, keepdim=True)!= 0).float()
        mask_ = mask + (1 - torch.sum(mask, dim=2,keepdim=True).repeat(1,1,mask.size(2),1,1))
        feat2d_ = feat2d.unsqueeze(2).repeat(1,1,self.res,1,1) * mask_ 
        return torch.cat([feat2d_, feat3d_],dim = 1)
    
    def upsampling(self, cost, res = 64, up_scale = None):
        # if up_scale is None not apply per-plane pixel shuffle
        if not up_scale == None:
            b, c, d, h, w = cost.size()
            cost = cost.transpose(1,2).reshape(b, -1, h, w)
            cost = F.pixel_shuffle(cost, up_scale)
        else:
            cost = cost.squeeze(1)
        prop = F.softmax(cost, dim = 1)
        pred =  disparity_regression(prop, res)
        return pred

def convxN_bn_relu(ch_in, ch_tmp, ch_out, kernel, num_conv=2, stride=1, padding=0, bn=True,
                 relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_tmp, kernel, stride, padding,
                            bias=not bn))
    if num_conv > 2:
        for i in range(num_conv - 2):
            layers.append(nn.Conv2d(ch_tmp, ch_tmp, kernel, stride, padding,
                                    bias=not bn))

    layers.append(nn.Conv2d(ch_tmp, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers

class CostDCNetConf(nn.Module):
    def __init__(self, args):
        super(CostDCNetConf, self).__init__()
        self.res = 16 # number of depth plane
        self.up_scale = 4 # scale factor of upsampling
        self.depth_max = 255
        self.conf_ft_scale = args.conf_ft_scale # [1, 2, 4, 8, 16]
        self.conf_num_conv = args.conf_num_conv # [2, 3, 4, 5, 6...]

        self.models = {}

        # Networks
        self.enc2d  = Encoder2D(in_ch=4, output_dim=16)  
        self.enc3d  = Encoder3D(1, 16, D= 3, planes=(32, 48, 64)) 
        self.unet3d = UNet3D(32, self.up_scale**2, f_maps=[32, 48, 64, 80], mode="nearest")

        self.id_conf = convxN_bn_relu(256, int(256/self.conf_ft_scale), 1, num_conv=self.conf_num_conv,
                                        kernel=3, stride=1, padding=1)

        self.z_step = self.depth_max /(self.res-1)
            
    def forward(self, input):
     
        outputs = {}
        losses = {}
        
        rgb = input['rgb']
        dep = input['d']

        ##############################################################
        ## [step 1] RGB-D Feature Volume Construction
        in_2d = torch.cat([rgb, dep],1)
        in_3d = self.depth2MDP(dep)
        feat2d = self.enc2d(in_2d)
        feat3d = self.enc3d(in_3d)
        rgbd_feat_vol = self.fusion(feat3d, feat2d)

        ## [step 2] Cost Volume Prediction
        cost_vol, cost_f0 = self.unet3d(rgbd_feat_vol)
        
        ## [step 3] Depth Regression
        # pred = self.upsampling(cost_vol, res = self.res, up_scale=self.up_scale) * self.z_step
        pred, conf_inv = self.upsampling(cost_vol, rgbd_feat_vol, cost_f0, feat2d, res = self.res, up_scale=self.up_scale)
        pred = pred * self.z_step

        ###############################################################
            
        return pred, conf_inv

    def depth2MDP(self, dep):
        # Depth to sparse tensor in MDP (multiple-depth-plane)        
        idx = torch.round(dep / self.z_step).type(torch.int64)
        idx[idx>(self.res-1)] = self.res - 1
        idx[idx<0] = 0
        inv_dep = (idx * self.z_step)
        res_map = (dep - inv_dep) /self.z_step

        if not torch.any(idx):
            idx[0] = 1

        B, C, H, W = dep.size()
        ones = (idx !=0).float()
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        grid_ = torch.stack((grid_y, grid_x), 2).to(dep.device)
        # grid_ = self.grid.clone().detach()
        grid_ = grid_.unsqueeze(0).repeat((B,1,1,1))
        points_yx = grid_.reshape(-1,2)
        point_z = idx.reshape(-1, 1)
        m = (idx != 0).reshape(-1)
        points3d = torch.cat([point_z, points_yx], dim=1)[m]
        split_list = torch.sum(ones, dim=[1,2,3], dtype=torch.int).tolist()
        coords = points3d.split(split_list)
        # feat = torch.ones_like(points3d)[:,0].reshape(-1,1)       ## if occ to feat
        feat = res_map
        feat = feat.permute(0,2,3,1).reshape(-1, feat.size(1))[m]   ## if res to feat
        
        # Convert to a sparse tensor
        in_field = ME.TensorField(
            features = feat, 
            coordinates=ME.utils.batched_coordinates(coords, dtype=torch.float32),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=dep.device,
        )
        return in_field.sparse() 

    def fusion(self, sout, feat2d):
        # sparse tensor to dense tensor
        B0,C0,H0,W0 = feat2d.size()
        dense_output_, min_coord, tensor_stride = sout.dense(min_coordinate=torch.IntTensor([0, 0, 0]))
        dense_output = dense_output_[:, :, :self.res, :H0, :W0]
        B,C,D,H,W = dense_output.size() 
        feat3d_ = torch.zeros((B0, C0, self.res, H0, W0), device = feat2d.device)
        feat3d_[:B,:,:D,:H,:W] += dense_output
        
        # construct type C feat vol
        mask = (torch.sum((feat3d_ != 0), dim=1, keepdim=True)!= 0).float()
        mask_ = mask + (1 - torch.sum(mask, dim=2,keepdim=True).repeat(1,1,mask.size(2),1,1))
        feat2d_ = feat2d.unsqueeze(2).repeat(1,1,self.res,1,1) * mask_ 
        return torch.cat([feat2d_, feat3d_],dim = 1)
    
    def upsampling(self, cost, rgbft, cost_f0, feat2d, res = 64, up_scale = None):
        # if up_scale is None not apply per-plane pixel shuffle
        if not up_scale == None:
            b, c, d, h, w = cost.size()
            cost = cost.transpose(1,2).reshape(b, -1, h, w)
            cost_up = F.pixel_shuffle(cost, up_scale)
        else:
            cost_up = cost.squeeze(1)
        prop = F.softmax(cost_up, dim = 1)
        pred =  disparity_regression(prop, res)

        conf_inv = None
        pred_conf_inv = self.id_conf(cost)
        if not up_scale == None:
            pred_conf_inv = F.interpolate(pred_conf_inv, scale_factor=up_scale, mode='bilinear')
        conf_inv = torch.clamp(pred_conf_inv, min=0.001, max=1.0)

        return pred, conf_inv
    

def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)