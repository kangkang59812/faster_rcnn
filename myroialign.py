import torch
import numpy
from torch import nn


class roiAlign(nn.Module):
    def __init__(self, nthreads=None, spatial_scale=0.0625, channels=1024, height=None, width=None, pooled_height=7, pooled_width=7, sampling_ratio=0):
        super(roiAlign, self).__init__()
        self.channels = channels
        self.pooled_width = pooled_width
        self.pooled_height = pooled_height
        self.nthreads = nthreads
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.width = width
        self.height = height

    def forward(self, bottom_data, bottom_rois):
        channels = self.channels
        pooled_width = self.pooled_width
        pooled_height = self.pooled_height
        nthreads = self.nthreads
        roi_cols = 5
        # n_rois=512, 4个128
        n_rois = int(nthreads / channels / pooled_width / pooled_height)

        for n in range(n_rois):
            # index_n = n * channels * pooled_width * pooled_height
            # offset_bottom_rois = bottom_rois + n * roi_cols

            roi_start_w = bottom_rois[n][1] * self.spatial_scale
            roi_start_h = bottom_rois[n][2] * self.spatial_scale
            roi_end_w = bottom_rois[n][3] * self.spatial_scale
            roi_end_h = bottom_rois[n][4] * self.spatial_scale

            roi_width = torch.max(roi_end_w - roi_start_w, torch.tensor(1.))
            roi_height = torch.max(roi_end_h - roi_start_h, torch.tensor(1.))
            bin_size_h = roi_height / self.pooled_height
            bin_size_w = roi_width / self.pooled_width
            roi_bin_grid_h = (self.sampling_ratio if self.sampling_ratio > 0 else torch.ceil(
                roi_height / self.pooled_height)).int()
            roi_bin_grid_w = (self.sampling_ratio if self.sampling_ratio > 0 else torch.ceil(
                roi_width / self.pooled_width)).int()

            count = roi_bin_grid_h * roi_bin_grid_w
            bilinear = bilinear_interpolate(
                roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height)


class bilinear_interpolate(nn.Module):
    '''
    提前计算双线性插值位置
    '''
    def __init__(self, num=None,
                 height=None,
                 width=None,
                 pooled_height=7,
                 pooled_width=7,
                 iy_upper=None,
                 ix_upper=None,
                 roi_start_h=None,
                 roi_start_w=None,
                 bin_size_h=None,
                 bin_size_w=None,
                 roi_bin_grid_h=None,
                 roi_bin_grid_w=None):
        super(bilinear_interpolate, self).__init__()
        self.num = num  # 588=3*4*7*7
        self.height = height  # 38
        self.width = width  # 51
        self.pooled_height = pooled_height  # 7
        self.pooled_width = pooled_width   # 7
        self.iy_upper = iy_upper  # 3
        self.ix_upper = ix_upper  # 4
        self.roi_start_h = roi_start_h  # 16.1684
        self.roi_start_w = roi_start_w  # 20.9849
        self.bin_size_h = bin_size_h  # 2.0995
        self.bin_size_w = bin_size_w  # 3.5075
        self.roi_bin_grid_h = roi_bin_grid_h  # 3
        self.roi_bin_grid_w = roi_bin_grid_w  # 4

    def forward(self,):
        pre_calc = torch.empty(self.num, 8)


bottom_data = torch.load('/home/lkk/code/copyfaster/input.pt')
bottom_rois = torch.load('/home/lkk/code/copyfaster/rois.pt')
height = bottom_data.shape[2]
width = bottom_data.shape[3]

nthreads = bottom_rois.shape[0]*49*1024
roi = roiAlign(nthreads=nthreads, height=height, width=width)
roi(bottom_data, bottom_rois)
print('')
