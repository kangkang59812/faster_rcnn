import torch
import numpy
from torch import nn
from tqdm import tqdm


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
        top_data = torch.empty(size=(512, 1024, 7, 7))
        for n in tqdm(range(n_rois)):
            index_n = n * channels * pooled_width * pooled_height
            roi_batch_ind = int(bottom_rois[n][0].item())
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
                num=roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height,
                height=self.height,
                width=self.width,
                iy_upper=roi_bin_grid_h,
                ix_upper=roi_bin_grid_w,
                roi_start_h=roi_start_h,
                roi_start_w=roi_start_w,
                bin_size_h=bin_size_h,
                bin_size_w=bin_size_w,
                roi_bin_grid_h=roi_bin_grid_h,
                roi_bin_grid_w=roi_bin_grid_w)

            pre_calc = bilinear()

            for c in range(channels):
                # _c = index_n + c * pooled_width * pooled_height
                offset_bottom_data = bottom_data[roi_batch_ind][c].view(-1)
                pre_calc_index = 0

                for ph in range(pooled_height):
                    for pw in range(pooled_width):
                        # index = index_n_c + ph * pooled_width + pw
                        output_val = 0.

                        for iy in range(roi_bin_grid_h):
                            for ix in range(roi_bin_grid_w):
                                pc = pre_calc[pre_calc_index]
                                output_val += pc[4] * offset_bottom_data[int(pc[0])] + \
                                    pc[5] * offset_bottom_data[int(pc[1])] + \
                                    pc[6] * offset_bottom_data[int(pc[2])] + \
                                    pc[7] * offset_bottom_data[int(pc[3])]

                                pre_calc_index += 1

                        output_val /= count
                        top_data[n][c][ph][pw] = output_val
            print('')
        return top_data


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

    def forward(self):
        pre_calc = torch.empty(self.num, 8)
        pre_calc_index = 0

        for ph in range(self.pooled_height):
            for pw in range(self.pooled_width):
                for iy in range(self.iy_upper):
                    yy = torch.tensor(self.roi_start_h + ph * self.bin_size_h +
                                      (iy + 0.5) * self.bin_size_h / self.roi_bin_grid_h)
                    for ix in range(self.ix_upper):
                        xx = torch.tensor(self.roi_start_w + pw * self.bin_size_w +
                                          (ix + 0.5) * self.bin_size_w / self.roi_bin_grid_w)

                        x = xx.item()
                        y = yy.item()

                        if (y < -1.0 or y > self.height or x < -1.0 or x > self.width):
                            pre_calc[pre_calc_index, :] = 0
                            continue
                        if y <= 0:
                            y = 0
                        if x <= 0:
                            x = 0

                        y_low = int(y)
                        x_low = int(x)
                        y_high = None
                        x_high = None

                        if y_low >= height - 1:
                            y_high = height - 1
                            y_low = height - 1
                            y = y_low
                        else:
                            y_high = y_low + 1

                        if x_low >= width - 1:
                            x_high = width - 1
                            x_low = width - 1
                            x = x_low
                        else:
                            x_high = x_low + 1

                        ly = y - y_low
                        lx = x - x_low
                        hy = 1. - ly
                        hx = 1. - lx
                        w1 = hy * hx
                        w2 = hy * lx
                        w3 = ly * hx
                        w4 = ly * lx
                        pre_calc[pre_calc_index] = torch.tensor(
                            [y_low * width + x_low, y_low * width + x_high, y_high * width + x_low, y_high * width + x_high, w1, w2, w3, w4])
                        pre_calc_index += 1

        return pre_calc


bottom_data = torch.load('/home/lkk/code/copyfaster/input.pt')
bottom_rois = torch.load('/home/lkk/code/copyfaster/rois.pt')
height = bottom_data.shape[2]
width = bottom_data.shape[3]

nthreads = bottom_rois.shape[0]*49*1024
roi = roiAlign(nthreads=nthreads, height=height, width=width)
roi(bottom_data, bottom_rois)
print('')
