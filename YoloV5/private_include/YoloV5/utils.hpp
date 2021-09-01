#pragma once

#include <torch/torch.h>

int make_divisible(const float x, const int div);

// Convert [x, y, w, h] boxes to [x1, y1, x2, y2] top left, bottom right
torch::Tensor xywh2xyxy(torch::Tensor x);

// TODO, GPU support would also be nice (https://github.com/pytorch/vision/blob/7947fc8fb38b1d3a2aca03f22a2e6a3caa63f2a0/torchvision/csrc/ops/cuda/nms_kernel.cu)
/// <summary>
/// Perform Non-Maximum Suppression over the given boxes (on CPU)
/// </summary>
/// <param name="boxes">[N, 4](Float) tensor, x1, y1 (top left), x2, y2 (bottom right)</param>
/// <param name="scores">[N](Float) tensor, score for each box</param>
/// <param name="iou_threshold">IoU threshold for suppression</param>
/// <returns>A [n](Long) tensor of kept indices, with n <= N</returns>
torch::Tensor nms_kernel(const torch::Tensor& boxes_, const torch::Tensor& scores_, const float iou_threshold);

void CopyRawDataToTensor(const std::vector<char>& src, torch::Tensor& dst);