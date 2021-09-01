#include <cmath>

#include "YoloV5/utils.hpp"

int make_divisible(const float x, const int div)
{
    return std::ceil(x / div) * div;
}

torch::Tensor xywh2xyxy(torch::Tensor x)
{
    torch::Tensor output = x.clone();
    output.index({ torch::indexing::Slice(), 0 }) =
        x.index({ torch::indexing::Slice(), 0 }) - x.index({ torch::indexing::Slice(), 2 }) / 2.0f;
    output.index({ torch::indexing::Slice(), 1 }) =
        x.index({ torch::indexing::Slice(), 1 }) - x.index({ torch::indexing::Slice(), 3 }) / 2.0f;
    output.index({ torch::indexing::Slice(), 2 }) =
        x.index({ torch::indexing::Slice(), 0 }) + x.index({ torch::indexing::Slice(), 2 }) / 2.0f;
    output.index({ torch::indexing::Slice(), 3 }) =
        x.index({ torch::indexing::Slice(), 1 }) + x.index({ torch::indexing::Slice(), 3 }) / 2.0f;

    return output;
}

torch::Tensor nms_kernel(const torch::Tensor& boxes_, const torch::Tensor& scores_, const float iou_threshold)
{
    torch::Tensor boxes = boxes_.cpu();
    torch::Tensor scores = scores_.cpu();

    if (boxes.size(0) == 0)
    {
        return torch::empty({ 0 }, torch::TensorOptions().dtype(torch::kLong));
    }

    torch::Tensor x1_t = boxes.index({ torch::indexing::Slice(), 0 }).contiguous();
    torch::Tensor y1_t = boxes.index({ torch::indexing::Slice(), 1 }).contiguous();
    torch::Tensor x2_t = boxes.index({ torch::indexing::Slice(), 2 }).contiguous();
    torch::Tensor y2_t = boxes.index({ torch::indexing::Slice(), 3 }).contiguous();

    torch::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);

    // Get the sorted indices
    torch::Tensor order_t = std::get<1>(scores.sort(0, true));

    const int64_t num_boxes = boxes.size(0);

    torch::Tensor suppressed_t = torch::zeros({ num_boxes }, torch::TensorOptions().dtype(torch::kBool));
    torch::Tensor kept_t = torch::zeros({ num_boxes }, torch::TensorOptions().dtype(torch::kLong));

    bool* suppressed = suppressed_t.data_ptr<bool>();
    int64_t* kept = kept_t.data_ptr<int64_t>();
    int64_t* order = order_t.data_ptr<int64_t>();
    float* x1 = x1_t.data_ptr<float>();
    float* y1 = y1_t.data_ptr<float>();
    float* x2 = x2_t.data_ptr<float>();
    float* y2 = y2_t.data_ptr<float>();
    float* areas = areas_t.data_ptr<float>();

    int64_t num_to_keep = 0;

    // For each box in score order
    for (size_t i = 0; i < num_boxes; ++i)
    {
        int64_t index = order[i];

        if (suppressed[index])
        {
            continue;
        }

        // Keep this one
        kept[num_to_keep++] = index;

        // For each other, check if IoU is < threshold
        for (size_t j = i + 1; j < num_boxes; ++j)
        {
            int64_t jndex = order[j];

            if (suppressed[jndex])
            {
                continue;
            }

            float intersection =
                std::max(0.0f,
                    std::min(x2[index], x2[jndex]) -
                    std::max(x1[index], x1[jndex])
                )
                *
                std::max(0.0f,
                    std::min(y2[index], y2[jndex]) -
                    std::max(y1[index], y1[jndex])
                );

            if (intersection / (areas[index] + areas[jndex] - intersection) > iou_threshold)
            {
                suppressed[jndex] = true;
            }
        }
    }

    return kept_t.index({ torch::indexing::Slice(0, num_to_keep) }).to(boxes.device());
}

void CopyRawDataToTensor(const std::vector<char>& src, torch::Tensor& dst)
{
    if (dst.is_floating_point())
    {
        // YoloV5 floating point tensors are saved with half precision
        torch::Tensor target = torch::zeros_like(dst, torch::TensorOptions().dtype(torch::kF16).layout(torch::kStrided));
        if (target.itemsize() * target.numel() != src.size())
        {
            throw std::runtime_error("Error trying to load raw data into tensor, sizes don't match");
        }

        std::copy(src.data(), src.data() + src.size(), reinterpret_cast<char*>(target.data_ptr()));
        dst.set_data(target.to(torch::kFloat));
    }
    else
    {
        torch::Tensor target = torch::zeros_like(dst, torch::TensorOptions().layout(torch::kStrided));
        if (target.itemsize() * target.numel() != src.size())
        {
            throw std::runtime_error("Error trying to load raw data into tensor, sizes don't match");
        }

        std::copy(src.data(), src.data() + src.size(), reinterpret_cast<char*>(target.data_ptr()));
        dst.set_data(target);
    }
}
