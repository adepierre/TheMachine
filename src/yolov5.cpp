#include "TheMachine/yolov5.hpp"
#include "TheMachine/layers.hpp"

#include <ryml_std.hpp>
#include <ryml.hpp>


YoloV5BlockImpl::YoloV5BlockImpl(const int attach_index_,
    const std::vector<int>& from_, const KnownBlock type_,
    torch::nn::Sequential seq_)
{
    attach_index = attach_index_;
    from = from_;
    type = type_;
    seq = seq_;

    register_module("seq", seq);
}

YoloV5BlockImpl::~YoloV5BlockImpl()
{
}

torch::Tensor YoloV5BlockImpl::forward(std::vector<torch::Tensor> x)
{
    switch (type)
    {
    case KnownBlock::Concat:
    case KnownBlock::Detect:
        return seq->forward(x);
    default:
        return seq->forward(x[0]);
    }
}






int make_divisible(const float x, const int div)
{
    return std::ceil(x / div) * div;
}

// Convert [x, y, w, h] boxes to [x1, y1, x2, y2] top left, bottom right
torch::Tensor xywh2xyxy(torch::Tensor x)
{
    torch::Tensor output = x.clone();
    output.index({ torch::indexing::Slice(), 0 }) =
        x.index({ torch::indexing::Slice(), 0 }) - x.index({ torch::indexing::Slice(), 2 }) / 2.0f;
    output.index({ torch::indexing::Slice(), 1 }) =
        x.index({ torch::indexing::Slice(), 1 }) - x.index({ torch::indexing::Slice(), 3 }) / 2.0f;
    output.index({ torch::indexing::Slice(), 2 }) =
        x.index({ torch::indexing::Slice(), 0 }) - x.index({ torch::indexing::Slice(), 2 }) / 2.0f;
    output.index({ torch::indexing::Slice(), 3 }) =
        x.index({ torch::indexing::Slice(), 1 }) - x.index({ torch::indexing::Slice(), 3 }) / 2.0f;

    return output;
}

// TODO, GPU support would also be nice (https://github.com/pytorch/vision/blob/7947fc8fb38b1d3a2aca03f22a2e6a3caa63f2a0/torchvision/csrc/ops/cuda/nms_kernel.cu)
/// <summary>
/// Perform Non-Maximum Suppression over the boxes given (on CPU)
/// </summary>
/// <param name="boxes">[N, 4](Float-CPU) tensor, x1, y1 (top left), x2, y2 (bottom right)</param>
/// <param name="scores">[N](Float-CPU) tensor, score for each box</param>
/// <param name="iou_threshold">IoU threshold for suppression</param>
/// <returns>A [n](Long) tensor of kept indices, with n <= N</returns>
torch::Tensor nms_kernel(const torch::Tensor& boxes, const torch::Tensor& scores, const float iou_threshold)
{
    if (boxes.size(0) == 0)
    {
        return torch::empty({ 0 }, torch::TensorOptions().dtype(torch::kLong));
    }

    torch::Tensor x1_t = boxes.index({torch::indexing::Slice(), 0}).contiguous();
    torch::Tensor y1_t = boxes.index({torch::indexing::Slice(), 1}).contiguous();
    torch::Tensor x2_t = boxes.index({torch::indexing::Slice(), 2}).contiguous();
    torch::Tensor y2_t = boxes.index({torch::indexing::Slice(), 3}).contiguous();

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
        for (size_t j = i+1; j < num_boxes; ++j)
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

    return kept_t.index({ torch::indexing::Slice(0, num_to_keep) });
}

YoloV5Impl::YoloV5Impl(const std::string& config_path, const int num_in_channels_)
{
    num_in_channels = num_in_channels_;
    ParseConfig(config_path);
    register_module("module_list", module_list);
    SetStride();
}

YoloV5Impl::~YoloV5Impl()
{

}

torch::Tensor YoloV5Impl::forward(torch::Tensor x)
{
    std::vector<torch::Tensor> backbone_outputs = forward_backbone(x);

    YoloV5BlockImpl* detect = module_list[module_list->size() - 1]->as<YoloV5Block>();
    std::vector<torch::Tensor> detect_inputs(detect->from.size());

    for (size_t i = 0; i < detect->from.size(); ++i)
    {
        int c = detect->from[i];
        if (c < 0)
        {
            c = backbone_outputs.size() + c;
        }
        detect_inputs[i] = backbone_outputs[c];
    }

    return detect->forward(detect_inputs);
}

std::vector<torch::Tensor> YoloV5Impl::NonMaxSuppression(torch::Tensor prediction,
    float conf_threshold, float iou_threshold, int max_det, const std::vector<int>& class_filter)
{
    const size_t batch_size = prediction.size(0);
    std::vector<torch::Tensor> outputs(batch_size, torch::zeros({0, 6}, torch::TensorOptions().device(prediction.device())));

    torch::Tensor candidates = prediction.index({ "...", 4 }) > conf_threshold;

    for (size_t i = 0; i < batch_size; ++i)
    {
        torch::Tensor x = prediction[i].index({ candidates[i] });

        // If no candidates, go to next image
        if (!x.size(0))
        {
            continue;
        }

        // Compute overall conf (obj conf * cls_conf)
        x.index({ torch::indexing::Slice(), torch::indexing::Slice(5, torch::indexing::None) }) *=
            x.index({ torch::indexing::Slice(), torch::indexing::Slice(4, 5) });

        torch::Tensor box = xywh2xyxy(x.index({ torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 4) }));

        torch::Tensor conf, j;
        std::tie(conf, j) = x.index({ torch::indexing::Slice(), torch::indexing::Slice(5, torch::indexing::None) }).max(1, true);

        x = torch::cat({ box, conf, j.to(torch::kFloat32) }, 1).index({ conf.view({-1}) > conf_threshold });

        if (class_filter.size() > 0)
        {
            x = x.index({
                (x.index({torch::indexing::Slice(), torch::indexing::Slice(5, 6)}) ==
                torch::from_blob((int*)class_filter.data(), class_filter.size(),
                    torch::TensorOptions().device(x.device())).to(torch::kFloat))
                .any(1) });
        }

        // If no candidates, go to next image
        if (!x.size(0))
        {
            continue;
        }

        // Spatially separate the classes so NMS is done per class
        torch::Tensor boxes = x.index({ torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 4) })
            + x.index({ torch::indexing::Slice(), torch::indexing::Slice(5,6) }) * 4096;
        torch::Tensor scores = x.index({ torch::indexing::Slice(), 4 });

        torch::Tensor indices = nms_kernel(boxes, scores, iou_threshold);

        outputs[i] = x.index({ indices });
    }

    return outputs;
}

std::vector<torch::Tensor> YoloV5Impl::forward_backbone(torch::Tensor x)
{
    std::vector<torch::Tensor> outputs(module_list->size() - 1);

    for (size_t i = 0; i < module_list->size() - 1; ++i)
    {
        YoloV5BlockImpl* block = module_list[i]->as<YoloV5Block>();

        std::vector<torch::Tensor> inputs(block->from.size());
        for (size_t j = 0; j < block->from.size(); ++j)
        {
            if (block->from[j] == -1)
            {
                inputs[j] = x;
            }
            else
            {
                inputs[j] = outputs[block->from[j]];
            }
        }

        x = block->forward(inputs);

        if (save[i])
        {
            outputs[i] = x;
        }
    }

    return outputs;
}

void YoloV5Impl::ParseConfig(const std::string& config_path)
{
    std::ifstream file(config_path, std::ios::in);
    std::vector<char> data((std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    file.close();

    auto s = ryml::csubstr(data.data(), data.size());
    ryml::Tree config = ryml::parse(s);

    float depth_multiple, width_multiple;
    config["depth_multiple"] >> depth_multiple;
    config["width_multiple"] >> width_multiple;

    std::vector<std::vector<int> > anchors;
    config["anchors"] >> anchors;

    const int num_anchors = anchors[0].size() / 2;

    int num_class;
    config["nc"] >> num_class;

    const int num_output = num_anchors * (num_class + 5);

    // Output channels is initialized with the input
    // so the -1 from of the first layers can use it
    std::vector<int> output_channels{ num_in_channels };

    ryml::NodeRef backbone = config["backbone"];
    ryml::NodeRef head = config["head"];

    std::vector<int> from;
    int block_depth;
    std::string module_name;
    ryml::NodeRef args;

    save = std::vector<bool>(backbone.num_children() + head.num_children(), false);

    // That's a bit messy but the job is done ¯\_("-")_/¯
    for (size_t i = 0; i < backbone.num_children() + head.num_children(); ++i)
    {
        auto v = i < backbone.num_children() ? backbone[i] : head[i - backbone.num_children()];

        if (v[0].is_seq())
        {
            v[0] >> from;
        }
        else
        {
            int val;
            v[0] >> val;
            from = { val };
        }

        v[1] >> block_depth;
        if (block_depth > 1)
        {
            block_depth = std::max(static_cast<int>(std::round(block_depth * depth_multiple)), 1);
        }

        v[2] >> module_name;

        args = v[3];

        int channel_out;
        torch::nn::Sequential internal_seq;
        KnownBlock module_type;

        if (module_name == "Concat")
        {
            module_type = KnownBlock::Concat;

            channel_out = 0;
            for (auto c : from)
            {
                if (c == -1)
                {
                    c = output_channels.size() - 1;
                }
                channel_out += output_channels[c];
            }

            int dimension;
            args[0] >> dimension;
            for (size_t j = 0; j < block_depth; j++)
            {
                internal_seq->push_back(Concat(dimension));
            }
        }
        else if (module_name == "Detect")
        {
            module_type = KnownBlock::Detect;

            channel_out = 0;

            std::vector<int> output_convs;
            output_convs.reserve(from.size());

            for (auto c : from)
            {
                if (c == -1)
                {
                    c = output_channels.size() - 1;
                }
                output_convs.push_back(output_channels[c]);
            }
            for (size_t j = 0; j < block_depth; j++)
            {
                internal_seq->push_back(Detect(num_class, anchors, output_convs));
            }
        }
        else if (module_name == "nn.Upsample")
        {
            int channel_in = 0;
            if (from[0] == -1)
            {
                channel_in = output_channels[output_channels.size() - 1];
            }
            else
            {
                channel_in = output_channels[from[0]];
            }
            channel_out = channel_in;

            module_type = KnownBlock::Upsample;

            double scale_factor;
            args[1] >> scale_factor;
            std::string mode;
            args[2] >> mode;

            if (mode == "nearest")
            {
                for (size_t j = 0; j < block_depth; j++)
                {
                    internal_seq->push_back(
                        torch::nn::Upsample(
                            torch::nn::UpsampleOptions()
                            .scale_factor(std::vector<double>({ scale_factor, scale_factor }))
                            .mode(torch::kNearest)
                    ));
                }
            }
            else
            {
                // TODO? Too lazy
            }
        }
        else
        {
            int channel_in = 0;
            if (from[0] == -1)
            {
                channel_in = output_channels[output_channels.size() - 1];
            }
            else
            {
                channel_in = output_channels[from[0]];
            }

            args[0] >> channel_out;
            if (channel_out != num_output)
            {
                channel_out = make_divisible(channel_out * width_multiple, 8);
            }

            // C3 is a bit special as the depth seq loop is already included
            if (module_name == "C3")
            {
                module_type = KnownBlock::C3;
                bool shortcut = true;
                if (args.num_children() > 1)
                {
                    args[1] >> shortcut;
                }

                internal_seq->push_back(C3(channel_in, channel_out,
                    block_depth, shortcut));
            }
            else if (module_name == "Conv")
            {
                module_type = KnownBlock::Conv;
                int kernel_size, stride;
                args[1] >> kernel_size;
                args[2] >> stride;

                for (size_t j = 0; j < block_depth; j++)
                {
                    internal_seq->push_back(Conv(channel_in, channel_out,
                        kernel_size, stride));
                }
            }
            else if (module_name == "Focus")
            {
                module_type = KnownBlock::Focus;
                int kernel_size;
                args[1] >> kernel_size;

                for (size_t j = 0; j < block_depth; j++)
                {
                    internal_seq->push_back(Focus(channel_in, channel_out,
                        kernel_size));
                }
            }
            else if (module_name == "SPP")
            {
                module_type = KnownBlock::SPP;
                std::vector<int> kernel_sizes;
                args[1] >> kernel_sizes;

                for (size_t j = 0; j < block_depth; j++)
                {
                    internal_seq->push_back(SPP(channel_in, channel_out,
                        kernel_sizes));
                }
            }
        }

        for (auto c : from)
        {
            if (c != -1)
            {
                save[c] = true;
            }
        }

        module_list->push_back(YoloV5Block(i, from, module_type, internal_seq));

        // Clean the initial 3 in the output_channels
        if (i == 0)
        {
            output_channels = {};
        }
        output_channels.push_back(channel_out);
    }

    std::cout << "Model successfully loaded from " << config_path << std::endl;
}

void YoloV5Impl::SetStride()
{
    int s = 256;
    torch::Tensor backbone_input = torch::zeros({ 1, num_in_channels, s, s });
    std::vector<torch::Tensor> backbone_outputs = forward_backbone(backbone_input);

    YoloV5BlockImpl* detect = module_list[module_list->size() - 1]->as<YoloV5Block>();
    torch::Tensor strides = torch::zeros(detect->from.size());

    for (size_t i = 0; i < detect->from.size(); ++i)
    {
        int c = detect->from[i];
        if (c < 0)
        {
            c = backbone_outputs.size() + c;
        }
        strides[i] = s / backbone_outputs[c].size(-2);
    }
    detect->seq[0]->as<Detect>()->SetStride(strides);
}
