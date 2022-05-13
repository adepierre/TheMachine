#include <ryml_std.hpp>
#include <ryml.hpp>

#include <WeightsLoading/weights_loader.hpp>

#include "YoloV5/yolov5.hpp"
#include "YoloV5/layers.hpp"
#include "YoloV5/utils.hpp"


YoloV5BlockImpl::YoloV5BlockImpl(const std::vector<int>& from_,
    const KnownBlock type_, torch::nn::Sequential seq_)
{
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

const std::vector<int>& YoloV5BlockImpl::From() const
{
    return from;
}

const KnownBlock YoloV5BlockImpl::Type() const
{
    return type;
}



YoloV5Impl::YoloV5Impl(const std::string& config_path, const int num_in_channels_)
{
    num_in_channels = num_in_channels_;
    ParseConfig(config_path);
    register_module("module_list", module_list);
    SetStride();
    InitWeights();
}

YoloV5Impl::~YoloV5Impl()
{

}

torch::Tensor YoloV5Impl::forward(torch::Tensor x)
{
    std::vector<torch::Tensor> backbone_outputs = forward_backbone(x);

    YoloV5BlockImpl* detect = module_list[module_list->size() - 1]->as<YoloV5Block>();
    std::vector<torch::Tensor> detect_inputs(detect->From().size());

    for (size_t i = 0; i < detect->From().size(); ++i)
    {
        int c = detect->From()[i];
        if (c < 0)
        {
            c = backbone_outputs.size() + c;
        }
        detect_inputs[i] = backbone_outputs[c];
    }

    return detect->forward(detect_inputs);
}

void YoloV5Impl::LoadWeights(const std::string& weights_file)
{
    PythonWeightsFile weights(weights_file);

    size_t counter_params = 0;
    size_t counter_buffers = 0;
    for (auto& submodule : modules())
    {
        for (auto& p : submodule->parameters(false))
        {
            CopyRawDataToTensor(weights.GetNextTensor(), p);
            counter_params += 1;
        }

        for (auto& b : submodule->buffers(false))
        {
            CopyRawDataToTensor(weights.GetNextTensor(), b);
            counter_buffers += 1;
        }
    }

    std::cout << counter_params << " parameter tensors successfully loaded" << std::endl;
    std::cout << counter_buffers << " buffer tensors successfully loaded" << std::endl;
}

std::vector<torch::Tensor> YoloV5Impl::NonMaxSuppression(torch::Tensor prediction, 
    float conf_threshold, float iou_threshold)
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

        // Compute overall conf (obj_conf * cls_conf)
        x.index({ torch::indexing::Slice(), torch::indexing::Slice(5, torch::indexing::None) }) *=
            x.index({ torch::indexing::Slice(), torch::indexing::Slice(4, 5) });

        torch::Tensor box = xywh2xyxy(x.index({ torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 4) }));

        torch::Tensor conf, j;
        std::tie(conf, j) = x.index({ torch::indexing::Slice(), torch::indexing::Slice(5, torch::indexing::None) }).max(1, true);

        x = torch::cat({ box, conf, j.to(torch::kFloat32) }, 1).index({ conf.view({-1}) > conf_threshold });

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

int YoloV5Impl::GetMaxStride() const
{
    return strides.max().item<float>();
}

void YoloV5Impl::FuseConvAndBN()
{
    apply([](torch::nn::Module& m)
        {
            if (auto* conv = m.as<Conv>())
            {
                conv->FuseConvAndBN();
            }
        });
    std::cout << "Batchnorms fused into convs" << std::endl;
}

std::vector<torch::Tensor> YoloV5Impl::forward_backbone(torch::Tensor x)
{
    std::vector<torch::Tensor> outputs(module_list->size() - 1);

    for (size_t i = 0; i < module_list->size() - 1; ++i)
    {
        YoloV5BlockImpl* block = module_list[i]->as<YoloV5Block>();

        std::vector<torch::Tensor> inputs(block->From().size());
        for (size_t j = 0; j < block->From().size(); ++j)
        {
            if (block->From()[j] == -1)
            {
                inputs[j] = x;
            }
            else
            {
                inputs[j] = outputs[block->From()[j]];
            }
        }
        x = block->forward(inputs);

        if (save_module_output[i])
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

    save_module_output = std::vector<bool>(backbone.num_children() + head.num_children(), false);

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
                // We set the anchors from the yaml file to get the right
                // tensor shape, but they will be overloaded with the ones
                // saved in the .pt file anyway.
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
                int kernel_size, stride, padding;
                args[1] >> kernel_size;
                args[2] >> stride;
                if (args.num_children() > 3)
                {
                    args[3] >> padding;
                }
                else
                {
                    padding = -1;
                }

                for (size_t j = 0; j < block_depth; j++)
                {
                    internal_seq->push_back(Conv(channel_in, channel_out,
                        kernel_size, stride, padding));
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
            else if (module_name == "SPPF")
            {
                module_type = KnownBlock::SPPF;
                int kernel_size;
                args[1] >> kernel_size;

                for (size_t j = 0; j < block_depth; j++)
                {
                    internal_seq->push_back(SPPF(channel_in, channel_out,
                        kernel_size));
                }
            }
            else
            {
                throw std::runtime_error("Unknown module name in model file: " + module_name);
            }
        }

        for (auto c : from)
        {
            if (c != -1)
            {
                save_module_output[c] = true;
            }
        }

        module_list->push_back(YoloV5Block(from, module_type, internal_seq));

        // Clean the initial num_in_channels in the output_channels
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
    strides = torch::zeros(detect->From().size());

    for (size_t i = 0; i < detect->From().size(); ++i)
    {
        int c = detect->From()[i];
        if (c < 0)
        {
            c = backbone_outputs.size() + c;
        }
        strides[i] = s / backbone_outputs[c].size(-2);
    }

    if (detect->Type() == KnownBlock::Detect)
    {
        detect->children()[0]->as<torch::nn::Sequential>()->at<DetectImpl>(0).SetStride(strides);
    }
}

void YoloV5Impl::InitWeights()
{
    apply([](torch::nn::Module& m)
        {
            if (auto* l = m.as<torch::nn::BatchNorm2d>())
            {
                l->options.eps(1e-3);
                l->options.momentum(0.03);
            }
        });
}
