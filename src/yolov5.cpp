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
            // assert(from.size() == 1);
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
            // assert(from.size() == 1);
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
        strides[i] = s / backbone_outputs[c].sizes()[backbone_outputs[c].sizes().size() - 2];
    }
    detect->seq[0]->as<Detect>()->SetStride(strides);
}
