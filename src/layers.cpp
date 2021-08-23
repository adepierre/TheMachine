#include "TheMachine/layers.hpp"


ConvImpl::ConvImpl(int channels_in, int channels_out,
    int kernel_size, int stride) :
    conv(torch::nn::Conv2dOptions(channels_in, channels_out, kernel_size)
        .stride(stride).padding(kernel_size / 2).bias(false)),
    bn(torch::nn::BatchNorm2d(channels_out)),
    act(torch::nn::SiLU())
{
    register_module("conv", conv);
    register_module("bn", bn);
    register_module("act", act);
}

ConvImpl::~ConvImpl()
{

}

torch::Tensor ConvImpl::forward(torch::Tensor x)
{
    return act(bn(conv(x)));
}





FocusImpl::FocusImpl(int channels_in, int channels_out,
    int kernel_size) :
    conv(Conv(4 * channels_in, channels_out, kernel_size))
{
    register_module("conv", conv);
}

FocusImpl::~FocusImpl()
{
}

// x(b, c, w, h)-->y(b, 4c, w / 2, h / 2)
torch::Tensor FocusImpl::forward(torch::Tensor x)
{
    return conv(torch::cat({
            x.index({"...", torch::indexing::Slice(0, torch::indexing::None, 2), torch::indexing::Slice(0, torch::indexing::None, 2) }),
            x.index({"...", torch::indexing::Slice(1, torch::indexing::None, 2), torch::indexing::Slice(1, torch::indexing::None, 2) }),
            x.index({"...", torch::indexing::Slice(0, torch::indexing::None, 2), torch::indexing::Slice(1, torch::indexing::None, 2) }),
            x.index({"...", torch::indexing::Slice(1, torch::indexing::None, 2), torch::indexing::Slice(1, torch::indexing::None, 2) })        
        }, 1));
}







ConcatImpl::ConcatImpl(int dimension_)
{
    dimension = dimension_;
}

ConcatImpl::~ConcatImpl()
{
}

torch::Tensor ConcatImpl::forward(std::vector<torch::Tensor> x)
{
    return torch::cat(x, dimension);
}







BottleneckImpl::BottleneckImpl(int channels_in, int channels_out,
    bool shortcut, float expansion) :
    cv1(Conv(channels_in, (int)(channels_out * expansion), 1, 1)),
    cv2(Conv((int)(channels_out* expansion), channels_out, 3, 1))
{
    register_module("cv1", cv1);
    register_module("cv2", cv2);
    add = shortcut && channels_in == channels_out;
}

BottleneckImpl::~BottleneckImpl()
{
}

torch::Tensor BottleneckImpl::forward(torch::Tensor x)
{
    return add ? x + cv2(cv1(x)) : cv2(cv1(x));
}




C3Impl::C3Impl(int channels_in, int channels_out,
    int n, bool shortcut, float expansion) :
    cv1(Conv(channels_in, (int)(channels_out * expansion), 1, 1)),
    cv2(Conv(channels_in, (int)(channels_out * expansion), 1, 1)),
    cv3(Conv(2 * (int)(channels_out * expansion), channels_out, 1, 1))
{
    register_module("cv1", cv1);
    register_module("cv2", cv2);
    register_module("cv3", cv3);
    const int hidden = (int)(channels_out * expansion);
    for (int i = 0; i < n; ++i)
    {
        m->push_back(Bottleneck(hidden, hidden, shortcut, 1.0f));
    }
    register_module("m", m);
}

C3Impl::~C3Impl()
{

}

torch::Tensor C3Impl::forward(torch::Tensor x)
{
    return cv3(torch::cat({
        m->forward(cv1(x)),
        cv2(x)
        }, 1));
}




SPPImpl::SPPImpl(const int channels_in, const int channels_out,
    const std::vector<int>& kernel_sizes) :
    cv1(Conv(channels_in, channels_in / 2, 1, 1)),
    cv2(Conv(channels_in / 2 * (kernel_sizes.size() + 1), channels_out, 1, 1))
{
    register_module("cv1", cv1);
    register_module("cv2", cv2);
    for (int i = 0; i < kernel_sizes.size(); ++i)
    {
        m->push_back(torch::nn::MaxPool2d(
            torch::nn::MaxPool2dOptions(kernel_sizes[i])
            .stride(1)
            .padding(kernel_sizes[i] / 2)
        ));
    }
    register_module("m", m);
}

SPPImpl::~SPPImpl()
{

}

torch::Tensor SPPImpl::forward(torch::Tensor x)
{
    x = cv1(x);

    std::vector<torch::Tensor> concats(m->size() + 1);
    concats[0] = x;

    for (int i = 0; i < m->size(); ++i)
    {
        concats[i + 1] = m[i]->as<torch::nn::MaxPool2d>()->forward(x);
    }

    return cv2(torch::cat(concats, 1));
}




DetectImpl::DetectImpl(const int nc, const std::vector<std::vector<int> >& anchors_,
    const std::vector<int>& output_convs)
{
    num_class = nc;
    num_output_per_anchor = nc + 5;
    num_detection_layers = anchors_.size();
    num_anchors = anchors_[0].size() / 2;

    // Init grid with dummy tensors
    grid.reserve(num_detection_layers);
    for (int i = 0; i < num_detection_layers; ++i)
    {
        grid.push_back(torch::zeros({ 1,1,0,0,2 }, torch::kFloat32));
    }

    anchors = torch::zeros({ num_detection_layers, num_anchors, 2 }, torch::kFloat32);

    for (int i = 0; i < num_detection_layers; ++i)
    {
        for (int j = 0; j < num_anchors; ++j)
        {
            anchors[i][j][0] = anchors_[i][2 * j + 0];
            anchors[i][j][1] = anchors_[i][2 * j + 1];
        }
    }

    anchor_grid = anchors.clone().view({ num_detection_layers, 1, -1, 1, 1, 2 });

    register_buffer("anchors", anchors);
    register_buffer("anchor_grid", anchor_grid);

    for (int i = 0; i < output_convs.size(); ++i)
    {
        m->push_back(torch::nn::Conv2d(
            output_convs[i], num_output_per_anchor * num_anchors, 1));
    }
    register_module("m", m);
}

DetectImpl::~DetectImpl()
{

}

torch::Tensor DetectImpl::forward(std::vector<torch::Tensor> x)
{
    std::vector<torch::Tensor> z;
    z.reserve(x.size());

    for (int i = 0; i < num_detection_layers; ++i)
    {
        x[i] = m[i]->as<torch::nn::Conv2d>()->forward(x[i]);
        const auto shape = x[i].sizes();
        const int batch_size = shape[0];
        const int n_y = shape[2];
        const int n_x = shape[3];
        x[i] = x[i]
            .view({ batch_size, num_anchors, num_output_per_anchor, n_y, n_x })
            .permute({ 0, 1, 3, 4, 2 })
            .contiguous();

        // Apply the grid transformation to the data
        if (grid[i].sizes()[2] != x[i].sizes()[2] ||
            grid[i].sizes()[3] != x[i].sizes()[3])
        {
            grid[i] = make_grid(n_x, n_y).to(x[i].device());
        }

        torch::Tensor y = x[i].sigmoid();
        y.index({ "...", torch::indexing::Slice(0, 2) }) =
            (y.index({ "...", torch::indexing::Slice(0, 2) }) * 2.0f - 0.5f + grid[i]) * stride[i];
        y.index({ "...", torch::indexing::Slice(2, 4) }) =
            (y.index({ "...", torch::indexing::Slice(2, 4) }) * 2.0f).pow(2) * anchor_grid[i];

        z.push_back(y.view({ batch_size, -1, num_output_per_anchor }));
    }

    return torch::cat(z, 1);
}

torch::Tensor DetectImpl::make_grid(int n_x, int n_y)
{
    std::vector<torch::Tensor> meshgrids = torch::meshgrid({ torch::arange(n_y), torch::arange(n_x) });
    return torch::stack({ meshgrids[1], meshgrids[0] }, 2).view({ 1,1,n_y, n_x, 2 }).to(torch::kFloat32);
}

void DetectImpl::SetStride(torch::Tensor stride_)
{
    stride = stride_;
}
