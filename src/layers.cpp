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
    int kernel_size, int stride) :
    conv(Conv(4 * channels_in, channels_out, kernel_size, stride))
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
    cv3(torch::cat({
        m->forward(cv1(x)),
        cv2(x)
        }, 1));
}




SPPImpl::SPPImpl(int channels_in, int channels_out,
    std::vector<int> kernel_sizes) :
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

    torch::Tensor concat = x;

    for (int i = 0; i < m->size(); ++i)
    {
        x = torch::cat({ concat, m[i]->as<torch::nn::MaxPool2d>()->forward(x) });
    }

    return cv2(concat);
}
