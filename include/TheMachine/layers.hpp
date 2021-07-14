#pragma once

#include <torch/torch.h>

class ConvImpl : public torch::nn::Module
{
public:
	ConvImpl(int channels_in, int channels_out, 
		int kernel_size = 1, int stride = 1);
	~ConvImpl();
	torch::Tensor forward(torch::Tensor x);

private:
	torch::nn::Conv2d conv;
	torch::nn::BatchNorm2d bn;
	torch::nn::SiLU act;
};
TORCH_MODULE(Conv);




class FocusImpl : public torch::nn::Module
{
public:
	FocusImpl(int channels_in, int channels_out,
		int kernel_size = 1, int stride = 1);
	~FocusImpl();

	torch::Tensor forward(torch::Tensor x);

private:
	Conv conv;
};
TORCH_MODULE(Focus);




class BottleneckImpl : public torch::nn::Module
{
public:
	BottleneckImpl(int channels_in, int channels_out,
		bool shortcut = true, float expansion = 0.5f);
	~BottleneckImpl();

	torch::Tensor forward(torch::Tensor x);

private:
	Conv cv1, cv2;
	bool add;
};
TORCH_MODULE(Bottleneck);




class C3Impl : public torch::nn::Module
{
public:
	C3Impl(int channels_in, int channels_out,
		int n = 1, bool shortcut = true, float expansion = 0.5f);
	~C3Impl();

	torch::Tensor forward(torch::Tensor x);

private:
	Conv cv1, cv2, cv3;
	torch::nn::Sequential m;
};
TORCH_MODULE(C3);




class SPPImpl : public torch::nn::Module
{
public:
	SPPImpl(int channels_in, int channels_out,
		std::vector<int> kernel_sizes = { 5, 9, 13 });
	~SPPImpl();

	torch::Tensor forward(torch::Tensor x);

private:
	Conv cv1, cv2;
	torch::nn::ModuleList m;
};
TORCH_MODULE(SPP);




class DetectImpl : public torch::nn::Module
{
public:
	DetectImpl(int channels_in, int channels_out,
		std::vector<int> kernel_sizes = { 5, 9, 13 });
	~DetectImpl();

	torch::Tensor forward(torch::Tensor x);

private:

};
TORCH_MODULE(Detect);