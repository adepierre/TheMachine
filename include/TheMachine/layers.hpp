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
		int kernel_size = 1);
	~FocusImpl();

	torch::Tensor forward(torch::Tensor x);

private:
	Conv conv;
};
TORCH_MODULE(Focus);




class ConcatImpl : public torch::nn::Module
{
public:
	ConcatImpl(int dimension_);
	~ConcatImpl();

	torch::Tensor forward(std::vector<torch::Tensor> x);

private:
	int dimension;
};
TORCH_MODULE(Concat);




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
	SPPImpl(const int channels_in, const int channels_out,
		const std::vector<int>& kernel_sizes = { 5, 9, 13 });
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
	DetectImpl(const int nc, const std::vector<std::vector<int> >& anchors_,
		const std::vector<int>& output_convs);
	~DetectImpl();

	torch::Tensor forward(std::vector<torch::Tensor> x);

	void SetStride(torch::Tensor stride_);

private:
	static torch::Tensor make_grid(int n_x, int n_y);

private:
	torch::Tensor stride;
	int num_class;
	int num_output_per_anchor;
	int num_detection_layers;
	int num_anchors;
	torch::Tensor anchors;
	torch::Tensor anchor_grid;
	std::vector<torch::Tensor> grid;
	torch::nn::ModuleList m;
};
TORCH_MODULE(Detect);