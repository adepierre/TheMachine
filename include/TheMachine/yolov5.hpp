#pragma once

#include <torch/torch.h>

enum class KnownBlock
{
	Focus,
	Conv,
	C3,
	SPP,
	Upsample,
	Concat,
	Detect
};

struct YoloV5BlockImpl : public torch::nn::Module
{
	YoloV5BlockImpl(const int attach_index_,
		const std::vector<int>& from_, const KnownBlock type_,
		torch::nn::Sequential seq_);
	~YoloV5BlockImpl();

	torch::Tensor forward(std::vector<torch::Tensor> x);

	int attach_index;
	std::vector<int> from;
	KnownBlock type;

	torch::nn::Sequential seq;
};
TORCH_MODULE(YoloV5Block);

class YoloV5Impl : public torch::nn::Module
{
public:
	YoloV5Impl(const std::string& config_path, const int num_in_channels_);
	~YoloV5Impl();

	torch::Tensor forward(torch::Tensor x);

private:
	std::vector<torch::Tensor> forward_backbone(torch::Tensor x);
	void ParseConfig(const std::string& config_path);
	void SetStride();

private:
	torch::nn::ModuleList module_list;
	std::vector<bool> save;
	int num_in_channels;
};
TORCH_MODULE(YoloV5);
