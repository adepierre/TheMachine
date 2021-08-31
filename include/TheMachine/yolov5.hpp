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

class PythonWeightsFile;

class YoloV5BlockImpl : public torch::nn::Module
{
public:
	YoloV5BlockImpl(const std::vector<int>& from_,
		const KnownBlock type_, torch::nn::Sequential seq_);
	~YoloV5BlockImpl();

	torch::Tensor forward(std::vector<torch::Tensor> x);
	const std::vector<int>& From() const;
	const KnownBlock Type() const;

private:
	std::vector<int> from;
	KnownBlock type;
	// All submodules are stored in a seq,
	// so we can store individual modules as 
	// well as seq. With a torch::nn::AnyModule,
	// we wouldn't be able to store a seq
	torch::nn::Sequential seq;
};
TORCH_MODULE(YoloV5Block);

class YoloV5Impl : public torch::nn::Module
{
public:
	YoloV5Impl(const std::string& config_path, const int num_in_channels_);
	~YoloV5Impl();

	torch::Tensor forward(torch::Tensor x);

	void LoadWeights(PythonWeightsFile& weights);

	int GetMaxStride() const;

	void FuseConvAndBN();

	/// <summary>
	/// Perform NMS on forward results.
	/// </summary>
	/// <param name='prediction'>Input detections, [batch, num det, num class + 5]</param>
	/// <param name='conf_threshold'>Confidence threshold</param>
	/// <param name='iou_threshold'>IoU threshold</param>
	/// <param name='max_det'>Max detection per image</param>
	/// <param name='class_filter'>If not empty, only consider classes present in this vector</param>
	/// <returns>A vector of size batch. For each image a tensor of size [num det kept, 6 (x1, y1, x2, y2, conf, cls)]</returns>
	std::vector<torch::Tensor> NonMaxSuppression(torch::Tensor prediction, float conf_threshold = 0.25f,
		float iou_threshold = 0.45f, const std::vector<int>& class_filter = {});

private:
	std::vector<torch::Tensor> forward_backbone(torch::Tensor x);
	void ParseConfig(const std::string& config_path);
	void SetStride();
	void InitWeights();

private:
	torch::nn::ModuleList module_list;
	std::vector<bool> save;
	int num_in_channels;
	torch::Tensor strides;
};
TORCH_MODULE(YoloV5);
