#pragma once

#include <random>

#include <opencv2/core.hpp>
#include <YoloV5/yolov5.hpp>

#include "TheMachine/utils.hpp"

/// <summary>
/// Simple struct to store a preprocessed image
/// </summary>
struct PreprocessedImage
{
	cv::Mat original;
	cv::Mat im;
	float ratio;
	int pad_x;
	int pad_y;
};

class TheMachine
{
public:
	/// <summary>
	/// Constructor
	/// </summary>
	/// <param name="detector_yaml_file">Yaml file describing the YoloV5
	/// architecture we want to use.</param>
	/// <param name="detector_weights_file">.pt file containing YoloV5 weights.</param>
	/// <param name="process_size_">The size of the images passed to the detector</param>
	/// <param name="device_">Torch device used for operations (default CPU)</param>
	/// <param name="boring_ui_">If true, display a simple rectangle around detections instead of cooler UI</param>
	TheMachine(const std::string& detector_yaml_file,
		const std::string& detector_weights_file,
		const int process_size_ = 640, const torch::Device device_ = torch::kCPU,
		const bool boring_ui_ = false);
	~TheMachine();

	TheMachine() = delete;
	TheMachine(const TheMachine&) = delete;

	/// <summary>
	/// Detect object in image and display the result.
	/// </summary>
	/// <param name="path">Image to process</param>
	/// <param name="save_path">If not empty, save the result here</param>
	void Detect(const std::string& path, const std::string& save_path = "");

private:
	PreprocessedImage Preprocess(const std::string& path);
	std::vector<Detection> PostProcess(const torch::Tensor& output_);
	void PlotResults(cv::Mat& img, const std::vector<Detection>& detections);

private:
	YoloV5 detector;
	int process_size;
	torch::Device device;
	std::vector<int> class_filter;
	std::mt19937 random_engine;
	std::uniform_int_distribution<int> color_distrib;
	bool boring_ui;
};