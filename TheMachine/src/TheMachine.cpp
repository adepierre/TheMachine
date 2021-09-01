#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "TheMachine/TheMachine.hpp"

TheMachine::TheMachine(const std::string& detector_yaml_file, 
    const std::string& detector_weights_file, const int process_size_,
    const torch::Device device_, const bool boring_ui_) : detector(detector_yaml_file, 3),
    process_size(process_size_), device(device_), boring_ui(boring_ui_)
{
    detector->LoadWeights(detector_weights_file);
    detector->FuseConvAndBN();
    detector->eval();
    detector->to(device);

    if (!boring_ui)
    {
        // We're only interested in person, car, truck, bus,
        // airplane, boat and train, 
        // not in broccoli, hot dog or hair drier
        class_filter = { 0, 2, 4, 5, 6, 7, 8 };
    }

    std::random_device rd;
    random_engine = std::mt19937(rd());
    color_distrib = std::uniform_int_distribution<int>(0, 255);
}

TheMachine::~TheMachine()
{

}

void TheMachine::Detect(const std::string& path, const std::string& save_path)
{
    int max_stride = detector->GetMaxStride();

    std::cout << "Processing image: " << path << std::endl;

    PreprocessedImage img = Preprocess(path);

    // Create HWC, B, G, R tensor
    torch::Tensor input = torch::from_blob(img.im.data, { img.im.rows, img.im.cols, img.im.channels() }, torch::TensorOptions().dtype(torch::kByte));

    // Convert to CHW, R,G,B
    input = input.permute({ 2,0,1 }).flip(0);

    // Add one batch channel
    input = input.unsqueeze(0);

    // Transfer to GPU if necessary, then convert to float
    input = input.to(device).to(torch::kFloat) / 255.0f;

    // Pass the image through YoloV5 and apply NMS
    torch::Tensor output = detector->forward(input);
    output = detector->NonMaxSuppression(output)[0];

    // Retransform the output to get boxes wrt the original image
    output.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 3, 2) }) -= img.pad_x;
    output.index({ torch::indexing::Slice(), torch::indexing::Slice(1, 4, 2) }) -= img.pad_y;
    output.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 4) }) /= img.ratio;

    // Transform tensor into Detection
    std::vector<Detection> detections = PostProcess(output);

    // Draw the detections
    PlotResults(img.original, detections);

    if (!save_path.empty())
    {
        cv::imwrite(save_path, img.original);
    }

    std::cout << "Press any key to quit..." << std::endl;

    cv::imshow("TheMachine", img.original);
    cv::waitKey(0);
}

PreprocessedImage TheMachine::Preprocess(const std::string& path)
{
    // Load image (HWC, B,G,R)
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);

    const float ratio = std::min(static_cast<float>(process_size) / img.rows, static_cast<float>(process_size) / img.cols);

    if (ratio == 1.0f)
    {
        return PreprocessedImage{ img, img, 1.0f, 0, 0 };
    }

    const int stride = detector->GetMaxStride();

    const int pad_h = std::round(img.rows * ratio);
    const int pad_w = std::round(img.cols * ratio);

    const float delta_w = (process_size - pad_w) % stride / 2.0f;
    const float delta_h = (process_size - pad_h) % stride / 2.0f;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(pad_w, pad_h), 0.0, 0.0, cv::INTER_LINEAR);

    const int top = std::round(delta_h - 0.1f);
    const int bottom = std::round(delta_h + 0.1f);
    const int left = std::round(delta_w - 0.1f);
    const int right = std::round(delta_w + 0.1f);

    cv::Mat output;
    cv::copyMakeBorder(resized, output, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));

    return PreprocessedImage{img, output, ratio, left, top };
}

std::vector<Detection> TheMachine::PostProcess(const torch::Tensor& output_)
{
    torch::Tensor output = output_.cpu();

    std::vector<Detection> detections;
    detections.reserve(output.size(0));

    const float* results_ptr = output.data_ptr<float>();

    // Revert order so highest scores are first
    for (int i = output.size(0) - 1; i > -1; --i)
    {
        const int cls = static_cast<int>(results_ptr[i * 6 + 5]);
        bool kept_class = class_filter.size() == 0;
        for (size_t j = 0; j < class_filter.size(); j++)
        {
            if (class_filter[j] == cls)
            {
                kept_class = true;
                break;
            }
        }
        if (kept_class)
        {
            detections.push_back({ results_ptr[i * 6 + 0], results_ptr[i * 6 + 1] ,
            results_ptr[i * 6 + 2] , results_ptr[i * 6 + 3],
            results_ptr[i * 6 + 4], cls });
        }
    }

    return detections;
}

void TheMachine::PlotResults(cv::Mat& img, const std::vector<Detection>& detections)
{
    for (size_t i = 0; i < detections.size(); ++i)
    {
        const cv::Scalar color(color_distrib(random_engine), color_distrib(random_engine), color_distrib(random_engine));

        if (boring_ui)
        {
            DrawRectangle(img, detections[i], color);
        }
        else
        {
            DrawDetection(img, detections[i], color);
        }
    }
}

