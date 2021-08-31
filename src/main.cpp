#include <torch/torch.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>

#include "TheMachine/yolov5.hpp"
#include "TheMachine/utils.hpp"

const static std::vector<std::string> class_names = { "person", "bicycle", "car", "motorcycle",
"airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
"stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
"cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
"tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
"baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
"chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
"mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
"refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };

struct ResizedImage
{
    cv::Mat im;
    float ratio;
    int pad_x;
    int pad_y;
};

ResizedImage ResizeImage(const cv::Mat& im, const int new_shape, const int stride)
{
    const float ratio = std::min(static_cast<float>(new_shape) / im.rows, static_cast<float>(new_shape) / im.cols);

    if (ratio == 1.0f)
    {
        return ResizedImage{ im, 1.0f, 0, 0};
    }

    const int pad_h = std::round(im.rows * ratio);
    const int pad_w = std::round(im.cols * ratio);

    const float delta_w = (new_shape - pad_w) % stride / 2.0f;
    const float delta_h = (new_shape - pad_h) % stride / 2.0f;

    cv::Mat resized;
    cv::resize(im, resized, cv::Size(pad_w, pad_h), 0.0, 0.0, cv::INTER_LINEAR);
    
    const int top = std::round(delta_h - 0.1f);
    const int bottom = std::round(delta_h + 0.1f);
    const int left = std::round(delta_w - 0.1f);
    const int right = std::round(delta_w + 0.1f);

    cv::Mat output;
    cv::copyMakeBorder(resized, output, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));

    return ResizedImage{ output, ratio, left, top };
}

void PlotResults(cv::Mat& im, const torch::Tensor& results)
{
    const float* results_ptr = results.data_ptr<float>();

    // Plot in reverse order so highest score are drawn on top
    for (int i = results.size(0) - 1; i > -1; --i)
    {
        cv::rectangle(im, cv::Point(results_ptr[i * 6 + 0], results_ptr[i * 6 + 1]),
            cv::Point(results_ptr[i * 6 + 2], results_ptr[i * 6 + 3]), cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
        const std::string label = class_names[results_ptr[i * 6 + 5]] + "  " + std::to_string(results_ptr[i * 6 + 4]);
        int baseline;
        cv::Size text_size = cv::getTextSize(label, 0, 1, 3, &baseline);
        cv::rectangle(im, cv::Point(results_ptr[i * 6 + 0], results_ptr[i * 6 + 1]),
            cv::Point(results_ptr[i * 6 + 0] + text_size.width, results_ptr[i * 6 + 1] - text_size.height - 3),
            cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
        cv::putText(im, label, cv::Point(results_ptr[i * 6 + 0], results_ptr[i * 6 + 1] - 2),
            0, 1, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

torch::Tensor UnAugmentResults(torch::Tensor results, const ResizedImage& im)
{
    results.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 3, 2) }) -= im.pad_x;
    results.index({ torch::indexing::Slice(), torch::indexing::Slice(1, 4, 2) }) -= im.pad_y;
    results.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 4) }) /= im.ratio;

    return results;
}

int main(int argc, char* argv[])
{
    // Disable gradients
    torch::NoGradGuard no_grad;

    YoloV5 yolo = YoloV5("yolov5s.yaml", 3);

    {
        PythonWeightsFile weights("yolov5s.pt");
        yolo->LoadWeights(weights);
    }
    yolo->FuseConvAndBN();
    yolo->eval();

    int max_stride = yolo->GetMaxStride();
    std::vector<std::string> names = { "bus.jpg", "zidane.jpg", "dog_bike_car.jpg", "street.jpg", "new_york.jpg" };

    for (auto &s : names)
    {
        std::cout << "processing image " << s << std::endl;
        // Load image (HWC, B,G,R)
        cv::Mat base_image = cv::imread(s, cv::IMREAD_COLOR);
        ResizedImage img = ResizeImage(base_image, 640, max_stride);
        // Copy data into tensor
        torch::Tensor input = torch::zeros({ img.im.rows, img.im.cols, img.im.channels() }, torch::TensorOptions().dtype(torch::kByte));
        std::copy(img.im.ptr(), img.im.ptr() + img.im.rows * img.im.cols * img.im.channels(), reinterpret_cast<unsigned char*>(input.data_ptr()));
        // Convert to CHW, R,G,B
        input = input.permute({ 2,0,1 }).flip(0).to(torch::kFloat) / 255.0f;

        torch::Tensor output = yolo->forward(input.unsqueeze(0));

        std::cout << "output size:" << std::endl;
        std::cout << output.sizes() << std::endl;

        std::vector<torch::Tensor> nms = yolo->NonMaxSuppression(output, 0.25f, 0.45f);

        std::cout << nms[0] << std::endl;

        PlotResults(img.im, nms[0]);
        PlotResults(base_image, UnAugmentResults(nms[0], img));

        cv::imshow("base_image", base_image);
        cv::imshow("img", img.im);
        cv::waitKey(0);
    }

    return 0;
}