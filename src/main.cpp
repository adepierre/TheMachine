#include <torch/torch.h>
#include <iostream>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "TheMachine/yolov5.hpp"
#include "TheMachine/utils.hpp"


int main(int argc, char* argv[])
{
    YoloV5 yolo = YoloV5("yolov5s.yaml", 3);
    yolo->eval();

    {
        PythonWeightsFile weights("yolov5s.pt");
        yolo->LoadWeights(weights);
    }

    std::vector<std::string> names = { "bus.jpg", "zidane.jpg" };

    for (auto &s : names)
    {
        std::cout << "processing image " << s << std::endl;
        int x, y, n;
        unsigned char* data = stbi_load(s.c_str(), &x, &y, &n, 0);
        torch::Tensor input = torch::zeros({ 1, n, y, x }, torch::TensorOptions().dtype(torch::kByte));
        std::copy(data, data + n * y * x, reinterpret_cast<unsigned char*>(input.data_ptr()));
        input = input.to(torch::kFloat) / 255.0f;
        stbi_image_free(data);

        torch::Tensor output = yolo->forward(input);

        std::cout << "output size:" << std::endl;
        std::cout << output.sizes() << std::endl;

        std::vector<torch::Tensor> nms = yolo->NonMaxSuppression(output);

        std::cout << "nms size:" << std::endl;
        for (size_t i = 0; i < nms.size(); ++i)
        {
            std::cout << nms[i].sizes() << std::endl;
        }
    }

    return 0;
}