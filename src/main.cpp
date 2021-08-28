#include <torch/torch.h>
#include <iostream>
#include <fstream>

#include "TheMachine/layers.hpp"
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

    torch::Tensor input = torch::zeros({ 2,3,320,320 });

    torch::Tensor output = yolo->forward(input);

    std::cout << "output size:" << std::endl;
    std::cout << output.sizes() << std::endl;

    std::vector<torch::Tensor> nms = yolo->NonMaxSuppression(output);

    std::cout << "nms size:" << std::endl;
    for (size_t i = 0; i < nms.size(); ++i)
    {
        std::cout << nms[i].sizes() << std::endl;
    }

    return 0;
}