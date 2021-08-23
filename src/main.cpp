#include <torch/torch.h>
#include <iostream>
#include <fstream>

#include "TheMachine/layers.hpp"
#include "TheMachine/yolov5.hpp"


int main(int argc, char* argv[])
{
    //try
    //{
        YoloV5 yolo = YoloV5("yolov5s.yaml", 3);
        yolo->eval();

        torch::Tensor input = torch::zeros({ 2,3,320,320 });

        torch::Tensor output = yolo->forward(input);

        std::cout << "output size:" << std::endl;
        std::cout << output.sizes() << std::endl;
        std::cout << output[0][1] << std::endl;
    //}
	//catch (const std::exception& e)
	//{
    //    std::cout << e.what() << std::endl;
	//}

    return 0;
}