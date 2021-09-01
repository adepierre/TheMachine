#include "TheMachine/TheMachine.hpp"

void ShowHelp(const char* argv0)
{
    std::cout << "Usage: " << argv0 << " <options>\n"
        << "Options:\n"
        << "\t-h, --help\tShow this help message\n"
        << "\t--model\tPath to the yaml file to use to build YoloV5 net, default: yolov5s.yaml\n"
        << "\t--weights\tPath to the .pt file containing trained weights for YoloV5, default: yolov5s.pt\n"
        << "\t--path\tPath to the image to process, default: empty\n"
        << "\t--save\tIf set, save the resulting image on the disk, default: empty\n"
        << "\t--gpu\tIf set, will try to use the GPU for inference, otherwise use the CPU\n"
        << "\t--simple_ui\tIf set, switch to basic YoloV5 without the machine UI\n"
        << std::endl;
}

int main(int argc, char* argv[])
{
    std::string model = "yolov5s.yaml";
    std::string weights = "yolov5s.pt";
    std::string path = "";
    std::string save = "";
    bool gpu = false;
    bool simple_ui = false;

    if (argc == 1)
    {
        ShowHelp(argv[0]);
        return 0;
    }

    // Parse arguments
    for (size_t i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help")
        {
            ShowHelp(argv[0]);
            return 0;
        }
        else if (arg == "--model")
        {
            if (i + 1 < argc)
            {
                model = argv[++i];
            }
            else
            {
                std::cerr << "--model requires an argument" << std::endl;
                return 1;
            }
        }
        else if (arg == "--weights")
        {
            if (i + 1 < argc)
            {
                weights = argv[++i];
            }
            else
            {
                std::cerr << "--weights requires an argument" << std::endl;
                return 1;
            }
        }
        else if (arg == "--path")
        {
            if (i + 1 < argc)
            {
                path = argv[++i];
            }
            else
            {
                std::cerr << "--path requires an argument" << std::endl;
                return 1;
            }
        }
        else if (arg == "--save")
        {
            if (i + 1 < argc)
            {
                save = argv[++i];
            }
            else
            {
                std::cerr << "--save requires an argument" << std::endl;
                return 1;
            }
        }
        else if (arg == "--gpu")
        {
            gpu = true;
        }
        else if (arg == "--simple_ui")
        {
            simple_ui = true;
        }
    }

    try
    {
        // Disable gradients
        torch::NoGradGuard no_grad;

        torch::Device device = torch::kCPU;
        if (gpu && torch::cuda::is_available())
        {
            device = torch::kCUDA;
        }

        TheMachine machine(model, weights, 640, device, simple_ui);

        machine.Detect(path, save);
    }
    catch (const std::exception& e)
    {
        std::cout << "EXCEPTION\n" << e.what() << std::endl;
    }

    return 0;
}