#include "TheMachine/TheMachine.hpp"

// person
// car, truck
// boat, train, bus
// airplane

int main(int argc, char* argv[])
{
    // Disable gradients
    torch::NoGradGuard no_grad;

    std::vector<std::string> names = { "bus.jpg", "zidane.jpg", "dog_bike_car.jpg", "street.jpg", "new_york.jpg" };

    TheMachine machine("../data/yolov5s.yaml", "../data/yolov5s.pt", 640, torch::kCPU, true);

    for (auto &s : names)
    {
        const std::string full_path = "../data/" + s;
        machine.Detect(full_path);
    }

    return 0;
}