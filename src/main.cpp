#include <torch/torch.h>
#include <iostream>

int main(int argc, char* argv[])
{
    torch::Tensor t = torch::rand({ 2,3 });
    std::cout << t << std::endl;

    return 0;
}