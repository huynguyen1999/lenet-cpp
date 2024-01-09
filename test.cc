#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "src/layer.h"
#include "src/layer/conv.h"
#include "src/layer/conv_gpu.h"
#include "src/layer/fully_connected.h"
#include "src/layer/ave_pooling.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse_loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"
#include "src/dnn_network.h"

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <CNN_Type>" << std::endl;
        return 1;
    }
    // 0. Get CNN version
    const char *version = argv[1];
    int cnn_version = -1;
    if (std::strcmp(version, "cpu") == 0)
    {
        cnn_version = 0;
        std::cout << "Selected CNN Type: CPU" << std::endl;
    }
    else if (std::strcmp(version, "gpu") == 0)
    {
        cnn_version = 1;
        std::cout << "Selected CNN Type: Basic GPU" << std::endl;
    }
    else if (std::strcmp(version, "optimize1") == 0)
    {
        cnn_version = 2;
        std::cout << "Selected CNN Type: Optimized GPU With Shared Memory " << std::endl;
    }
    else if (std::strcmp(version, "optimize2") == 0)
    {
        cnn_version = 3;
        std::cout << "Selected CNN Type: Optimized GPU With Constant Memory" << std::endl;
    }
    else
    {
        std::cerr << "Invalid CNN Type. Supported types: cpu, gpu, optimize1, optimize2" << std::endl;
        return 1;
    }

    // 1. Load data
    MNIST dataset("../data/fashion-mnist/");
    dataset.read();
    int n_train = dataset.train_data.cols();
    int dim_in = dataset.train_data.rows();
    std::cout << "mnist train number: " << n_train << std::endl;
    std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;

    float accuracy = 0.0;
    std::cout << "==============================" << std::endl;
    accuracy = compute_accuracy(dnn.output(), dataset.test_labels);
    std::cout << "test accuracy: " << accuracy << std::endl;

    return 0;
}