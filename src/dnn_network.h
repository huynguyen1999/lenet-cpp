#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include "./layer/conv_gpu.h"
#include "./layer/conv.h"
#include "./layer.h"
#include "./layer/fully_connected.h"
#include "./layer/ave_pooling.h"
#include "./layer/max_pooling.h"
#include "./layer/relu.h"
#include "./layer/sigmoid.h"
#include "./layer/softmax.h"
#include "./loss.h"
#include "./loss/mse_loss.h"
#include "./loss/cross_entropy_loss.h"
#include "./mnist.h"
#include "./network.h"
#include "./optimizer.h"
#include "./optimizer/sgd.h"

Network DnnNetwork(int cnn_version)
{
    Network dnn2;
    Layer *conv1, *conv2;
    switch (cnn_version)
    {
    case 0:
        conv1 = new Conv(1, 28, 28, 6, 5, 5);
        conv2 = new Conv(6, 12, 12, 16, 5, 5);
        break;
    case 1:
        conv1 = new ConvGpu(1, 28, 28, 6, 5, 5);
        conv2 = new ConvGpu(6, 12, 12, 16, 5, 5);
        break;
    case 2:
        break;
    case 3:
        break;
    }
    Layer *pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
    Layer *pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);
    Layer *fc1 = new FullyConnected(pool2->output_dim(), 120);
    Layer *fc2 = new FullyConnected(120, 84);
    Layer *fc3 = new FullyConnected(84, 10);
    Layer *relu_conv1 = new ReLU;
    Layer *relu_conv2 = new ReLU;
    Layer *relu_fc1 = new ReLU;
    Layer *relu_fc2 = new ReLU;
    Layer *softmax = new Softmax;
    dnn2.add_layer(conv1);
    dnn2.add_layer(relu_conv1);
    dnn2.add_layer(pool1);
    dnn2.add_layer(conv2);
    dnn2.add_layer(relu_conv2);
    dnn2.add_layer(pool2);
    dnn2.add_layer(fc1);
    dnn2.add_layer(relu_fc1);
    dnn2.add_layer(fc2);
    dnn2.add_layer(relu_fc2);
    dnn2.add_layer(fc3);
    dnn2.add_layer(softmax);

    // loss
    Loss *loss = new CrossEntropy;
    dnn2.add_loss(loss);

    return dnn2;
}