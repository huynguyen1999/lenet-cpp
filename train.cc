/*
 * CNN demo for MNIST dataset
 * Author: Kai Han (kaihana@163.com)
 * Details in https://github.com/iamhankai/mini-dnn-cpp
 * Copyright 2018 Kai Han
 */
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "src/dnn_network.h"
#include "src/layer.h"
#include "src/layer/conv.h"
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

int main()
{
	// data
	MNIST dataset("../data/fashion-mnist/");
	dataset.read();
	int n_train = dataset.train_data.cols();
	int dim_in = dataset.train_data.rows();
	std::cout << "mnist train number: " << n_train << std::endl;
	std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
	// dnn
	Network dnn = DnnNetwork(0);
	// train & test
	SGD opt(0.005, 5e-3, 0.9, true);
	// SGD opt(0.001);
	const int n_epoch = 5;
	const int batch_size = 128;
	int best_acc = 0;
	for (int epoch = 0; epoch < n_epoch; epoch++)
	{
		shuffle_data(dataset.train_data, dataset.train_labels);
		for (int start_idx = 0; start_idx < n_train; start_idx += batch_size)
		{
			int ith_batch = start_idx / batch_size;
			Matrix x_batch = dataset.train_data.block(0, start_idx, dim_in,
													  std::min(batch_size, n_train - start_idx));
			Matrix label_batch = dataset.train_labels.block(0, start_idx, 1,
															std::min(batch_size, n_train - start_idx));
			Matrix target_batch = one_hot_encode(label_batch, 10);
			if (false && ith_batch % 10 == 1)
			{
				std::cout << ith_batch << "-th grad: " << std::endl;
				dnn.check_gradient(x_batch, target_batch, 10);
			}
			dnn.forward(x_batch);
			dnn.backward(x_batch, target_batch);
			// display
			if (ith_batch % 20 == 0)
			{
				std::cout << ith_batch << "-th batch, loss: " << dnn.get_loss()
						  << std::endl;
			}
			// optimize
			dnn.update(opt);
		}
		// test
		dnn.forward(dataset.test_data);
		float acc = compute_accuracy(dnn.output(), dataset.test_labels);
		std::cout << std::endl;
		std::cout << epoch + 1 << "-th epoch, test acc: " << acc << std::endl;
		std::cout << std::endl;
		if (acc > best_acc)
		{
			dnn.save_weights("../weights/fashion_weights.bin");
			best_acc = acc;
		}
	}
	return 0;
}
