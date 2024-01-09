#include "conv_gpu.h"
#include <math.h>
#include <iostream>
#include <typeinfo>
#include <assert.h>

#define TILE_WIDTH 16

void ConvGpu::init()
{
    height_out = (1 + (height_in - height_kernel + 2 * pad_h) / stride);
    width_out = (1 + (width_in - width_kernel + 2 * pad_w) / stride);
    dim_out = height_out * width_out * channel_out;
    weight.resize(channel_in * height_kernel * width_kernel, channel_out);
    bias.resize(channel_out);
    grad_weight.resize(channel_in * height_kernel * width_kernel, channel_out);
    grad_bias.resize(channel_out);
    set_normal_random(weight.data(), weight.size(), 0, 0.01);
    set_normal_random(bias.data(), bias.size(), 0, 0.01);
}

__global__ void convolution_kernel(float *result, const float *input_data, const float *filter,
                                   const int num_samples, const int num_output_channels, const int num_input_channels,
                                   const int input_height, const int input_width, const int filter_size)
{
    const int output_height = input_height - filter_size + 1;
    const int output_width = input_width - filter_size + 1;

    int width_grid = ceil(1.0 * output_width / TILE_WIDTH);

    int batch_index = blockIdx.x;                                         // Batch number
    int output_feature_index = blockIdx.y;                                // Output feature index
    int row_index = (blockIdx.z / width_grid) * TILE_WIDTH + threadIdx.y; // Row index in the image matrix
    int col_index = (blockIdx.z % width_grid) * TILE_WIDTH + threadIdx.x; // Column index in the image matrix

    float result_accumulator = 0.0f;

    if (row_index < output_height && col_index < output_width)
    {
        for (int input_channel_index = 0; input_channel_index < num_input_channels; input_channel_index++) // Sum over all input channels
        {
            for (int filter_row = 0; filter_row < filter_size; filter_row++) // Filter of size filter_size x filter_size
            {
                for (int filter_col = 0; filter_col < filter_size; filter_col++)
                {
                    int input_row = row_index + filter_row;
                    int input_col = col_index + filter_col;
                    result_accumulator += input_data[(batch_index * (num_input_channels * input_height * input_width)) +
                                                     (input_channel_index * (input_height * input_width)) +
                                                     (input_row * input_width) +
                                                     input_col] *
                                          filter[(output_feature_index * (num_input_channels * filter_size * filter_size)) +
                                                 (input_channel_index * (filter_size * filter_size)) +
                                                 (filter_row * filter_size) +
                                                 filter_col];
                }
            }
        }
        result[(batch_index * (num_output_channels * output_height * output_width)) +
               (output_feature_index * (output_height * output_width)) +
               (row_index * output_width) +
               col_index] = result_accumulator;
    }
}

void ConvGpu::perform_convolution_gpu(float *output, const float *input, const float *filter,
                                      const int num_samples, const int num_output_channels, const int num_input_channels,
                                      const int input_height, const int input_width, const int filter_size)
{
    const int output_height = input_height - filter_size + 1;
    const int output_width = input_width - filter_size + 1;

    // Allocate device memory
    float *device_input, *device_output, *device_filter;
    CHECK(cudaMalloc((void **)&device_input, num_samples * num_input_channels * input_height * input_width * sizeof(float)));         // Input feature map with num_input_channels
    CHECK(cudaMalloc((void **)&device_output, num_samples * num_output_channels * output_height * output_width * sizeof(float)));     // Output feature map with num_output_channels
    CHECK(cudaMalloc((void **)&device_filter, num_output_channels * num_input_channels * filter_size * filter_size * sizeof(float))); // Filter with size filter_size * filter_size

    // Copy input and filter data to device
    CHECK(cudaMemcpy(device_input, input, num_samples * num_input_channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_filter, filter, num_output_channels * num_input_channels * filter_size * filter_size * sizeof(float), cudaMemcpyHostToDevice));

    // Set the kernel dimensions and call the kernel
    int Z = ceil(1.0 * output_height / TILE_WIDTH) * ceil(1.0 * output_width / TILE_WIDTH);
    dim3 num_threads_per_block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 num_blocks_in_grid(num_samples, num_output_channels, Z);

    // Launch the kernel
    GpuTimer timer;
    timer.Start();
    convolution_kernel<<<num_blocks_in_grid, num_threads_per_block>>>(device_output, device_input, device_filter, num_samples, num_output_channels, num_input_channels, input_height, input_width, filter_size);
    timer.Stop();
    std::cout << "\t- Layer has kernel time: " << timer.Elapsed() << " ms" << std::endl;

    // Copy the output back to the host
    CHECK(cudaMemcpy(output, device_output, num_samples * num_output_channels * output_height * output_width * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK(cudaFree(device_input));
    CHECK(cudaFree(device_output));
    CHECK(cudaFree(device_filter));
}

void ConvGpu::forward(const Matrix &bottom)
{
    GpuTimer timer;
    timer.Start();

    int n_sample = bottom.cols();
    top.resize(height_out * width_out * channel_out, n_sample);
    float *input_data = (float *)bottom.data();
    float *output_data = (float *)top.data();
    float *weight_data = (float *)weight.data();

    const int num_samples = n_sample;
    const int input_channel = channel_in;
    const int output_channel = channel_out;
    const int kernel_height = height_kernel; // Assuming width_kernel is also K

    if (input_channel == 1)
        std::cout << "Convolution c1 - GPU";
    else
        std::cout << "Convolution c3 - GPU";
    perform_convolution_gpu(output_data, input_data, weight_data,
                            num_samples, output_channel, input_channel,
                            height_in, width_in, kernel_height);

    // Stop layer timer
    timer.Stop();
    std::cout << "\t- Total layer time: " << timer.Elapsed() << " ms" << std::endl;
}

void ConvGpu::im2col(const Vector &image, Matrix &data_col)
{
    int hw_in = height_in * width_in;
    int hw_kernel = height_kernel * width_kernel;
    int hw_out = height_out * width_out;
    // im2col
    data_col.resize(hw_out, hw_kernel * channel_in);
    for (int c = 0; c < channel_in; c++)
    {
        Vector map = image.block(hw_in * c, 0, hw_in, 1); // c-th channel map
        for (int i = 0; i < hw_out; i++)
        {
            int step_h = i / width_out;
            int step_w = i % width_out;
            int start_idx = step_h * width_in * stride + step_w * stride; // left-top idx of window
            for (int j = 0; j < hw_kernel; j++)
            {
                int cur_col = start_idx % width_in + j % width_kernel - pad_w; // col after padding
                int cur_row = start_idx / width_in + j / width_kernel - pad_h;
                if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
                    cur_row >= height_in)
                {
                    data_col(i, c * hw_kernel + j) = 0;
                }
                else
                {
                    int pick_idx = cur_row * width_in + cur_col;
                    data_col(i, c * hw_kernel + j) = map(pick_idx); // pick which pixel
                }
            }
        }
    }
}

void ConvGpu::col2im(const Matrix &data_col, Vector &image)
{
    int hw_in = height_in * width_in;
    int hw_kernel = height_kernel * width_kernel;
    int hw_out = height_out * width_out;
    // col2im
    image.resize(hw_in * channel_in);
    image.setZero();
    for (int c = 0; c < channel_in; c++)
    {
        for (int i = 0; i < hw_out; i++)
        {
            int step_h = i / width_out;
            int step_w = i % width_out;
            int start_idx = step_h * width_in * stride + step_w * stride; // left-top idx of window
            for (int j = 0; j < hw_kernel; j++)
            {
                int cur_col = start_idx % width_in + j % width_kernel - pad_w; // col after padding
                int cur_row = start_idx / width_in + j / width_kernel - pad_h;
                if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
                    cur_row >= height_in)
                {
                    continue;
                }
                else
                {
                    // int pick_idx = start_idx + (j / width_kernel) * width_in + j % width_kernel;
                    int pick_idx = cur_row * width_in + cur_col;
                    image(c * hw_in + pick_idx) += data_col(i, c * hw_kernel + j); // pick which pixel
                }
            }
        }
    }
}

void ConvGpu::backward(const Matrix &bottom, const Matrix &grad_top)
{
    int n_sample = bottom.cols();
    grad_weight.setZero();
    grad_bias.setZero();
    grad_bottom.resize(height_in * width_in * channel_in, n_sample);
    grad_bottom.setZero();
    for (int i = 0; i < n_sample; i++)
    {

        // Forward no longer needs the next 3 rows
        Matrix data_col;
        im2col(bottom.col(i), data_col);
        data_cols[i] = data_col;

        // im2col of grad_top
        Matrix grad_top_i = grad_top.col(i);
        Matrix grad_top_i_col = Eigen::Map<Matrix>(grad_top_i.data(),
                                                   height_out * width_out, channel_out);
        // d(L)/d(w) = \sum{ d(L)/d(z_i) * d(z_i)/d(w) }
        grad_weight += data_cols[i].transpose() * grad_top_i_col;
        // d(L)/d(b) = \sum{ d(L)/d(z_i) * d(z_i)/d(b) }
        grad_bias += grad_top_i_col.colwise().sum().transpose();
        // d(L)/d(x) = \sum{ d(L)/d(z_i) * d(z_i)/d(x) } = d(L)/d(z)_col * w'
        Matrix grad_bottom_i_col = grad_top_i_col * weight.transpose();
        // col2im of grad_bottom
        Vector grad_bottom_i;
        col2im(grad_bottom_i_col, grad_bottom_i);
        grad_bottom.col(i) = grad_bottom_i;
    }
}

void ConvGpu::update(Optimizer &opt)
{
    Vector::AlignedMapType weight_vec(weight.data(), weight.size());
    Vector::AlignedMapType bias_vec(bias.data(), bias.size());
    Vector::ConstAlignedMapType grad_weight_vec(grad_weight.data(), grad_weight.size());
    Vector::ConstAlignedMapType grad_bias_vec(grad_bias.data(), grad_bias.size());

    opt.update(weight_vec, grad_weight_vec);
    opt.update(bias_vec, grad_bias_vec);
}

std::vector<float> ConvGpu::get_parameters() const
{
    std::vector<float> res(weight.size() + bias.size());
    // Copy the data of weights and bias to a long vector
    std::copy(weight.data(), weight.data() + weight.size(), res.begin());
    std::copy(bias.data(), bias.data() + bias.size(), res.begin() + weight.size());
    return res;
}

void ConvGpu::set_parameters(const std::vector<float> &param)
{
    if (static_cast<int>(param.size()) != weight.size() + bias.size())
        throw std::invalid_argument("Parameter size does not match");
    std::copy(param.begin(), param.begin() + weight.size(), weight.data());
    std::copy(param.begin() + weight.size(), param.end(), bias.data());
}

std::vector<float> ConvGpu::get_derivatives() const
{
    std::vector<float> res(grad_weight.size() + grad_bias.size());
    // Copy the data of weights and bias to a long vector
    std::copy(grad_weight.data(), grad_weight.data() + grad_weight.size(), res.begin());
    std::copy(grad_bias.data(), grad_bias.data() + grad_bias.size(),
              res.begin() + grad_weight.size());
    return res;
}