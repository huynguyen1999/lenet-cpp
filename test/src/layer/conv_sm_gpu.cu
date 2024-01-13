#include "conv_sm_gpu.h"
#include <math.h>
#include <iostream>
#include <typeinfo>
#include <assert.h>

#define TILE_WIDTH_SHARED_C1 16
#define TILE_WIDTH_SHARED_C3 12

void ConvSmGpu::init()
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

__global__ void sm_convolution_kernel(float *result, const float *input, const float *kernel,
                                      const int num_samples, const int num_output_channels, const int num_input_channels,
                                      const int input_height, const int input_width, const int filter_size)
{
    int TILE_WIDTH_SHARED;
    if (num_input_channels == 1)
    {
        TILE_WIDTH_SHARED = TILE_WIDTH_SHARED_C1;
    }
    else
    {
        TILE_WIDTH_SHARED = TILE_WIDTH_SHARED_C3;
    }

    extern __shared__ float shared_input[];

    const int output_height = input_height - filter_size + 1;
    const int output_width = input_width - filter_size + 1;

    int W_grid = ceil(1.0 * output_width / TILE_WIDTH_SHARED);

    int batch_idx = blockIdx.x;          // batch number
    int output_feature_idx = blockIdx.y; // output feature
    int thread_y = threadIdx.y;          // thread ID in the current TILE
    int thread_x = threadIdx.x;

    int row_idx = (blockIdx.z / W_grid) * TILE_WIDTH_SHARED + thread_y; // row of the input image matrix
    int col_idx = (blockIdx.z % W_grid) * TILE_WIDTH_SHARED + thread_x; // col of the input image matrix

    int start_of_tile_row = (blockIdx.z / W_grid) * TILE_WIDTH_SHARED; // row of the input image matrix
    int start_of_tile_col = (blockIdx.z % W_grid) * TILE_WIDTH_SHARED; // col of the input image matrix

    for (int channel_idx = 0; channel_idx < num_input_channels; channel_idx++)
    {
        for (int i = thread_y; i < TILE_WIDTH_SHARED + filter_size - 1; i += TILE_WIDTH_SHARED)
        {
            for (int j = thread_x; j < TILE_WIDTH_SHARED + filter_size - 1; j += TILE_WIDTH_SHARED)
            {
                if (start_of_tile_row + i < input_height && start_of_tile_col + j < input_width)
                {
                    shared_input[channel_idx * (TILE_WIDTH_SHARED + filter_size - 1) * (TILE_WIDTH_SHARED + filter_size - 1) + i * (TILE_WIDTH_SHARED + filter_size - 1) + j] =
                        input[batch_idx * (num_input_channels * input_height * input_width) + channel_idx * (input_height * input_width) + (start_of_tile_row + i) * input_width + start_of_tile_col + j];
                }
            }
        }
    }
    __syncthreads();

    if ((row_idx < output_height) && (col_idx < output_width))
    {
        float accum = 0.0f;
        for (int channel_idx = 0; channel_idx < num_input_channels; channel_idx++) // sum over all input features
        {
            for (int p = 0; p < filter_size; p++) // filter_size x filter_size filter
                for (int q = 0; q < filter_size; q++)
                    accum += shared_input[channel_idx * (TILE_WIDTH_SHARED + filter_size - 1) * (TILE_WIDTH_SHARED + filter_size - 1) + (p + thread_y) * (TILE_WIDTH_SHARED + filter_size - 1) + (q + thread_x)] * kernel[output_feature_idx * (num_input_channels * filter_size * filter_size) + channel_idx * (filter_size * filter_size) + p * filter_size + q];
        }
        result[batch_idx * (num_output_channels * output_height * output_width) + output_feature_idx * (output_height * output_width) + row_idx * output_width + col_idx] = accum;
    }
}

void ConvSmGpu::perform_convolution_gpu(float *output, const float *input, const float *filter,
                                        const int num_samples, const int num_output_channels, const int num_input_channels,
                                        const int input_height, const int input_width, const int filter_size)
{
    int TILE_WIDTH_SHARED;
    if (num_input_channels == 1)
    {
        TILE_WIDTH_SHARED = TILE_WIDTH_SHARED_C1;
    }
    else
    {
        TILE_WIDTH_SHARED = TILE_WIDTH_SHARED_C3;
    }

    const int output_height = input_height - filter_size + 1;
    const int output_width = input_width - filter_size + 1;

    int input_size = num_samples * num_input_channels * input_height * input_width * sizeof(float);
    int output_size = num_samples * num_output_channels * output_height * output_width * sizeof(float);

    float *device_input, *device_output, *device_filter;

    CHECK(cudaMalloc((void **)&device_input, input_size));
    CHECK(cudaMalloc((void **)&device_output, output_size));
    CHECK(cudaMalloc((void **)&device_filter, num_output_channels * num_input_channels * filter_size * filter_size * sizeof(float)));

    CHECK(cudaMemcpy(device_input, input, input_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_filter, filter, num_output_channels * num_input_channels * filter_size * filter_size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 num_threads_per_block, num_blocks_in_grid;

    num_threads_per_block = dim3(TILE_WIDTH_SHARED, TILE_WIDTH_SHARED, 1);
    int shmem_size = num_input_channels * (TILE_WIDTH_SHARED + filter_size - 1) * (TILE_WIDTH_SHARED + filter_size - 1) * sizeof(float);
    num_blocks_in_grid = dim3(num_samples, num_output_channels, ceil(1.0 * output_height / TILE_WIDTH_SHARED) * ceil(1.0 * output_width / TILE_WIDTH_SHARED));

    // Launch kernel
    GpuTimer timer;
    timer.Start();
    sm_convolution_kernel<<<num_blocks_in_grid, num_threads_per_block, shmem_size>>>(device_output, device_input, device_filter, num_samples, num_output_channels, num_input_channels, input_height, input_width, filter_size);
    timer.Stop();
    std::cout << "\t- Kernel time: " << timer.Elapsed() << " ms" << std::endl;

    CHECK(cudaMemcpy(output, device_output, output_size, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(device_input));
    CHECK(cudaFree(device_output));
    CHECK(cudaFree(device_filter));
}

void ConvSmGpu::forward(const Matrix &bottom)
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
        std::cout << "Convolution c1 - GPU" << std::endl;
    else
        std::cout << "Convolution c3 - GPU" << std::endl;
    perform_convolution_gpu(output_data, input_data, weight_data,
                            num_samples, output_channel, input_channel,
                            height_in, width_in, kernel_height);

    // Stop layer timer
    timer.Stop();
    std::cout << "\t- Total layer time: " << timer.Elapsed() << " ms" << std::endl;
}

void ConvSmGpu::im2col(const Vector &image, Matrix &data_col)
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

void ConvSmGpu::col2im(const Matrix &data_col, Vector &image)
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

void ConvSmGpu::backward(const Matrix &bottom, const Matrix &grad_top)
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

void ConvSmGpu::update(Optimizer &opt)
{
    Vector::AlignedMapType weight_vec(weight.data(), weight.size());
    Vector::AlignedMapType bias_vec(bias.data(), bias.size());
    Vector::ConstAlignedMapType grad_weight_vec(grad_weight.data(), grad_weight.size());
    Vector::ConstAlignedMapType grad_bias_vec(grad_bias.data(), grad_bias.size());

    opt.update(weight_vec, grad_weight_vec);
    opt.update(bias_vec, grad_bias_vec);
}

std::vector<float> ConvSmGpu::get_parameters() const
{
    std::vector<float> res(weight.size() + bias.size());
    // Copy the data of weights and bias to a long vector
    std::copy(weight.data(), weight.data() + weight.size(), res.begin());
    std::copy(bias.data(), bias.data() + bias.size(), res.begin() + weight.size());
    return res;
}

void ConvSmGpu::set_parameters(const std::vector<float> &param)
{
    if (static_cast<int>(param.size()) != weight.size() + bias.size())
        throw std::invalid_argument("Parameter size does not match");
    std::copy(param.begin(), param.begin() + weight.size(), weight.data());
    std::copy(param.begin() + weight.size(), param.end(), bias.data());
}

std::vector<float> ConvSmGpu::get_derivatives() const
{
    std::vector<float> res(grad_weight.size() + grad_bias.size());
    // Copy the data of weights and bias to a long vector
    std::copy(grad_weight.data(), grad_weight.data() + grad_weight.size(), res.begin());
    std::copy(grad_bias.data(), grad_bias.data() + grad_bias.size(),
              res.begin() + grad_weight.size());
    return res;
}