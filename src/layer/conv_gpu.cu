#include "conv_gpu.h"
#include <math.h>
#include <iostream>
#include <chrono>
#include <vector>

#define MAX_WEIGHT_SIZE 1600
__constant__ float const_weights[MAX_WEIGHT_SIZE];

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

__global__ void conv(float *image, float *result, int height_in, int width_in, int height_kernel, int width_kernel,
                     int height_out, int width_out, int channel_in, int channel_out, int stride, int pad_w, int pad_h)
{
    /**
     * Function responsible for performing GPU Convolution on the provided image, using the kernel stored in constant
     * memory, and story the result in the result array. **Note all data is stored Column-Major**
     */

    // Define
    int hw_in = height_in * width_in;
    int hw_kernel = height_kernel * width_kernel;
    int hw_out = height_out * width_out;

    // Define the pixel being operated on from the input image
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Define the index of the channel out being written to
    int c_out = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < hw_out && c_out < channel_out)
    {
        // create temp var to store intermediate values
        float temp = 0;
        for (int c = 0; c < channel_in; c++)
        {
            // Get the image for the current channel
            float *map = &(image[hw_in * c]);
            int step_h = i / width_out;
            int step_w = i % width_out;
            int start_idx = step_h * width_in * stride + step_w * stride; // left-top idx of window
            for (int j = 0; j < hw_kernel; j++)
            {
                int cur_col = start_idx % width_in + j % width_kernel - pad_w; // col after padding
                int cur_row = start_idx / width_in + j / width_kernel - pad_h;
                float pixel_value;
                if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
                    cur_row >= height_in)
                {
                    pixel_value = 0;
                }
                else
                {
                    // Get column-major index
                    int pick_idx = cur_row * width_in + cur_col;
                    pixel_value = map[pick_idx];
                }
                // Get column-major index
                int weight_idx = hw_kernel * channel_in * c_out + (c * hw_kernel + j);
                temp += pixel_value * const_weights[weight_idx];
            }
        }
        // Write out column-major
        result[hw_out * c_out + i] = temp;
    }
}

void ConvGpu::forward(const Matrix &bottom)
{
    GpuTimer timer;
    timer.Start();
    int n_sample = bottom.cols();
    top.resize(height_out * width_out * channel_out, n_sample);
    data_cols.resize(n_sample);
    for (int i = 0; i < n_sample; i++)
    {

        float *d_image;
        float *d_results;

        size_t size_image = sizeof(float) * height_in * width_in * channel_in;
        size_t size_result = sizeof(float) * height_out * width_out * channel_out;
        size_t size_weight = sizeof(float) * channel_in * height_kernel * width_kernel * channel_out;

        CHECK(cudaMalloc((void **)&d_image, size_image));
        CHECK(cudaMalloc((void **)&d_results, size_result));

        // Access only the current sample using column-major indexing
        int image_idx = i * height_in * width_in * channel_in;

        // Copy the image data
        CHECK(cudaMemcpy(d_image, &bottom.data()[image_idx], size_image, cudaMemcpyHostToDevice));
        // Copy the weight data
        CHECK(cudaMemcpyToSymbol(const_weights, weight.data(), size_weight));

        // Define more X threads as there are usually more pixels
        // than there are channels out
        int num_threadsX = 256;
        int num_threadY = 4;

        // Use float to prevent int division
        float hw_out = height_out * width_out;

        int dimGridSizeX = ceil(hw_out / num_threadsX);
        int dimGridSizeY = ceil(channel_out / num_threadY);

        dim3 DimGrid(dimGridSizeX, dimGridSizeY, 1);
        dim3 DimBlock(num_threadsX, num_threadY, 1);

        // Launch Kernel Using Defined dimensions
        conv<<<DimGrid, DimBlock>>>(d_image, d_results, height_in, width_in, height_kernel,
                                    width_kernel, height_out, width_out, channel_in, channel_out, stride, pad_w, pad_h);

        // Ensure Kernel executed sucessfully
        CHECK(cudaGetLastError());

        // Copy the result over to a host array
        float result[height_out * width_out * channel_out];
        cudaMemcpy(result, d_results, size_result, cudaMemcpyDeviceToHost);
        // Create an Eigen::Matrix and convert the 1D array to 2D
        Matrix output = Eigen::Map<Matrix>(result, height_out * width_out, channel_out);
        // Add bias
        output.rowwise() += bias.transpose();

        // Flatten the output
        top.col(i) = Eigen::Map<Vector>(output.data(), output.size());

        // Free Cuda Memory
        CHECK(cudaFree(d_image));
        CHECK(cudaFree(d_results));
        // Check for any Cuda Errrors
        CHECK(cudaGetLastError());
    }

    timer.Stop();
    std::cout << "\t - Layer Time: " << timer.Elapsed() << " ms" << std::endl;
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