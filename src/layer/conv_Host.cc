#include "conv_Host.h"
#include <math.h>
#include <iostream>
#include <typeinfo>
#include <assert.h>

static void HandleError(cudaError_t err, const char *file, int line);
__global__ void conv(float *image, float *weights, float *result, int height_in, int width_in, int height_kernel, int width_kernel,
                     int height_out, int width_out, int channel_in, int channel_out, int stride, int pad_w, int pad_h);
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

inline void error_check(cudaError_t err, const char *file, int line)
{
  if (err != cudaSuccess)
  {
    ::fprintf(stderr, "CUDA ERROR at %s[%d] : %s\n", file, line, cudaGetErrorString(err));
    abort();
  }
}
#define CUDA_CHECK(err)                   \
  do                                      \
  {                                       \
    error_check(err, __FILE__, __LINE__); \
  } while (0)

void Conv::init() {
  height_out = (1 + (height_in - height_kernel + 2 * pad_h) / stride);
  width_out =   (1 + (width_in - width_kernel + 2 * pad_w) / stride);
  dim_out = height_out * width_out * channel_out;

  weight.resize(channel_in * height_kernel * width_kernel, channel_out);
  bias.resize(channel_out);
  grad_weight.resize(channel_in * height_kernel * width_kernel, channel_out);
  grad_bias.resize(channel_out);
  set_normal_random(weight.data(), weight.size(), 0, 0.01);
  set_normal_random(bias.data(), bias.size(), 0, 0.01);
  //std::cout << weight.colwise().sum() << std::endl;
  //std::cout << weight.colwise().sum() + bias.transpose() << std::endl;
}

// im2col, used for bottom
// image size: Vector (height_in * width_in * channel_in)
// data_col size: Matrix (hw_out, hw_kernel * channel_in)
void Conv::im2col(const Vector& image, Matrix& data_col) {
  int hw_in = height_in * width_in;
  int hw_kernel = height_kernel * width_kernel;
  int hw_out = height_out * width_out;
  // im2col
  data_col.resize(hw_out, hw_kernel * channel_in);
  for (int c = 0; c < channel_in; c ++) {
    Vector map = image.block(hw_in * c, 0, hw_in, 1);  // c-th channel map
    for (int i = 0; i < hw_out; i ++) {
      int step_h = i / width_out;
      int step_w = i % width_out;
      int start_idx = step_h * width_in * stride + step_w * stride;  // left-top idx of window
      for (int j = 0; j < hw_kernel; j ++) {
        int cur_col = start_idx % width_in + j % width_kernel - pad_w;  // col after padding
        int cur_row = start_idx / width_in + j / width_kernel - pad_h;
        if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
            cur_row >= height_in) {
          data_col(i, c * hw_kernel + j) = 0;
        }
        else {
          //int pick_idx = start_idx + (j / width_kernel) * width_in + j % width_kernel;
          int pick_idx = cur_row * width_in + cur_col;
          data_col(i, c * hw_kernel + j) = map(pick_idx);  // pick which pixel
        }
      }
    }
  }
}

void Conv::forward(const Matrix& bottom) {
  int n_sample = bottom.cols();
  top.resize(height_out * width_out * channel_out, n_sample);
  data_cols.resize(n_sample);
  for (int i = 0; i < n_sample; i ++) {
    // im2col
    Matrix data_col;
    im2col(bottom.col(i), data_col);
    data_cols[i] = data_col;
    // conv by product
    Matrix result = data_col * weight;  // result: (hw_out, channel_out)
    result.rowwise() += bias.transpose();
    top.col(i) = Eigen::Map<Vector>(result.data(), result.size());
  }
}

void Conv::forward_Device(const Matrix &bottom)
{
  int n_sample = bottom.cols();
  top.resize(height_out * width_out * channel_out, n_sample);
  data_cols.resize(n_sample);
  for (int i = 0; i < n_sample; i++)
  {
    // im2col
    Matrix data_col;
    im2col(bottom.col(i), data_col);
    data_cols[i] = data_col;
    // conv by product
    Matrix result1 = data_col * weight; // result: (hw_out, channel_out)
    result1.rowwise() += bias.transpose();

    float *d_image;
    float *d_weights;
    float *d_results;

    size_t size_image = sizeof(float) * height_in * width_in * channel_in;
    size_t size_weights = sizeof(float) * channel_in * height_kernel * width_kernel * channel_out;
    size_t size_result = sizeof(float) * height_out * width_out * channel_out;
    // size_t size_weight_data = sizeof(float) * channel_in * height_kernel * width_kernel * channel_out;

    // cudaMemcpyToSymbol(const_weights,weight.data(),sizeof(float) * channel_in * height_kernel * width_kernel * channel_out);

    HANDLE_ERROR(cudaMalloc((void **)&d_image, size_image));
    HANDLE_ERROR(cudaMalloc((void **)&d_weights, size_weights));
    HANDLE_ERROR(cudaMalloc((void **)&d_results, size_result));

    HANDLE_ERROR(cudaMemcpy(d_image, &bottom.data()[i * height_in * width_in * channel_in], size_image, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_weights, weight.data(), size_weights, cudaMemcpyHostToDevice));

    int num_threads = 1024;

    float hw_out = height_out * width_out;

    int dimGridSizeX = ceil(hw_out / num_threads);

    dim3 DimGrid(dimGridSizeX);
    dim3 DimBlock(num_threads);

    // Kernel call to initialize d_B
    conv<<<DimGrid, DimBlock>>>(d_image, d_weights, d_results, height_in, width_in, height_kernel,
                                width_kernel, height_out, width_out, channel_in, channel_out, stride, pad_w, pad_h);
    // conv(image, weight, results2, height_in, width_in, height_kernel,
    //  width_kernel, height_out, width_out, channel_in, channel_out, stride, pad_w, pad_h);
    CUDA_CHECK(cudaGetLastError());

    // float result[height_out * width_out * channel_out];
    float result[height_out * width_out * channel_out];

    cudaMemcpy(result, d_results, size_result, cudaMemcpyDeviceToHost);

    // std::cout << "Got this far" << std::endl;

    // float *image2 = new float[height_in * width_in * channel_in];
    // cudaMemcpy(image2, d_image, size_image, cudaMemcpyDeviceToHost);
    // for(int i = 0; i < height_in * width_in * channel_in; i++){
    //   std::cout << "idx: " << i << " | " << image2[i] << " - " << image(i) << std::endl;
    // }
    // Matrix *weights2 = new Matrix(channel_in * height_kernel * width_kernel, channel_out);
    // cudaMemcpy(weights2, d_weights, size_weights, cudaMemcpyDeviceToHost);
    // for(int i = 0; i < channel_in * height_kernel * width_kernel * channel_out; i++){
    //   std::cout << "idx: " << i << " | " << (*weights2)(i) << " - " << weight(i) << std::endl;
    // }

    // Matrix result = data_col * weight;  // result: (hw_out, channel_out)
    Matrix output = Eigen::Map<Matrix>(result, height_out * width_out, channel_out);

    // std::cout << "size og " << test.size() << std::endl;
    // std::cout << "size new " << output.size() << std::endl;
    // std::cout << "is equal? " << test.isApprox(output) << std::endl;

    // for(int i = 0; i < width_out*height_out*channel_out; i++){
    //   std::cout << "idx: " << i << " | " << result1(i) << " - " << output(i) << std::endl;
    // }

    // output.rowwise() += bias.transpose();
    // top.col(i) = output;
    // free(result);
    output.rowwise() += bias.transpose();

    top.col(i) = Eigen::Map<Vector>(output.data(), output.size());
    // std::cout << "got hur" << std::endl;
    HANDLE_ERROR(cudaFree(d_image));
    HANDLE_ERROR(cudaFree(d_weights));
    HANDLE_ERROR(cudaFree(d_results));
    CUDA_CHECK(cudaGetLastError());
    // std::cout << "got past" << std::endl;
  }
}

__global__ void conv(float *image, float *weights, float *result, int height_in, int width_in, int height_kernel, int width_kernel,
                     int height_out, int width_out, int channel_in, int channel_out, int stride, int pad_w, int pad_h)
{
  /**
   * Function responsible for performing GPU matrix multiplication on d_A and d_B
   * and storing the result in d_C.
   */

  int hw_in = height_in * width_in;
  int hw_kernel = height_kernel * width_kernel;
  int hw_out = height_out * width_out;
  int size_kernel = hw_kernel * channel_in * channel_out;
  // im2col

  // Matrix shared_weights(channel_in * height_kernel * width_kernel, channel_out);
  // int idx;
  // for(int w_idx = 0; w_idx < ceil((float)((size_kernel) / 1024.00)); w_idx++){
  //     idx = w_idx * 1024 + i;

  //     if(idx < size_kernel){
  //       shared_weights(idx) = weights(idx);
  //     }
  // }
  // __syncthreads();

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (int c_out = 0; c_out < channel_out; c_out++)
  {
    float temp = 0;
    for (int c = 0; c < channel_in; c++)
    {
      // Vector map = image.block(hw_in * c, 0, hw_in, 1);  // c-th channel map // Block of size (hw_in,1) starting at (hw_in*c, 0)
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
          // int pick_idx = start_idx + (j / width_kernel) * width_in + j % width_kernel;
          int pick_idx = cur_row * width_in + cur_col;
          // pixel_value = map(pick_idx);
          pixel_value = map[pick_idx];
        }
        // hw_out, hw_kernel * channel_in
        // weight.resize(channel_in * height_kernel * width_kernel, channel_out);
        // weight.resize(channel_in * height_kernel * width_kernel, channel_out);
        temp += pixel_value * weights[hw_kernel * channel_in * c_out + (c * hw_kernel + j)];
      }
    }
    // width * row + col;
    // height * col + row
    // hw_out * c_out + i
    // result[(i, c_out) ]= temp;
    result[hw_out * c_out + i] = temp;
  }
}

// col2im, used for grad_bottom
// data_col size: Matrix (hw_out, hw_kernel * channel_in)
// image size: Vector (height_in * width_in * channel_in)
void Conv::col2im(const Matrix& data_col, Vector& image) {
  int hw_in = height_in * width_in;
  int hw_kernel = height_kernel * width_kernel;
  int hw_out = height_out * width_out;
  // col2im
  image.resize(hw_in * channel_in);
  image.setZero();
  for (int c = 0; c < channel_in; c ++) {
    for (int i = 0; i < hw_out; i ++) {
      int step_h = i / width_out;
      int step_w = i % width_out;
      int start_idx = step_h * width_in * stride + step_w * stride;  // left-top idx of window
      for (int j = 0; j < hw_kernel; j ++) {
        int cur_col = start_idx % width_in + j % width_kernel - pad_w;  // col after padding
        int cur_row = start_idx / width_in + j / width_kernel - pad_h;
        if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
            cur_row >= height_in) {
          continue;
        }
        else {
          //int pick_idx = start_idx + (j / width_kernel) * width_in + j % width_kernel;
          int pick_idx = cur_row * width_in + cur_col;
          image(c * hw_in + pick_idx) += data_col(i, c * hw_kernel + j);  // pick which pixel
        }
      }
    }
  }
}

void Conv::backward(const Matrix& bottom, const Matrix& grad_top) {
  int n_sample = bottom.cols();
  grad_weight.setZero();
  grad_bias.setZero();
  grad_bottom.resize(height_in * width_in * channel_in, n_sample);
  grad_bottom.setZero();
  for (int i = 0; i < n_sample; i ++) {
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

void Conv::update(Optimizer& opt) {
  Vector::AlignedMapType weight_vec(weight.data(), weight.size());
  Vector::AlignedMapType bias_vec(bias.data(), bias.size());
  Vector::ConstAlignedMapType grad_weight_vec(grad_weight.data(), grad_weight.size());
  Vector::ConstAlignedMapType grad_bias_vec(grad_bias.data(), grad_bias.size());

  opt.update(weight_vec, grad_weight_vec);
  opt.update(bias_vec, grad_bias_vec);
}

std::vector<float> Conv::get_parameters() const {
  std::vector<float> res(weight.size() + bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(weight.data(), weight.data() + weight.size(), res.begin());
  std::copy(bias.data(), bias.data() + bias.size(), res.begin() + weight.size());
  return res;
}

void Conv::set_parameters(const std::vector<float>& param) {
  if(static_cast<int>(param.size()) != weight.size() + bias.size())
      throw std::invalid_argument("Parameter size does not match");
  std::copy(param.begin(), param.begin() + weight.size(), weight.data());
  std::copy(param.begin() + weight.size(), param.end(), bias.data());
}

std::vector<float> Conv::get_derivatives() const {
  std::vector<float> res(grad_weight.size() + grad_bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(grad_weight.data(), grad_weight.data() + grad_weight.size(), res.begin());
  std::copy(grad_bias.data(), grad_bias.data() + grad_bias.size(),
            res.begin() + grad_weight.size());
  return res;
}
