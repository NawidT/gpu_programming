%%writefile image_cv_ops.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>

__global__ void apply_gaussian_blur(const float *image, float *out_image, int height, int width) {
    // conditional in place when num threads exceed needed operations.

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
      return;
    }

    int pixel_index = y * width + x;

    // create the edge kernel
    float kernel[3][3] = {
        {1.0/16, 2.0/16, 1.0/16},
        {2.0/16, 4.0/16, 2.0/16},
        {1.0/16, 2.0/16, 1.0/16}
    };

    // apply blur to image, pasting to out_image
    int half_kernel_size = 1;
    float sum = 0.0f;

    for (int i = -half_kernel_size; i <= half_kernel_size; i++) {
        for (int j = -half_kernel_size; j <= half_kernel_size; j++) {
            int neighbor_x = x + j;
            int neighbor_y = y + i;
            // Check for image edge conditions
            if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {
                int neighbor_index = neighbor_y * width + neighbor_x;
                sum += image[neighbor_index] * kernel[i + half_kernel_size][j + half_kernel_size];
            }
        }
    }
    out_image[pixel_index] = sum;
}

void run_gaussian_blur(const float *image, float *out_image, int height, int width) {
    // copy image to VRAM (shared memory)
    int num_pixels = height * width;
    size_t size_pixels = num_pixels * sizeof(float);
    float* d_image, *d_out_image;
    cudaMalloc(&d_image, size_pixels);
    cudaMalloc(&d_out_image, size_pixels);
    cudaMemcpy(d_image, image, size_pixels, cudaMemcpyHostToDevice);

    // TODO: apply gaussian blur to image

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    apply_gaussian_blur<<<grid, block>>>(d_image, d_out_image, height, width);

    // copy image back to host
    cudaMemcpy(out_image, d_out_image, size_pixels, cudaMemcpyDeviceToHost);

    // clean up
    cudaFree(d_image);
    cudaFree(d_out_image);
}

int main() {
   // open image using cv::imread()
   cv::Mat image = cv::imread("1b_cat.bmp", cv::IMREAD_GRAYSCALE);
   if (image.empty()) {
       printf("Could not open or find the image\n");
       return -1;
   }


   int num_pixels = image.rows * image.cols;
   float *h_image = (float*) malloc(num_pixels * sizeof(float));;
   float *h_out_image = (float*) malloc(num_pixels * sizeof(float));

   // Convert the image data from uchar to float.
   for (int i = 0; i < num_pixels; ++i) {
      h_image[i] = static_cast<float>(image.data[i]);
    }

   printf("number of pixels are %d \n", num_pixels);

   // apply gaussian blur to image
   run_gaussian_blur(h_image, h_out_image, image.rows, image.cols);
   run_gaussian_blur(h_out_image, h_out_image, image.rows, image.cols);
   run_gaussian_blur(h_out_image, h_out_image, image.rows, image.cols);
   run_gaussian_blur(h_out_image, h_out_image, image.rows, image.cols);

   // convert the output image back to cv::Mat
   cv::Mat out_image_mat(image.size(), CV_32F, h_out_image);
   cv::Mat out_image_8U;
   out_image_mat.convertTo(out_image_8U, CV_8U);
   cv::imwrite("blurred_image.jpg", out_image_8U);

   // clean up
   free(h_out_image);
   return 0;
}