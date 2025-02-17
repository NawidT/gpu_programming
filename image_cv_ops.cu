#include <cuda_runtime.h>
#include <stdio.h>
#include <fstream>

void save_image_to_pgm(const char* filename, const float* image) {
    int width = 512;
    int height = 512;
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file) {
        printf("Could not open file for writing\n");
        return;
    }

    // Write the PGM header
    file << "P5\n" << width << " " << height << "\n255\n";

    // Write the pixel data
    for (int i = 0; i < width * height; ++i) {
        unsigned char pixel = static_cast<unsigned char>(image[i] * 255.0f);
        file.write(reinterpret_cast<char*>(&pixel), sizeof(pixel));
    }

    file.close();
}


__global__ void apply_gaussian_blur(const float *image, float *out_image, int num_pixels) {
    // conditional in place when num threads exceed needed operations.
    if (threadIdx.x >= num_pixels) {
        return;
    }

    int pixel_index = threadIdx.x;

    // create the gaussian kernel
    float kernel[3][3] = {
        {1.0/16, 2.0/16, 1.0/16},
        {2.0/16, 4.0/16, 2.0/16},
        {1.0/16, 2.0/16, 1.0/16}
    };

    // apply blur to image, pasting to out_image
    int kernel_size = 3;
    float sum = 0;
    for (int i = 0; i < kernel_size; i++) {
        for (int j  = 0; j < kernel_size; j++) {
            int image_index = pixel_index + i * kernel_size + j;
            sum += image[image_index] * kernel[i][j];
        }
    }
    out_image[pixel_index] = sum;
}

void run_gaussian_blur(const float *image, float *out_image, int num_pixels) {
    // copy image to VRAM (shared memory)
    size_t size_pixels = num_pixels * sizeof(float);
    float* d_image, *d_out_image;
    cudaMalloc(&d_image, size_pixels);
    cudaMalloc(&d_out_image, size_pixels);
    cudaMemcpy(d_image, image, size_pixels, cudaMemcpyHostToDevice);

    // apply gaussian blur to image
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_pixels + threadsPerBlock - 1) / threadsPerBlock;
    apply_gaussian_blur<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_out_image, num_pixels);

    // copy image back to host
    cudaMemcpy(out_image, d_out_image, size_pixels, cudaMemcpyDeviceToHost);

    // clean up
    cudaFree(d_image);
    cudaFree(d_out_image);
}

int main() {
    // PROBLEM: apply a gaussian blur to an image using CUDA

    // Define image dimensions
    int width = 512;
    int height = 512;
    int num_pixels = width * height;

    // create h_image using "1b_cat.bmp" to grayscale
    float* h_image = (float*) malloc(num_pixels * sizeof(float));
    std::ifstream file("1b_cat.bmp", std::ios::in | std::ios::binary);
    if (!file) {
        printf("Could not open file for reading\n");
        return 1;
    }

    // Allocate memory for the output image
    float* h_out_image = (float*) malloc(num_pixels * sizeof(float));

    // Apply gaussian blur to image
    run_gaussian_blur(h_image, h_out_image, num_pixels);

    // Save the output image to a PGM file
    save_image_to_pgm("blurred_image.pgm", h_out_image, width, height);

    // Clean up
    free(h_image);
    free(h_out_image);

    return 0;
}