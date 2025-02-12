{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6hfDUrD3h2Ru",
        "outputId": "bd8ca0b9-4ec3-44b7-dd47-947a699b57ad"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Writing code.h\n"
          ]
        }
      ],
      "source": [
        "%%writefile code.h\n",
        "\n",
        "int multiply(int a, int b);"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "m05D3eqLh2Rv",
        "outputId": "900bb627-15e0-40a9-9d05-86539a3576c3"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Overwriting code.cpp\n"
          ]
        }
      ],
      "source": [
        "%%writefile code.cpp\n",
        "\n",
        "#include \"code.h\"\n",
        "using namespace std;\n",
        "\n",
        "int multiply(int a, int b) {\n",
        "  return a * b;\n",
        "};"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "AB36py4uh2Rv",
        "outputId": "5fc15e99-6105-4aa5-f718-622952b2e555"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Overwriting main.cpp\n"
          ]
        }
      ],
      "source": [
        "%%writefile main.cpp\n",
        "\n",
        "#include <iostream>\n",
        "using namespace std;\n",
        "#include \"code.h\"\n",
        "\n",
        "int main() {\n",
        "  cout << multiply(6,7) << '\\n';\n",
        "  return 0;\n",
        "};"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NCn3frt-h2Rv",
        "outputId": "830db017-3433-4319-e800-f14d49bd655f"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "42\n"
          ]
        }
      ],
      "source": [
        "!g++ main.cpp code.cpp -o main\n",
        "!./main"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OL7Nagcbh2Rw",
        "outputId": "d0632a3f-7d33-4966-fa27-a2a92e3fb2dc"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Wed Feb 12 19:05:12 2025       \n",
            "+-----------------------------------------------------------------------------------------+\n",
            "| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |\n",
            "|-----------------------------------------+------------------------+----------------------+\n",
            "| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n",
            "| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n",
            "|                                         |                        |               MIG M. |\n",
            "|=========================================+========================+======================|\n",
            "|   0  NVIDIA A100-SXM4-40GB          Off |   00000000:00:04.0 Off |                    0 |\n",
            "| N/A   32C    P0             45W /  400W |       0MiB /  40960MiB |      0%      Default |\n",
            "|                                         |                        |             Disabled |\n",
            "+-----------------------------------------+------------------------+----------------------+\n",
            "                                                                                         \n",
            "+-----------------------------------------------------------------------------------------+\n",
            "| Processes:                                                                              |\n",
            "|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |\n",
            "|        ID   ID                                                               Usage      |\n",
            "|=========================================================================================|\n",
            "|  No running processes found                                                             |\n",
            "+-----------------------------------------------------------------------------------------+\n"
          ]
        }
      ],
      "source": [
        "!nvidia-smi"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 17,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4Uyr0jBKh2Rw",
        "outputId": "c348b354-861f-4b18-e700-191874e4795e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Overwriting test.cu\n"
          ]
        }
      ],
      "source": [
        "%%writefile test.cu\n",
        "\n",
        "#include <cstdio>\n",
        "\n",
        "__global__ void helloCUDA() {\n",
        "    printf(\"Hello from CUDA kernel cuda!\\n\");\n",
        "}\n",
        "\n",
        "int main() {\n",
        "    printf(\"Hello from CUDA kernel main!\\n\");\n",
        "    helloCUDA<<<1,1>>>();\n",
        "    cudaDeviceSynchronize();\n",
        "    return 0;\n",
        "}"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3wND2hySh2Rw",
        "outputId": "3ab753bd-f6fa-43fc-cc10-ffed23d152fa"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Hello from CUDA kernel main!\n"
          ]
        }
      ],
      "source": [
        "!nvcc test.cu -o test_cuda\n",
        "!./test_cuda"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%%writefile idxing.cu\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "H70igzoRll7j",
        "outputId": "fcc0a776-d7fa-49da-c773-38ee1f076121"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/bin/bash: line 1: ./deviceQuery: No such file or directory\n"
          ]
        }
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.11.4"
    },
    "colab": {
      "provenance": [],
      "gpuType": "A100"
    },
    "accelerator": "GPU"
  },
  "nbformat": 4,
  "nbformat_minor": 0
}