#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

int main()
{
    int device_count = 0;
    cudaError_t error_id = cudaGetDeviceCount(&device_count);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n",
               (int) (error_id), cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    for (int device = 0; device < device_count; device++) {
        cudaSetDevice(device);
        struct cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        printf("%d%d", deviceProp.major, deviceProp.minor);
        if (device < device_count-1) {
            printf(";");
        }
    }

    return 0;
}
