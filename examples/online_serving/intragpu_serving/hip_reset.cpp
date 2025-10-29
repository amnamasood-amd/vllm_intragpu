#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    hipError_t err;

    // Your HIP operations would go here, creating streams and other resources...

    // Reset the current device, destroying all associated resources
    for (int i=4; i<8; i++) {
        printf("device %d",i);
        err=hipSetDevice(i);
        if (err != hipSuccess) {
            std::cerr << "Error setting device: " << hipGetErrorString(err) << std::endl;
            return 1;
        }
        err = hipDeviceReset();
        if (err != hipSuccess) {
            std::cerr << "Error resetting device: " << hipGetErrorString(err) << std::endl;
            return 1;
        }
        std::cout << "Device reset successfully. All streams and resources have been destroyed." << std::endl;
    }
    return 0;
}