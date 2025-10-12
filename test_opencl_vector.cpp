#include "include/modules/gpu/structures/staticGPUVector.hpp"
#include "include/modules/gpu/OpenCLManager.hpp"
#include <iostream>
#include <vector>

int main() {
  try {
    // Initialize OpenCL
    OpenCLManager::init();

    // Test basic functionality
    std::cout << "Testing StaticGPUVector with OpenCL..." << std::endl;

    // Test 1: Create vector from host data
    std::vector<float> host_data = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
    StaticGPUVector<float> gpu_vec(host_data);

    std::cout << "Created GPU vector with size: " << gpu_vec.size() << std::endl;

    // Test 2: Copy data back to host
    std::vector<float> result = gpu_vec.toHost();
    std::cout << "Copied data back to host: ";
    for (float val : result) {
      std::cout << val << " ";
    }
    std::cout << std::endl;

    // Test 3: Test individual item access
    float item = gpu_vec.getItem(2);
    std::cout << "Item at index 2: " << item << std::endl;

    // Test 4: Test setting an item
    gpu_vec.setItem(2, 99.0f);
    float new_item = gpu_vec.getItem(2);
    std::cout << "Item at index 2 after setting to 99: " << new_item << std::endl;

    // Test 5: Test copy constructor
    StaticGPUVector<float> gpu_vec_copy = gpu_vec.copy();
    std::vector<float> copy_result = gpu_vec_copy.toHost();
    std::cout << "Copied vector data: ";
    for (float val : copy_result) {
      std::cout << val << " ";
    }
    std::cout << std::endl;

    // Test 6: Test resize
    gpu_vec.resize(10);
    std::cout << "Resized vector size: " << gpu_vec.size() << std::endl;

    std::cout << "All tests passed!" << std::endl;

  }
  catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  // Cleanup
  OpenCLManager::cleanup();
  return 0;
}
