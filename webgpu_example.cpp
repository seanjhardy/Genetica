#define WEBGPU_CPP_IMPLEMENTATION
#include <webgpu/webgpu.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <thread>
#include <chrono>

// Helper function to read shader file
std::string readFile(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  std::string content((std::istreambuf_iterator<char>(file)),
    std::istreambuf_iterator<char>());
  return content;
}

// Initialize WebGPU
wgpu::Instance createInstance() {
  std::cout << "Creating WebGPU instance..." << std::endl;
  return wgpu::createInstance(wgpu::InstanceDescriptor{});
}

wgpu::Adapter requestAdapter(wgpu::Instance& instance) {
  std::cout << "Requesting adapter..." << std::endl;

  wgpu::RequestAdapterOptions options{};
  options.powerPreference = wgpu::PowerPreference::HighPerformance;
  auto adapter = instance.requestAdapter(options);

  if (!adapter) {
    throw std::runtime_error("Failed to get adapter");
  }

  return adapter;
}

wgpu::Device requestDevice(wgpu::Adapter& adapter) {
  std::cout << "Requesting device..." << std::endl;

  wgpu::DeviceDescriptor deviceDesc{};
  deviceDesc.label = "My Device";

  auto device = adapter.requestDevice(deviceDesc);
  if (!device) {
    throw std::runtime_error("Failed to get device");
  }

  // Set up error callback
  device.setUncapturedErrorCallback([](wgpu::ErrorType type, const char* message) {
    std::cerr << "WebGPU Error: " << message << std::endl;
    });

  return device;
}

wgpu::ShaderModule createShaderModule(wgpu::Device& device, const std::string& shaderSource) {
  wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
  wgslDesc.chain.next = nullptr;
  wgslDesc.chain.sType = wgpu::SType::ShaderModuleWGSLDescriptor;
  wgslDesc.code = shaderSource.c_str();

  wgpu::ShaderModuleDescriptor shaderDesc{};
  shaderDesc.nextInChain = &wgslDesc.chain;
  shaderDesc.label = "Compute Shader";

  return device.createShaderModule(shaderDesc);
}

int main() {
  try {
    std::cout << "=== WebGPU 3x3 Kernel Example ===" << std::endl;

    // 1. Initialize WebGPU
    auto instance = createInstance();
    auto adapter = requestAdapter(instance);
    auto device = requestDevice(adapter);

    // 2. Create shader module from file
    std::string shaderSource = readFile("webgpu_example_shader.wgsl");
    auto shaderModule = createShaderModule(device, shaderSource);
    std::cout << "Created shader module from file" << std::endl;

    // 3. Store two numbers in GPU memory (uniform buffer)
    struct Constants {
      float multiplier_x = 2.0f;
      float multiplier_y = 3.0f;
    } constants;

    wgpu::BufferDescriptor uniformBufferDesc{};
    uniformBufferDesc.size = sizeof(Constants);
    uniformBufferDesc.usage = wgpu::BufferUsage::Uniform;
    uniformBufferDesc.mappedAtCreation = true;
    uniformBufferDesc.label = "Constants Buffer";

    auto uniformBuffer = device.createBuffer(uniformBufferDesc);

    // Write constants to buffer
    memcpy(uniformBuffer.getMappedRange(0, sizeof(Constants)), &constants, sizeof(Constants));
    uniformBuffer.unmap();
    std::cout << "Created uniform buffer with constants: x=" << constants.multiplier_x
      << ", y=" << constants.multiplier_y << std::endl;

    // 4. Create storage buffer for results (3x3 = 9 elements)
    const size_t resultCount = 9;
    const size_t resultBufferSize = resultCount * sizeof(float);

    struct Result {
      float value;
    };

    wgpu::BufferDescriptor storageBufferDesc{};
    storageBufferDesc.size = resultBufferSize;
    storageBufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    storageBufferDesc.label = "Results Buffer";

    auto storageBuffer = device.createBuffer(storageBufferDesc);
    std::cout << "Created storage buffer for " << resultCount << " results" << std::endl;

    // 5. Create bind group layout
    wgpu::BindGroupLayoutEntry uniformEntry{};
    uniformEntry.binding = 0;
    uniformEntry.visibility = wgpu::ShaderStage::Compute;
    uniformEntry.buffer.type = wgpu::BufferBindingType::Uniform;
    uniformEntry.buffer.hasDynamicOffset = false;

    wgpu::BindGroupLayoutEntry storageEntry{};
    storageEntry.binding = 1;
    storageEntry.visibility = wgpu::ShaderStage::Compute;
    storageEntry.buffer.type = wgpu::BufferBindingType::Storage;
    storageEntry.buffer.hasDynamicOffset = false;

    std::vector<wgpu::BindGroupLayoutEntry> entries = { uniformEntry, storageEntry };

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc{};
    bindGroupLayoutDesc.entryCount = entries.size();
    bindGroupLayoutDesc.entries = entries.data();
    bindGroupLayoutDesc.label = "Bind Group Layout";

    auto bindGroupLayout = device.createBindGroupLayout(bindGroupLayoutDesc);
    std::cout << "Created bind group layout" << std::endl;

    // 6. Create bind group
    wgpu::BindGroupEntry uniformBindEntry{};
    uniformBindEntry.binding = 0;
    uniformBindEntry.buffer = uniformBuffer;
    uniformBindEntry.offset = 0;
    uniformBindEntry.size = sizeof(Constants);

    wgpu::BindGroupEntry storageBindEntry{};
    storageBindEntry.binding = 1;
    storageBindEntry.buffer = storageBuffer;
    storageBindEntry.offset = 0;
    storageBindEntry.size = resultBufferSize;

    std::vector<wgpu::BindGroupEntry> bindEntries = { uniformBindEntry, storageBindEntry };

    wgpu::BindGroupDescriptor bindGroupDesc{};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = bindEntries.size();
    bindGroupDesc.entries = bindEntries.data();
    bindGroupDesc.label = "Bind Group";

    auto bindGroup = device.createBindGroup(bindGroupDesc);
    std::cout << "Created bind group" << std::endl;

    // 7. Create compute pipeline (let WebGPU infer layout automatically)
    wgpu::ComputePipelineDescriptor pipelineDesc{};
    pipelineDesc.compute.module = shaderModule;
    pipelineDesc.compute.entryPoint = "main";
    pipelineDesc.label = "Compute Pipeline";

    auto computePipeline = device.createComputePipeline(pipelineDesc);
    std::cout << "Created compute pipeline" << std::endl;

    // 8. Create command encoder and compute pass
    auto encoder = device.createCommandEncoder(wgpu::CommandEncoderDescriptor{});
    std::cout << "Created command encoder" << std::endl;

    wgpu::ComputePassDescriptor computePassDesc{};
    computePassDesc.label = "My Compute Pass";

    auto computePass = encoder.beginComputePass(computePassDesc);
    std::cout << "Started compute pass" << std::endl;

    try {
      computePass.setPipeline(computePipeline);
      std::cout << "Set compute pipeline" << std::endl;
      computePass.setBindGroup(0, bindGroup, 0, nullptr);
      std::cout << "Set bind group" << std::endl;
    }
    catch (const std::exception& e) {
      std::cerr << "Error in compute pass setup: " << e.what() << std::endl;
      throw;
    }

    // Dispatch 3x3 workgroups (9 total invocations)
    try {
      computePass.dispatchWorkgroups(9, 1, 1);
      std::cout << "Dispatched workgroups" << std::endl;
    }
    catch (const std::exception& e) {
      std::cerr << "Error in dispatch: " << e.what() << std::endl;
      throw;
    }

    // Try to end the compute pass - this is required for proper cleanup
    std::cout << "Attempting to end compute pass..." << std::endl;

    // Add some debugging information before the call
    std::cout << "About to call computePass.end()..." << std::endl;

    // Simply try to end the compute pass
    try {
      computePass.end();
      std::cout << "Ended compute pass successfully" << std::endl;
    }
    catch (...) {
      std::cerr << "computePass.end() failed but continuing..." << std::endl;
    }

    // 9. Create readback buffer to copy results to CPU
    wgpu::BufferDescriptor readbackBufferDesc{};
    readbackBufferDesc.size = resultBufferSize;
    readbackBufferDesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    readbackBufferDesc.label = "Readback Buffer";

    auto readbackBuffer = device.createBuffer(readbackBufferDesc);
    std::cout << "Readback buffer created" << std::endl;

    // Copy results to readback buffer
    try {
      std::cout << "Adding buffer copy command..." << std::endl;
      encoder.copyBufferToBuffer(storageBuffer, 0, readbackBuffer, 0, resultBufferSize);
      std::cout << "Buffer copy command added successfully" << std::endl;
    }
    catch (const std::exception& e) {
      std::cerr << "Error in buffer copy command: " << e.what() << std::endl;
      throw;
    }

    // 10. Submit commands and wait
    try {
      std::cout << "Finishing command encoder..." << std::endl;
      auto commands = encoder.finish();
      std::cout << "Command encoder finished successfully" << std::endl;

      std::cout << "Submitting commands to queue..." << std::endl;
      device.getQueue().submit(1, &commands);
      std::cout << "Commands submitted to queue" << std::endl;
    }
    catch (const std::exception& e) {
      std::cerr << "Error in encoder finish or queue submit: " << e.what() << std::endl;
      throw;
    }

    std::cout << "Dispatching 3x3 kernel..." << std::endl;

    // 11. Map and read results
    bool mappingComplete = false;
    readbackBuffer.mapAsync(wgpu::MapMode::Read, 0, resultBufferSize,
      [&mappingComplete](wgpu::BufferMapAsyncStatus status) {
        mappingComplete = true;
        std::cout << "Buffer mapping complete with status: " << status << std::endl;
      });

    // Wait for mapping to complete (simple polling approach for this example)
    std::cout << "Waiting for buffer mapping..." << std::endl;
    while (!mappingComplete) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const float* results = static_cast<const float*>(readbackBuffer.getMappedRange(0, resultBufferSize));

    if (results) {
      std::cout << "\nResults from 3x3 kernel:" << std::endl;
      for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
          int index = y * 3 + x;
          float expected = x * constants.multiplier_x + y * constants.multiplier_y;
          std::cout << "Position (" << x << "," << y << "): result=" << results[index]
            << ", expected=" << expected << std::endl;
        }
      }
    }
    else {
      std::cerr << "Failed to map results buffer" << std::endl;
    }

    readbackBuffer.unmap();

    std::cout << "\nWebGPU example completed successfully!" << std::endl;

  }
  catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
