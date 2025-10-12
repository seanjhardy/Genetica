#include <modules/physics/point.hpp>
#include <modules/physics/constraints.hpp>
#include <modules/gpu/updatePoints.hpp>
#include <modules/gpu/OpenCLManager.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>

// Path to the OpenCL kernel file
static const std::string KERNEL_FILE = "assets/kernels/updatePoints.cl";
static const std::string PROGRAM_NAME = "updatePoints";

// Ensure kernels are loaded
static void ensureKernelsLoaded() {
    static bool loaded = false;
    if (!loaded) {
        try {
            OpenCLManager::loadProgram(KERNEL_FILE);
            loaded = true;
        }
        catch (const std::exception& e) {
            std::cerr << "Failed to load updatePoints kernels: " << e.what() << std::endl;
            throw;
        }
    }
}

void updatePoints(GPUVector<Point>& points, CGPUValue<sf::FloatRect>& bounds) {
    if (points.size() == 0) return;

    ensureKernelsLoaded();

    try {
        // Get kernels
        cl_kernel updateKernel = OpenCLManager::getKernel(KERNEL_FILE, "updatePointsKernel");
        cl_kernel collisionKernel = OpenCLManager::getKernel(KERNEL_FILE, "computeCollisionsKernel");

        // Get buffers
        cl_mem pointsBuffer = points.getBuffer();
        cl_mem boundsBuffer = bounds.deviceData();
        int numPoints = static_cast<int>(points.size());

        // Set kernel arguments for update kernel
        cl_int err;
        err = clSetKernelArg(updateKernel, 0, sizeof(cl_mem), &pointsBuffer);
        clCheckError(err, "clSetKernelArg (updateKernel points)");
        err = clSetKernelArg(updateKernel, 1, sizeof(int), &numPoints);
        clCheckError(err, "clSetKernelArg (updateKernel numPoints)");
        err = clSetKernelArg(updateKernel, 2, sizeof(cl_mem), &boundsBuffer);
        clCheckError(err, "clSetKernelArg (updateKernel bounds)");

        // Run update kernel
        size_t globalSize = points.size();
        size_t localSize = 256;
        // Round up to nearest multiple of localSize
        globalSize = ((globalSize + localSize - 1) / localSize) * localSize;
        OpenCLManager::runKernel1D(updateKernel, globalSize, localSize);

        // Set kernel arguments for collision kernel
        err = clSetKernelArg(collisionKernel, 0, sizeof(cl_mem), &pointsBuffer);
        clCheckError(err, "clSetKernelArg (collisionKernel points)");
        err = clSetKernelArg(collisionKernel, 1, sizeof(int), &numPoints);
        clCheckError(err, "clSetKernelArg (collisionKernel numPoints)");

        // Run collision kernel (2D)
        size_t globalSizeX = points.size();
        size_t globalSizeY = points.size();
        size_t localSizeX = 32;
        size_t localSizeY = 32;
        // Round up to nearest multiple
        globalSizeX = ((globalSizeX + localSizeX - 1) / localSizeX) * localSizeX;
        globalSizeY = ((globalSizeY + localSizeY - 1) / localSizeY) * localSizeY;
        OpenCLManager::runKernel2D(collisionKernel, globalSizeX, globalSizeY, localSizeX, localSizeY);

    }
    catch (const std::exception& e) {
        std::cerr << "Error in updatePoints: " << e.what() << std::endl;
        throw;
    }
}

int movePoint(GPUVector<Point>& points, int pointIndex, const sf::Vector2f& newPos) {
    if (pointIndex < 0 || pointIndex >= points.size()) {
        return -1;
    }

    ensureKernelsLoaded();

    try {
        // Get kernel
        cl_kernel moveKernel = OpenCLManager::getKernel(KERNEL_FILE, "movePointKernel");

        // Create buffer for entity ID result
        cl_int err;
        cl_context context = OpenCLManager::getContext();
        cl_command_queue queue = OpenCLManager::getQueue();

        cl_mem entityIDBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int), nullptr, &err);
        clCheckError(err, "clCreateBuffer (entityID)");

        // Get points buffer
        cl_mem pointsBuffer = points.getBuffer();

        // Create float2 for new position
        float newPosArray[2] = { newPos.x, newPos.y };

        // Set kernel arguments
        err = clSetKernelArg(moveKernel, 0, sizeof(cl_mem), &pointsBuffer);
        clCheckError(err, "clSetKernelArg (moveKernel points)");
        err = clSetKernelArg(moveKernel, 1, sizeof(int), &pointIndex);
        clCheckError(err, "clSetKernelArg (moveKernel pointIndex)");
        err = clSetKernelArg(moveKernel, 2, sizeof(float) * 2, newPosArray);
        clCheckError(err, "clSetKernelArg (moveKernel newPos)");
        err = clSetKernelArg(moveKernel, 3, sizeof(cl_mem), &entityIDBuffer);
        clCheckError(err, "clSetKernelArg (moveKernel entityID)");

        // Run kernel (single work item)
        OpenCLManager::runKernel1D(moveKernel, 1, 1);

        // Read back entity ID
        int entityID;
        err = clEnqueueReadBuffer(queue, entityIDBuffer, CL_TRUE, 0, sizeof(int), &entityID, 0, nullptr, nullptr);
        clCheckError(err, "clEnqueueReadBuffer (entityID)");

        // Cleanup
        clReleaseMemObject(entityIDBuffer);

        return entityID;

    }
    catch (const std::exception& e) {
        std::cerr << "Error in movePoint: " << e.what() << std::endl;
        return -1;
    }
}

// Stub for findNearest - can be implemented later with OpenCL if needed
std::pair<int, float> findNearest(const GPUVector<Point>& points, float x, float y, float minDistance) {
    // For now, use CPU implementation
    if (points.size() == 0) {
        return std::make_pair(-1, -1.0f);
    }

    auto hostPoints = points.toHost();
    int closestIdx = -1;
    float closestDist = minDistance;

    for (size_t i = 0; i < hostPoints.size(); i++) {
        float dx = hostPoints[i].pos.s[0] - x;
        float dy = hostPoints[i].pos.s[1] - y;
        float dist = std::sqrt(dx * dx + dy * dy);

        if (dist < closestDist) {
            closestDist = dist;
            closestIdx = static_cast<int>(i);
        }
    }

    return std::make_pair(closestIdx, closestDist);
}
