#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include <modules/physics/fluid.hpp>
#include <iostream>
#include <modules/utils/floatOps.hpp>
#include <modules/utils/print.hpp>

using uint8_t = unsigned char;

FluidSimulator::FluidSimulator(float scale, size_t width, size_t height, FluidSimulator::Config config)  {
    this->width = width * scale;
    this->height = height * scale;
    this->scale = scale;
    this->config = config;
    init();
}

// inits all buffers, must be called before computeField function call
void FluidSimulator::init()
{
    reset();
    pixelBuffer = std::vector<uint8_t>( width * height * 4);
    texture.create(width, height);

    cudaSetDevice(0);
    cudaMalloc(&colorField, width * height * 4 * sizeof(uint8_t));
    cudaMalloc(&oldField, width * height * sizeof(Particle));
    cudaMalloc(&newField, width * height * sizeof(Particle));
    cudaMalloc(&pressureOld, width * height * sizeof(float));
    cudaMalloc(&pressureNew, width * height * sizeof(float));
    cudaMalloc(&vorticityField, width * height * sizeof(float));
}

// releases all buffers, must be called on program exit
void FluidSimulator::reset()
{
    cudaFree(colorField);
    cudaFree(oldField);
    cudaFree(newField);
    cudaFree(pressureOld);
    cudaFree(pressureNew);
    cudaFree(vorticityField);
}

// interpolates quantity of grid cells
__device__ Particle interpolate(float2 v, Particle* field, size_t width, size_t height)
{
    float x1 = (int)v.x;
    float y1 = (int)v.y;
    float x2 = (int)v.x + 1;
    float y2 = (int)v.y + 1;
    Particle q1, q2, q3, q4;
#define CLAMP(val, minv, maxv) min(maxv, max(minv, val))
#define SET(Q, x, y) Q = field[int(CLAMP(y, 0.0f, height - 1.0f)) * width + int(CLAMP(x, 0.0f, width - 1.0f))]
    SET(q1, x1, y1);
    SET(q2, x1, y2);
    SET(q3, x2, y1);
    SET(q4, x2, y2);
#undef SET
#undef CLAMP
    float t1 = (x2 - v.x) / (x2 - x1);
    float t2 = (v.x - x1) / (x2 - x1);
    float2 f1 = q1.u * t1 + q3.u * t2;
    float2 f2 = q2.u * t1 + q4.u * t2;
    Color3f C1 = q2.color * t1 + q4.color * t2;
    Color3f C2 = q2.color * t1 + q4.color * t2;
    float t3 = (y2 - v.y) / (y2 - y1);
    float t4 = (v.y - y1) / (y2 - y1);
    Particle res;
    res.u = f1 * t3 + f2 * t4;
    res.color = C1 * t3 + C2 * t4;
    return res;
}

// performs iteration of jacobi method on velocity grid field
__device__ float2 jacobiVelocity(Particle* field, size_t width, size_t height, float2 v, float2 B, float alpha, float beta)
{
    float2 vU = B * -1.0f, vD = B * -1.0f, vR = B * -1.0f, vL = B * -1.0f;
#define SET(U, x, y) if (x < width && x >= 0 && y < height && y >= 0) U = field[int(y) * width + int(x)].u
    SET(vU, v.x, v.y - 1);
    SET(vD, v.x, v.y + 1);
    SET(vL, v.x - 1, v.y);
    SET(vR, v.x + 1, v.y);
#undef SET
    v = (vU + vD + vL + vR + B * alpha) * (1.0f / beta);
    return v;
}

// performs iteration of jacobi method on pressure grid field
__device__ float jacobiPressure(float* pressureField, size_t width, size_t height, int x, int y, float B, float alpha, float beta)
{
    float C = pressureField[int(y) * width + int(x)];
    float xU = C, xD = C, xL = C, xR = C;
#define SET(P, x, y) if (x < width && x >= 0 && y < height && y >= 0) P = pressureField[int(y) * width + int(x)]
    SET(xU, x, y - 1);
    SET(xD, x, y + 1);
    SET(xL, x - 1, y);
    SET(xR, x + 1, y);
#undef SET
    float pressure = (xU + xD + xL + xR + alpha * B) * (1.0f / beta);
    return pressure;
}

// performs iteration of jacobi method on color grid field
__device__ Color3f jacobiColor(Particle* colorField, size_t width, size_t height, float2 pos, Color3f B, float alpha, float beta)
{
    Color3f xU, xD, xL, xR, res;
    int x = pos.x;
    int y = pos.y;
#define SET(P, x, y) if (x < width && x >= 0 && y < height && y >= 0) P = colorField[int(y) * width + int(x)]
    SET(xU, x, y - 1).color;
    SET(xD, x, y + 1).color;
    SET(xL, x - 1, y).color;
    SET(xR, x + 1, y).color;
#undef SET
    res = (xU + xD + xL + xR + B * alpha) * (1.0f / beta);
    return res;
}

// computes divergency of velocity field
__device__ float divergency(Particle* field, size_t width, size_t height, int x, int y)
{
    Particle& C = field[int(y) * width + int(x)];
    float x1 = -1 * C.u.x, x2 = -1 * C.u.x, y1 = -1 * C.u.y, y2 = -1 * C.u.y;
#define SET(P, x, y) if (x < width && x >= 0 && y < height && y >= 0) P = field[int(y) * width + int(x)]
    SET(x1, x + 1, y).u.x;
    SET(x2, x - 1, y).u.x;
    SET(y1, x, y + 1).u.y;
    SET(y2, x, y - 1).u.y;
#undef SET
    return (x1 - x2 + y1 - y2) * 0.5f;
}

// computes gradient of pressure field
__device__ float2 gradient(float* field, size_t width, size_t height, int x, int y)
{
    float C = field[y * width + x];
#define SET(P, x, y) if (x < width && x >= 0 && y < height && y >= 0) P = field[int(y) * width + int(x)]
    float x1 = C, x2 = C, y1 = C, y2 = C;
    SET(x1, x + 1, y);
    SET(x2, x - 1, y);
    SET(y1, x, y + 1);
    SET(y2, x, y - 1);
#undef SET
    float2 res = { (x1 - x2) * 0.5f, (y1 - y2) * 0.5f };
    return res;
}

// computes absolute value gradient of vorticity field
__device__ float2 absGradient(float* field, size_t width, size_t height, int x, int y)
{
    float C = field[int(y) * width + int(x)];
#define SET(P, x, y) if (x < width && x >= 0 && y < height && y >= 0) P = field[int(y) * width + int(x)]
    float x1 = C, x2 = C, y1 = C, y2 = C;
    SET(x1, x + 1, y);
    SET(x2, x - 1, y);
    SET(y1, x, y + 1);
    SET(y2, x, y - 1);
#undef SET
    float2 res = { (abs(x1) - abs(x2)) * 0.5f, (abs(y1) - abs(y2)) * 0.5f };
    return res;
}

// computes curl of velocity field
__device__ float curl(Particle* field, size_t width, size_t height, int x, int y)
{
    float2 C = field[int(y) * width + int(x)].u;
#define SET(P, x, y) if (x < width && x >= 0 && y < height && y >= 0) P = field[int(y) * width + int(x)]
    float x1 = -C.x, x2 = -C.x, y1 = -C.y, y2 = -C.y;
    SET(x1, x + 1, y).u.x;
    SET(x2, x - 1, y).u.x;
    SET(y1, x, y + 1).u.y;
    SET(y2, x, y - 1).u.y;
#undef SET
    float res = ((y1 - y2) - (x1 - x2)) * 0.5f;
    return res;
}

// adds quantity to particles using bilinear interpolation
__global__ void advect(Particle* newField, Particle* oldField, size_t width, size_t height, float dDiffusion, float dt)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float decay = 1.0f / (1.0f + dDiffusion * dt);
        float2 pos = {x * 1.0f, y * 1.0f};
        Particle &Pold = oldField[y * width + x];
        // find new particle tracing where it came from
        Particle p = interpolate(pos - Pold.u * dt, oldField, width, height);
        p.u = p.u * decay;
        p.color.x = min(1.0f, pow(p.color.x, 1.005f) * decay);
        p.color.y = min(1.0f, pow(p.color.y, 1.005f) * decay);
        p.color.z = min(1.0f, pow(p.color.z, 1.005f) * decay);
        newField[y * width + x] = p;
    }
}

// calculates color field diffusion
__global__ void computeColor(Particle* newField, Particle* oldField, size_t width, size_t height, float cDiffusion, float dt)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float2 pos = {x * 1.0f, y * 1.0f};
        Color3f c = oldField[y * width + x].color;
        float alpha = cDiffusion * cDiffusion / dt;
        float beta = 4.0f + alpha;
        // perfom one iteration of jacobi method (diffuse method should be called 20-50 times per cell)
        newField[y * width + x].color = jacobiColor(oldField, width, height, pos, c, alpha, beta);
    }
}

// fills output image with corresponding color
__global__ void paint(uint8_t* colorField, Particle* field, size_t width, size_t height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float R = field[y * width + x].color.x;
        float G = field[y * width + x].color.y;
        float B = field[y * width + x].color.z;

        colorField[4 * (y * width + x) + 0] = min(255.0f, 255.0f * R);
        colorField[4 * (y * width + x) + 1] = min(255.0f, 255.0f * G);
        colorField[4 * (y * width + x) + 2] = min(255.0f, 255.0f * B);
        colorField[4 * (y * width + x) + 3] = max(R, max(G, B)) * 255.0f;
    }
}

// calculates nonzero divergency velocity field u
__global__ void diffuse(Particle* newField, Particle* oldField, size_t width, size_t height, float vDiffusion, float dt)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float2 pos = {x * 1.0f, y * 1.0f};
        float2 u = oldField[y * width + x].u;
        // perfoms one iteration of jacobi method (diffuse method should be called 20-50 times per cell)
        float alpha = vDiffusion * vDiffusion / dt;
        float beta = 4.0f + alpha;
        newField[y * width + x].u = jacobiVelocity(oldField, width, height, pos, u, alpha, beta);
    }
}

// performs iteration of jacobi method on pressure field
__global__ void computePressureImpl(Particle* field, size_t width, size_t height, float* pNew, float* pOld, float pressure, float dt)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float div = divergency(field, width, height, x, y);

        float alpha = -1.0f * pressure * pressure;
        float beta = 4.0;
        pNew[y * width + x] = jacobiPressure(pOld, width, height, x, y, div, alpha, beta);
    }
}

// projects pressure field on velocity field
__global__ void project(Particle* newField, size_t width, size_t height, float* pField)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float2 &u = newField[y * width + x].u;
        u = u - gradient(pField, width, height, x, y);
    }
}

// computes vorticity field which should be passed to applyVorticity function
__global__ void computeVorticity(float* vField, Particle* field, size_t width, size_t height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        vField[y * width + x] = curl(field, width, height, x, y);
    }
}

// applies vorticity to velocity field
__global__ void applyVorticity(Particle* newField, Particle* oldField, float* vField, size_t width, size_t height, float vorticity, float dt)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    Particle& pOld = oldField[y * width + x];
    Particle& pNew = newField[y * width + x];

    float2 v = absGradient(vField, width, height, x, y);
    v.y *= -1.0f;

    float length = sqrtf(v.x * v.x + v.y * v.y) + 1e-6f;
    float2 vNorm = v * (1.0f / length);

    float2 vF = vNorm * vField[y * width + x] * vorticity;
    pNew = pOld;
    pNew.u = pNew.u + vF * dt;
}

// adds flashlight effect near the mouse position
__global__ void applyBloom(uint8_t* colorField, size_t width, size_t height, int xpos, int ypos, float radius, float bloomIntense)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pos = 4 * (y * width + x);

    float e = bloomIntense * expf(-((x - xpos) * (x - xpos) + (y - ypos) * (y - ypos) + 1.0f) / (radius * radius));

    uint8_t R = colorField[pos + 0];
    uint8_t G = colorField[pos + 1];
    uint8_t B = colorField[pos + 2];

    uint8_t maxval = fmaxf(R, fmaxf(G, B));

    colorField[pos + 0] = fminf(255.0f, R + maxval * e);
    colorField[pos + 1] = fminf(255.0f, G + maxval * e);
    colorField[pos + 2] = fminf(255.0f, B + maxval * e);
}

// performs several iterations over velocity and color fields
void FluidSimulator::computeDiffusion(dim3 numBlocks, dim3 threadsPerBlock, float dt)
{
    // diffuse velocity and color
    for (int i = 0; i < sConfig.velocityIterations; i++)
    {
        diffuse <<<numBlocks, threadsPerBlock >>> (newField, oldField, width, height, config.velocityDiffusion, dt);
        computeColor <<<numBlocks, threadsPerBlock >>> (newField, oldField, width, height, config.colorDiffusion, dt);
        std::swap(newField, oldField);
    }
}

// performs several iterations over pressure field
void FluidSimulator::computePressure(dim3 numBlocks, dim3 threadsPerBlock, float dt)
{
    for (int i = 0; i < sConfig.pressureIterations; i++)
    {
        computePressureImpl <<<numBlocks, threadsPerBlock >>> (oldField, width, height, pressureNew, pressureOld, config.pressure, dt);
        std::swap(pressureOld, pressureNew);
    }
}

// main function, calls vorticity -> diffusion -> force -> pressure -> project -> advect -> paint -> bloom
void FluidSimulator::update(float dt)
{
    deltaTime = dt;
    dim3 threadsPerBlock(sConfig.xThreads, sConfig.yThreads);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // curls and vortisity
    computeVorticity <<<numBlocks, threadsPerBlock >>> (vorticityField, oldField, width, height);
    applyVorticity <<<numBlocks, threadsPerBlock >>> (newField, oldField, vorticityField, width, height, config.vorticity, dt);
    std::swap(oldField, newField);

    // diffuse velocity and color
    computeDiffusion(numBlocks, threadsPerBlock, dt);

    // compute pressure
    computePressure(numBlocks, threadsPerBlock, dt);

    // project
    project <<<numBlocks, threadsPerBlock >>> (oldField, width, height, pressureOld);
    cudaMemset(pressureOld, 0, width * height * sizeof(float));

    // advect
    advect <<<numBlocks, threadsPerBlock >>> (newField, oldField, width, height, config.densityDiffusion, dt);
    std::swap(newField, oldField);

    // paint image
    paint <<<numBlocks, threadsPerBlock >>> (colorField, oldField, width, height);


    // copy image to cpu
    size_t size = width * height * 4 * sizeof(uint8_t);
    cudaMemcpy(pixelBuffer.data(), colorField, size, cudaMemcpyDeviceToHost);

    /*cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << cudaGetErrorName(error) << std::endl;
    }*/
}

__global__ void applyForce(Particle* field, int width, int height,
                                 float2 pos, float2 F,
                                 float radius, float dt, int minX, int minY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + minX;
    int y = blockIdx.y * blockDim.y + threadIdx.y + minY;

    if (x < width && y < height && x >= 0 && y >= 0) {
        float e = expf(-((x - pos.x) * (x - pos.x) + (y - pos.y) * (y - pos.y)) / radius);
        float2 uF = F * dt * e;
        Particle &p = field[y * width + x];
        p.u = p.u + uF;
        p.color = p.color + Color3f(0.3,0.5, 1.0) * e;
    }
}

void FluidSimulator::addForce(float2 position, float2 vector) {
    float2 F = vector * config.forceScale;

    // Calculate the bounding box of the affected area
    int minX = max(0.0f, (position.x - config.radius));
    int minY = max(0.0f, (position.y - config.radius));
    int maxX = min(width - 1, (position.x + config.radius));
    int maxY = min(height - 1, (position.y + config.radius));

    int boxWidth = maxX - minX + 1;
    int boxHeight = maxY - minY + 1;

    // Set up the kernel launch parameters
    dim3 blockSize(512, 512);  // You can adjust this based on your GPU's capabilities
    dim3 gridSize((boxWidth + blockSize.x - 1) / blockSize.x,
                  (boxHeight + blockSize.y - 1) / blockSize.y);

    // Launch the kernel only for the affected area
    applyForce<<<gridSize, blockSize>>>(
      oldField, width, height, position, F,
      config.radius, deltaTime, minX, minY
    );
}