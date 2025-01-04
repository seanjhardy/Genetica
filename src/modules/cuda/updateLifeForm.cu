#include <modules/cuda/lifeForm.hpp>
#include <geneticAlgorithm/LifeForm.hpp>

__global__ void cloneLifeFormKernel(GPUVector<LifeForm>& lifeForms, int lifeFormID) {
    LifeForm& lf = lifeForms[lifeFormID];
    //lf.clone(false);
}

__global__ void mutateLifeFormKernel(GPUVector<LifeForm>& lifeForms, int lifeFormID) {
    LifeForm& lf = lifeForms[lifeFormID];
    //lf.mutate();
}

__global__ void killLifeFormKernel(GPUVector<LifeForm>& lifeForms, int lifeFormID) {
    LifeForm& lf = lifeForms[lifeFormID];
    //lf.kill();
}

__global__ void energiseLifeFormKernel(GPUVector<LifeForm>& lifeForms, int lifeFormID) {
    LifeForm& lf = lifeForms[lifeFormID];
    lf.energy += 100;
}

__global__ void getLifeFormKernel(GPUVector<LifeForm>& lifeForms, int lifeFormID, LifeForm* lf) {
    //*lf = lifeForms[lifeFormID];
}


void cloneLifeForm(GPUVector<LifeForm>& lifeForms, int lifeFormID) {
    cloneLifeFormKernel<<<1, 1>>>(lifeForms, lifeFormID);
}

void mutateLifeForm(GPUVector<LifeForm>& lifeForms, int lifeFormID) {
    mutateLifeFormKernel<<<1, 1>>>(lifeForms, lifeFormID);
}

void killLifeForm(GPUVector<LifeForm>& lifeForms, int lifeFormID) {
    killLifeFormKernel<<<1, 1>>>(lifeForms, lifeFormID);
}

void energiseLifeForm(GPUVector<LifeForm>& lifeForms, int lifeFormID) {
    energiseLifeFormKernel<<<1, 1>>>(lifeForms, lifeFormID);
}

LifeForm* getLifeForm(GPUVector<LifeForm>& lifeForms, int lifeFormID) {
    LifeForm* lf_host;
    LifeForm* lf;
    cudaMalloc(&lf, sizeof(LifeForm));
    getLifeFormKernel<<<1, 1>>>(lifeForms, lifeFormID, lf);
    cudaMemcpy(&lf_host, lf, sizeof(LifeForm), cudaMemcpyDeviceToHost);
    return lf;
}