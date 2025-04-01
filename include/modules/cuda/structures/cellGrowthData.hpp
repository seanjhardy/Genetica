#ifndef CELL_DIVISION_DATA
#define CELL_DIVISION_DATA

// CellDivisionData class in header file
#include "cuda_runtime.h"

// Maximum number of division events to track in the entire simulation each step
#define MAX_DIVISION_EVENTS 200

class cellGrowthData {
public:
    bool enabled;
    int* numDivisions = nullptr;
    size_t* dividingCellIdxs = nullptr;

    cellGrowthData();
    void reset();
    void setEnabled(bool enabled);
    __device__ bool push(size_t cellIdx);

    [[nodiscard]] int getNumDivisions() const;

    // Retrieve dividing cell indices (only when needed)
    void getDividingCellIndices(size_t* hostIndices, int count);
};

#endif