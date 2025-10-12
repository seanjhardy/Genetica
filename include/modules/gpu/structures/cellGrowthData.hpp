#ifndef CELL_DIVISION_DATA
#define CELL_DIVISION_DATA

// CellDivisionData class in header file

// Maximum number of division events to track in the entire simulation each step
#define MAX_DIVISION_EVENTS 200

class cellGrowthData {
public:
    bool enabled;
    int* numDivisions = nullptr;
    size_t* dividingCellIdxs = nullptr;

    cellGrowthData() {};
    void reset() {};
    void setEnabled(bool enabled) {};
    bool push(size_t cellIdx) { return false; };

    [[nodiscard]] int getNumDivisions() const { return 0; };

    // Retrieve dividing cell indices (only when needed)
    void getDividingCellIndices(size_t* hostIndices, int count) {};
};

#endif