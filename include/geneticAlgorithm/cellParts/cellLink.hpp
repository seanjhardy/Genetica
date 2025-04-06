#ifndef CELL_LINK
#define CELL_LINK

#include <geneticAlgorithm/cellParts/cell.hpp>
#include <modules/physics/point.hpp>

class LifeForm;

class CellLink {
public:
    size_t lifeFormIdx;
    size_t cellAIdx;
    size_t cellBIdx;
    size_t pointAIdx;
    size_t pointBIdx;
    float angleFromA;
    float angleFromB;
    float stiffness = 1.0f;
    double length;
    double targetLength;

    CellLink() = default;
    CellLink(size_t lifeFormIdx, size_t cellAIdx, size_t cellBIdx, size_t pointAIdx, size_t pointBIdx,
             float startLength, float targetLength, float angleFromA, float angleFromB, float stiffness);

    __host__ void renderCellWalls(VertexManager& vertexManager, vector<Cell>& cells, vector<Point>& points);
    __host__ void renderBody(VertexManager& vertexManager, vector<Cell>& cells, vector<Point>& points);
    __host__ void renderDetails(VertexManager& vertexManager, vector<Cell>& cells, vector<Point>& points);

    __host__ __device__ void adjustSize(float distance) {
        length += distance;
    }
};

#endif