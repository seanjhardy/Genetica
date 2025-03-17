#ifndef CELL_LINK
#define CELL_LINK

#include <geneticAlgorithm/cellParts/cell.hpp>
#include <modules/physics/point.hpp>

class LifeForm;

class CellLink {
public:
    int lifeFormId;
    int cellAId;
    int cellBId;
    int p1;
    int p2;
    float angle;
    float stiffness = 1.0f;
    float length;
    float targetLength;

    CellLink() = default;
    CellLink(size_t lifeFormId, size_t cellAId, size_t cellBId, size_t p1, size_t p2, float startLength, float angle, float stiffness);

    __host__ void renderCellWalls(VertexManager& vertexManager, vector<Cell>& cells, vector<Point>& points);
    __host__ void renderBody(VertexManager& vertexManager, vector<Cell>& cells, vector<Point>& points);

    __host__ __device__ void adjustSize(float distance) {
        length += distance;
    }
};

#endif