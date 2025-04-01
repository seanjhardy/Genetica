#ifndef CELL_LINK
#define CELL_LINK

#include <geneticAlgorithm/cellParts/cell.hpp>
#include <modules/physics/point.hpp>

class LifeForm;

class CellLink {
public:
    int lifeFormIdx;
    int cellAIdx;
    int cellBIdx;
    int pointAIdx;
    int pointBIdx;
    float angle;
    float prevAngle = -1.0f;
    float stiffness = 1.0f;
    double length;
    double targetLength;

    CellLink() = default;
    CellLink(size_t lifeFormId, size_t cellAId, size_t cellBId, size_t p1, size_t p2,
             float startLength, float targetLength, float angle, float stiffness);

    __host__ void renderCellWalls(VertexManager& vertexManager, vector<Cell>& cells, vector<Point>& points);
    __host__ void renderBody(VertexManager& vertexManager, vector<Cell>& cells, vector<Point>& points);
    __host__ void renderDetails(VertexManager& vertexManager, vector<Cell>& cells, vector<Point>& points);

    __host__ __device__ void adjustSize(float distance) {
        length += distance;
    }
};

#endif