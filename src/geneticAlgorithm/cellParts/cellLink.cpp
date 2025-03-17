#include <geneticAlgorithm/cellParts/cellLink.hpp>
#include "geneticAlgorithm/lifeform.hpp"

CellLink::CellLink(const size_t lifeFormId, const size_t cellAId, const size_t cellBId, const size_t p1, const size_t p2, float startLength, float angle, float stiffness)
: lifeFormId(lifeFormId), cellAId(cellAId), cellBId(cellBId), p1(p1), p2(p2), length(startLength), targetLength(startLength), angle(angle), stiffness(stiffness) {};


void CellLink::renderCellWalls(VertexManager& vertexManager, vector<Cell>& cells, vector<Point>& points) {
    const Point point1 = points[p1];
    const Point point2 = points[p2];
    const Cell* cell1 = &cells[cellAId];
    const Cell* cell2 = &cells[cellBId];
    const sf::Color cell1Color = brightness(cell1->getColor(), 0.6);
    const sf::Color cell2Color = brightness(cell2->getColor(), 0.6);

    // Find the angle between the points and draw a polygon connecting them:
    // From the center of the first, add a vertex on the circumference of the point tangential to the angle between p1 and p2
    // Then connect it to the vertex on the circumference of the second tangential to the angle between p2 and p1
    // Repeat for the other side
    const float angle = atan2(point2.pos.y - point1.pos.y, point2.pos.x - point1.pos.x);
    const float angle1 = angle + M_PI_HALF;
    const float angle2 = angle - M_PI_HALF;

    if (vertexManager.getSizeInView(point1.radius) < 5 && vertexManager.getSizeInView(point2.radius) < 5) return;
    float2 v1 = point1.getPos() + make_float2(cos(angle1), sin(angle1)) * (point1.radius + cell1->thickness);
    float2 v2 = point2.getPos() + make_float2(cos(angle1), sin(angle1)) * (point2.radius + cell2->thickness);
    float2 v3 = point2.getPos() + make_float2(cos(angle2), sin(angle2)) * (point2.radius + cell2->thickness);
    float2 v4 = point1.getPos() + make_float2(cos(angle2), sin(angle2)) * (point1.radius + cell1->thickness);
    vertexManager.addPolygon(std::vector<VertexManager::Vertex>({
        {v1, cell1Color},
        {v2, cell2Color},
        {v3, cell2Color},

        {v3, cell2Color},
        {v4, cell1Color},
        {v1, cell1Color}}));
}

void CellLink::renderBody(VertexManager& vertexManager, vector<Cell>& cells, vector<Point>& points) {
    const Point point1 = points[p1];
    const Point point2 = points[p2];
    const Cell* cell1 = &cells[cellAId];
    const Cell* cell2 = &cells[cellBId];
    const sf::Color cell1Color = cell1->getColor();
    const sf::Color cell2Color = cell2->getColor();

    // Find the angle between the points and draw a polygon connecting them:
    // From the center of the first, add a vertex on the circumference of the point tangential to the angle between p1 and p2
    // Then connect it to the vertex on the circumference of the second tangential to the angle between p2 and p1
    // Repeat for the other side
    const float angle = atan2(point2.pos.y - point1.pos.y, point2.pos.x - point1.pos.x);
    const float angle1 = angle + M_PI_HALF;
    const float angle2 = angle - M_PI_HALF;
    float2 v1 = point1.getPos() + make_float2(cos(angle1), sin(angle1)) * point1.radius;
    float2 v2 = point2.getPos() + make_float2(cos(angle1), sin(angle1)) * point2.radius;
    float2 v3 = point2.getPos() + make_float2(cos(angle2), sin(angle2)) * point2.radius;
    float2 v4 = point1.getPos() + make_float2(cos(angle2), sin(angle2)) * point1.radius;
    vertexManager.addPolygon(std::vector<VertexManager::Vertex>({
        {v1, cell1Color},
        {v2, cell2Color},
        {v3, cell2Color},

        {v3, cell2Color},
        {v4, cell1Color},
        {v1, cell1Color}}));
}



/*
void CellLink::moveCell1(Cell* newCell) {
    Connection* connection = lifeForm->getEnv()->getConnection(connectionIdx);
    connection->p1 = newCell->pointIdx;
    lifeForm->getEnv()->updateConnection(connectionIdx, *connection);
    cell1 = newCell;
}

void CellLink::moveCell2(Cell* newCell) {
    Connection* connection = lifeForm->getEnv()->getConnection(connectionIdx);
    connection->p2 = newCell->pointIdx;
    lifeForm->getEnv()->updateConnection(connectionIdx, *connection);
    cell2 = newCell;
}

float CellLink::getBuildCost() const {
    Connection* connection = lifeForm->getEnv()->getConnection(connectionIdx);
    Point* p1 = lifeForm->getEnv()->getPoint(cell1->pointIdx);
    Point* p2 = lifeForm->getEnv()->getPoint(cell2->pointIdx);
    float buildCost = 1.0f;
    //+ 0.2f * (bone ? 1.0f : 0.0f) * boneDensity
    //+ 0.5f * (muscle ? 1.0f : 0.0f) * muscleStrength
    //+ 0.01f * (nerve ? 1.0f : 0.0f)
    //+ 0.01f * (fat ? 1.0f : 0.0f) * fatSize;
    buildCost *= connection->length * (p1->radius + p2->radius) / 2;
    return buildCost;
};
*/
/*void simulate(float dt, float massChange) {
        Connection* connection = lifeForm->getEnv()->getConnection(connectionIdx);
        Point* p1 = lifeForm->getEnv()->getPoint(connection->p1);
        Point* p2 = lifeForm->getEnv()->getPoint(connection->p2);

        float startWidth = p1->radius;
        float endWidth = p2->radius;
        float avgWidth = (startWidth + endWidth) * 0.5f;

        // Don't ask me how I got this equation :skull:
        float deltaGrowth = -growthProgress + sqrt(growthProgress*growthProgress
                                                   + (massChange * dt) / (avgWidth * finalLength));

        // Ensure growthFraction + deltaGrowth does not go above 1
        double newGrowthFraction = min(growthProgress + deltaGrowth, 1.0f);
        deltaGrowth = newGrowthFraction - growthProgress;

        float growthEnergyCost = getBuildCost() * LifeForm::BUILD_COST_SCALE * deltaGrowth;

        if (lifeForm->energy < growthEnergyCost) return;

        lastGrowthProgress = growthProgress;
        growthProgress = newGrowthFraction;
        lifeForm->energy -= growthEnergyCost;

        // Calculate the growthFraction of each point based on gene, segment growthFraction %, and lifeform growthFraction
        p1->radius = startWidth * growthProgress;
        p2->radius = endWidth * growthProgress;


        if (lifeForm != nullptr || false) {
            float distToParentStart = sqrt(sum(pointOnParent * pointOnParent));
            lifeForm->getEnv()->addConnection(parent->startPoint, startPoint, distToParentStart);

            float parentLength = dynamic_cast<SegmentType *>(parent->schematic->type)->length * lifeForm->size;
            float2 d = pointOnParent - make_float2(parentLength, 0);
            float distToParentEnd = sqrt(sum(d * d));
            lifeForm->getEnv()->addConnection(parent->endPoint, startPoint, distToParentEnd);
        }

        // Return true if fully built
        return true;
    }*/
