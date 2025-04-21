#include "cmath"
#include "modules/physics/point.hpp"
#include "modules/utils/operations.hpp"
#include "modules/utils/GPU/mathUtils.hpp"

__host__ __device__ float2 vec(float direction) {
    return {cosf(direction), sinf(direction)};
}

__host__ __device__ float dir(float2 p1, float2 p2) {
    float2 d = p1 - p2;
    return std::atan2f(d.y, d.x);
}

__host__ __device__ float2 rotate(float2 point, float angle) {
    float2 d = vec(angle);
    float x = point.x * d.x - point.y * d.y;
    float y = point.x * d.y + point.y * d.x;
    return make_float2(x, y);
}

__host__ __device__ double2 rotate(double2 point, float angle) {
    float2 d = vec(angle);
    double x = point.x * d.x - point.y * d.y;
    double y = point.x * d.y + point.y * d.x;
    return make_double2(x, y);
}


__host__ __device__ float2 rotateOrigin(float2 point, float2 origin, float angle) {
    float2 d = vec(angle);
    float x = (point.x - origin.x) * d.x - (point.y - origin.y) * d.y;
    float y = (point.x - origin.x) * d.y + (point.y - origin.y) * d.x;
    return make_float2(x, y);
}

__host__ __device__ float diff(float2 p1, float2 p2) {
    return abs(p1.x - p2.x) + abs(p1.y - p2.y);
}

__host__ __device__ float sum(float2 p1) {
    return p1.x + p1.y;
}

__host__ __device__ float sum(double2 p1) {
    return p1.x + p1.y;
}

__host__ __device__ float sum(float3 p1) {
    return p1.x + p1.y + p1.z;
}

__host__ __device__ float distanceBetween(float2 p1, float2 p2) {
    float2 d = p1 - p2;
    return sqrt(sum(d * d));
}

__host__ __device__ float distanceBetween(double2 p1, double2 p2) {
    double2 d = p1 - p2;
    return sqrt(sum(d * d));
}

__host__ __device__ float distanceBetween(float3 p1, float3 p2) {
    float3 d = p1 - p2;
    return sqrt(sum(d * d));
}

__host__ __device__ float angleBetween(float2 p1, float2 p2) {
    float2 d = p2 - p1;
    return atan2f(d.y, d.x);
}

__host__ __device__ float magnitude(float2 p1) {
    return sqrtf(p1.x * p1.x + p1.y * p1.y);
}

__host__ __device__ double magnitude(double2 p1) {
    return sqrt(p1.x * p1.x + p1.y * p1.y);
}

__host__ __device__ float magnitude(float3 p1) {
    return sqrt(p1.x * p1.x + p1.y * p1.y + p1.z * p1.z);
}

std::vector<float2> findPerpendicularPoints(const Point& point1, const Point& point2, float r1, float r2) {
    float x1 = point1.pos.x, y1 = point1.pos.y;
    float x2 = point2.pos.x, y2 = point2.pos.y;

    // Ensure the points are not identical to avoid division by zero in atan2
    if (point1.pos == point2.pos) {
        x1 += 1e-8;
    }

    double angle = FastMath::atan2f(y2 - y1, x2 - x1);
    float anglePlus90 = angle + M_PI / 2;
    float angleMinus90 = angle - M_PI / 2;

    return {
        point1.getPos() + vec(anglePlus90) * r1,
        point1.getPos() + vec(angleMinus90) * r1,
        point2.getPos() + vec(angleMinus90) * r2,
        point2.getPos() + vec(anglePlus90) * r2
    };
}


sf::Color interpolate(const sf::Color c1, const sf::Color c2, float x) {
    int r = static_cast<int>(c1.r + (c2.r - c1.r) * x);
    int g = static_cast<int>(c1.g + (c2.g - c1.g) * x);
    int b = static_cast<int>(c1.b + (c2.b - c1.b) * x);
    int a = static_cast<int>(c1.a + (c2.a - c1.a) * x);
    return sf::Color(clamp(0, r, 255),
                     clamp(0, g, 255),
                     clamp(0, b, 255),
                     clamp(0, a, 255));
}

__host__ __device__ float normAngle(float angle) {
    return atan2f(sin(angle), cos(angle));
}

__host__ __device__ double normAngle(double angle) {
    return atan2(sin(angle), cos(angle));
}

float angleDiff(float angle1, float angle2, bool norm) {
    float diff = angle2 - angle1;
    if (norm) {
        diff = normAngle(diff);
    }
    return diff;
}

std::vector<std::pair<int, int>> bezier(int x0, int y0, int x1, int y1, int x2, int y2, int num_points) {
    auto bezierInterpolation = [](float t, int p0, int p1, int p2) {
        float x = (1 - t) * (1 - t) * p0 + 2 * (1 - t) * t * p1 + t * t * p2;
        return static_cast<int>(x);
    };

    std::vector<std::pair<int, int>> points;
    for (int i = 0; i <= num_points; ++i) {
        float t = static_cast<float>(i) / num_points;
        int x = bezierInterpolation(t, x0, x1, x2);
        int y = bezierInterpolation(t, y0, y1, y2);
        points.emplace_back(x, y);
    }
    return points;
}

float smoothAngle(float angle1, float angle2, float tolerance) {
    float diff = angleDiff(angle1, angle2);
    tolerance = tolerance * M_PI / 180;

    if (std::abs(diff) < tolerance) {
        if (diff > 0) {
            return normAngle(angle2 - tolerance);
        }
        else {
            return normAngle(angle2 + tolerance);
        }
    }
    return angle1;
}

float2 getPointOnSegment(float length, float r1, float r2, float angle) {
    const float half_length = length * 0.5f;
    const float two_pi = 2.0f * M_PI;

    // Normalize angle to [0, 2π)
    angle = fmod(angle, two_pi);
    angle += (angle < 0) * two_pi;

    const float tr = FastMath::atan2f(r2, half_length);
    const float br = two_pi - tr;

    if (angle > br || angle < tr) {
        float new_a = (angle - (angle > br ? br : br - two_pi)) * (M_PI / (tr + tr)) - M_PI_2;
        return vec(new_a) * r2 + make_float2(length, 0);
    }

    const float tr1 = FastMath::atan2f(r1, half_length);
    const float tl = M_PI - tr1;
    const float bl = two_pi - tr1;

    if (tl < angle && angle < bl) {
        float new_a = (angle - tl) * (M_PI / (bl - tl)) + M_PI_2;
        return vec(new_a) * r1;
    }

    float i, start_r, end_r;
    if (angle > bl) {
        i = (angle - bl) / (br - bl);
        start_r = -r1;
        end_r = -r2;
    }
    else {
        i = (angle - tr) / (tl - tr);
        start_r = r2;
        end_r = r1;
    }

    return {(1.0f - i) * length, (1.0f - i) * start_r + i * end_r};
}


std::vector<float> geometricProgression(int n, float r) {
    std::vector<float> percentages;

    // Calculate the initial value A
    double A = (100.0 * (1 - r)) / (1 - pow(r, n));

    // Generate the percentages
    for (int i = 0; i < n; ++i) {
        percentages.push_back(A * pow(r, i));
    }

    return percentages;
}
