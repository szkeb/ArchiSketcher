#pragma once

#include <unordered_set>
#include <variant>
#include "ellipse_fit.h"

#define _USE_MATH_DEFINES
#include <math.h>

namespace DLA {
    struct Vector2d
    {
        float x;
        float y;

        double Angle() const
        {
            return std::atan2(y, x);
        }

        double NormalizedAngle() const
        {
            double angle = std::atan2(y, x);
            if (angle < 0)
            {
                angle += 2 * M_PI;
            }

            return angle;
        }
    };

    struct Point
    {
        float x;
        float y;

        static constexpr float kEpsilon = 1e-8;

        bool operator==(const Point& other) const
        {
            float sqDist = (x - other.x) * (x - other.x) + (y - other.y) * (y - other.y);
            return sqDist < kEpsilon;
        }

        Vector2d operator-(const Point& other) const
        {
            return { x - other.x, y - other.y };
        }
    };
}

namespace std {
    template<> struct hash<DLA::Point>
    {
        size_t operator()(const DLA::Point& point) const
        {
            return hash<float>()(point.x) ^ (hash<float>()(point.y) << 1);
        }
    };
}

struct Line
{
    DLA::Point start;
    DLA::Point end;

    Line(const std::vector<DLA::Point>& points)
    {
        size_t n = points.size();

        float xMin = (*points.begin()).x;
        float xMax = (*points.begin()).x;
        float yMin = (*points.begin()).y;
        float yMax = (*points.begin()).y;

        float xMean = 0;
        float yMean = 0;
        for (const auto& p : points)
        {
            xMean += p.x;
            yMean += p.y;
            xMin = std::min(xMin, p.x);
            xMax = std::max(xMax, p.x);
            yMin = std::min(yMin, p.y);
            yMax = std::max(yMax, p.y);
        }
        xMean /= n;
        yMean /= n;
        float xDelta = xMax - xMin;
        float yDelta = yMax - yMin;

        float numerator = 0;
        float denominator = 0;
        for (const auto& p : points)
        {
            numerator += (p.x - xMean) * (p.y - yMean);
            denominator += (p.x - xMean) * (p.x - xMean);
        }
        
        float m = numerator / denominator;
        float b = yMean - m * xMean;

        if (xDelta > yDelta)
        {
            start = { xMin, m * xMin + b };
            end = { xMax, m * xMax + b };
        }
        else
        {
            start = { (yMin - b) / m, yMin};
            end = { (yMax - b ) / m, yMax };
        }
    }

    std::vector<DLA::Point> Tessellate(size_t n) const
    {
        std::vector<DLA::Point> result;
        float dx = (end.x - start.x) / n;
        float dy = (end.y - start.y) / n;
        for (size_t i = 0; i < n; i++)
        {
            result.push_back({start.x + i*dx, start.y + i*dy});
        }
        return result;
    }
};

struct Ellipse
{
    double Rx;
    double Ry;
    double Cx;
    double Cy;
    double R;

    double start;
    double end;

    Ellipse(std::vector<DLA::Point> points)
    {
        std::vector<std::vector<double>> data;
        for (const auto& p : points)
        {
            data.push_back({ p.x, p.y });
        }

        EllipseFit ellipse;
        ellipse.set(data);
        ellipse.fit(Cx, Cy, R, Rx, Ry);

        const DLA::Point center{ Cx, Cy };
        size_t cw = 0;
        size_t ccw = 0;
        double last = (center - points[0]).Angle();
        for (size_t i = 1; i < points.size(); i++)
        {
            const auto v = center - points[i];
            double t = v.Angle();
            if (last < t)
            {
                cw++;
            }
            else
            {
                if (last > 0 && t < 0 && std::abs(last - t) > M_PI)
                {
                    cw++;
                }
                else
                {
                    ccw++;
                }
            }
            last = t;
        }

        start = (points[0] - center).NormalizedAngle();
        end = (points[points.size() - 1] - center).NormalizedAngle();

        bool clockwise = cw > ccw;

        if (start > end)
        {
            if (clockwise)
            {
                end += 2 * M_PI;
            }
            else
            {
                std::swap(start, end);
            }
        }
        else if(!clockwise)
        {
            start += 2 * M_PI;
            std::swap(start, end);
        }
    }

    std::vector<DLA::Point> Tessellate(size_t n) const
    {
        auto X = [&](float t) { return (float)(Rx * std::cos(t) * std::cos(R) - Ry * std::sin(t)*std::sin(R) + Cx); };
        auto Y = [&](float t) { return (float)(Rx * std::cos(t) * std::sin(R) + Ry * std::sin(t) * std::cos(R) + Cy); };

        std::vector<DLA::Point> points;

        float step = (end - start) / n;
        for (float t = start; t <= end; t += step)
        {
            points.push_back(DLA::Point{ X(t), Y(t) });
        }
        return points;
    }
};