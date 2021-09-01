#include <opencv2/imgproc.hpp>

#include "TheMachine/utils.hpp"

const static std::vector<std::string> class_names = { "person", "bicycle", "car", "motorcycle",
"airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
"stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
"cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
"tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
"baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
"chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
"mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
"refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };

void DrawRectangle(cv::Mat& img, const Detection& d, const cv::Scalar& color)
{
    const cv::Point tl(d.x1, d.y1);
    const cv::Point br(d.x2, d.y2);

    cv::rectangle(img, tl, br, color, 2, cv::LINE_AA);

    int baseline;
    cv::Size text_size = cv::getTextSize(class_names[d.cls], cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseline);
    cv::rectangle(img, tl, tl + cv::Point(text_size.width, text_size.height + baseline), color, cv::FILLED);
    cv::putText(img, class_names[d.cls], tl + cv::Point(0, text_size.height), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255,255,255), 1, cv::LINE_AA, false);
}

void DrawDetection(cv::Mat& img, const Detection& d, const cv::Scalar& color)
{
    // Person
    if (d.cls == 0)
    {
        DrawPerson(img, d, color);
    }
    // Car or truck or bus
    else if (d.cls == 2 || d.cls == 7 || d.cls == 5)
    {
        DrawCar(img, d, color);
    }
    // Train or boat
    else if (d.cls == 6 || d.cls == 8)
    {
        DrawTrain(img, d, color);
    }
    // Airplane
    else if (d.cls == 4)
    {
        DrawPlane(img, d, color);
    }
}

void DrawPerson(cv::Mat& img, const Detection& d, const cv::Scalar& color)
{
    const cv::Point tl(d.x1, d.y1);
    const cv::Point br(d.x2, d.y2);
    const float width = d.x2 - d.x1;
    const float height = d.y2 - d.y1;
    const float min_size = std::min(width, height);

    const cv::Point tl_square = min_size == width ? tl : (tl + cv::Point((width - height) / 2.0f, 0));
    const cv::Point br_square = tl_square + cv::Point(min_size, min_size);

    const float base_thickness = std::min(2.0f, std::max(1.0f, 10.0f * min_size / std::max(img.cols, img.rows)));

    // Square "machine like" detection
    // Corners
    const float corner_size = std::min(min_size, std::max(1.0f, 0.15f * min_size));
    // Top Left
    cv::line(img, tl_square, tl_square + cv::Point(corner_size, 0), color, 4 * base_thickness, cv::LINE_AA);
    cv::line(img, tl_square, tl_square + cv::Point(0, corner_size), color, 4 * base_thickness, cv::LINE_AA);
    //Top Right
    cv::line(img, tl_square + cv::Point(min_size, 0), tl_square + cv::Point(min_size - corner_size, 0), color, 4 * base_thickness, cv::LINE_AA);
    cv::line(img, tl_square + cv::Point(min_size, 0), tl_square + cv::Point(min_size, corner_size), color, 4 * base_thickness, cv::LINE_AA);
    // Bottom Right
    cv::line(img, tl_square + cv::Point(min_size, min_size), tl_square + cv::Point(min_size - corner_size, min_size), color, 4 * base_thickness, cv::LINE_AA);
    cv::line(img, tl_square + cv::Point(min_size, min_size), tl_square + cv::Point(min_size, min_size - corner_size), color, 4 * base_thickness, cv::LINE_AA);
    //Bottom Left
    cv::line(img, tl_square + cv::Point(0, min_size), tl_square + cv::Point(corner_size, min_size), color, 4 * base_thickness, cv::LINE_AA);
    cv::line(img, tl_square + cv::Point(0, min_size), tl_square + cv::Point(0, min_size - corner_size), color, 4 * base_thickness, cv::LINE_AA);


    // Side lines
    // We want N dashes and N+1 holes 40% their size between them
    size_t N = 11;
    float dash_size;
    do
    {
        N -= 2;
        dash_size = (min_size - 2 * corner_size) / (N + 0.4f * (N + 1));
    } while (dash_size < 15 && N > 1);

    for (size_t j = 0; j < N; j++)
    {
        // Top
        cv::line(img, tl_square + cv::Point(corner_size + j * dash_size + (j + 1) * 0.4f * dash_size, 0),
            tl_square + cv::Point(corner_size + (j + 1) * 1.4f * dash_size, 0), color, 2 * base_thickness, cv::LINE_AA);
        // Bottom
        cv::line(img, tl_square + cv::Point(corner_size + j * dash_size + (j + 1) * 0.4f * dash_size, min_size),
            tl_square + cv::Point(corner_size + (j + 1) * 1.4f * dash_size, min_size), color, 2 * base_thickness, cv::LINE_AA);
        // Left
        cv::line(img, tl_square + cv::Point(0, corner_size + j * dash_size + (j + 1) * 0.4f * dash_size),
            tl_square + cv::Point(0, corner_size + (j + 1) * 1.4f * dash_size), color, 2 * base_thickness, cv::LINE_AA);
        // Right
        cv::line(img, tl_square + cv::Point(min_size, corner_size + j * dash_size + (j + 1) * 0.4f * dash_size),
            tl_square + cv::Point(min_size, corner_size + (j + 1) * 1.4f * dash_size), color, 2 * base_thickness, cv::LINE_AA);
    }

    // Small dash in the middle
    // Top
    cv::line(img, tl_square + cv::Point(min_size / 2.0f, 0), tl_square + cv::Point(min_size / 2.0f, dash_size / 3.0f),
        color, 2 * base_thickness, cv::LINE_AA);
    // Bottom
    cv::line(img, tl_square + cv::Point(min_size / 2.0f, min_size), tl_square + cv::Point(min_size / 2.0f, min_size - dash_size / 3.0f),
        color, 2 * base_thickness, cv::LINE_AA);
    // Left
    cv::line(img, tl_square + cv::Point(0, min_size / 2.0f), tl_square + cv::Point(dash_size / 3.0f, min_size / 2.0f),
        color, 2 * base_thickness, cv::LINE_AA);
    // Right
    cv::line(img, tl_square + cv::Point(min_size, min_size / 2.0f), tl_square + cv::Point(min_size - dash_size / 3.0f, min_size / 2.0f),
        color, 2 * base_thickness, cv::LINE_AA);
}

void DrawCar(cv::Mat& img, const Detection& d, const cv::Scalar& color)
{
    // Draw the base layout
    DrawPerson(img, d, color);

    const cv::Point tl(d.x1, d.y1);
    const cv::Point br(d.x2, d.y2);
    
    const float width = d.x2 - d.x1;
    const float height = d.y2 - d.y1;
    const float min_size = std::min(width, height);

    const cv::Point tl_square = min_size == width ? tl : (tl + cv::Point((width - height) / 2.0f, 0));
    const cv::Point center = tl_square + cv::Point(min_size / 2, min_size / 2);
    const float base_thickness = std::min(2.0f, std::max(1.0f, 10.0f * min_size / std::max(img.cols, img.rows)));

    // Add the center circle
    cv::circle(img, center, 0.3f * min_size, color, 1 * base_thickness, cv::LINE_AA);

    const float diag_dist = std::sqrt(2.0f) * 0.15f * min_size;

    // Add the thin lines connected to the circle
    cv::line(img, tl_square, center + cv::Point(-diag_dist, -diag_dist), color, 1 * base_thickness, cv::LINE_AA);
    cv::line(img, tl_square + cv::Point(min_size, 0), center + cv::Point(diag_dist, -diag_dist), color, 1 * base_thickness, cv::LINE_AA);
    cv::line(img, tl_square + cv::Point(0, min_size), center + cv::Point(-diag_dist, diag_dist), color, 1 * base_thickness, cv::LINE_AA);
    cv::line(img, tl_square + cv::Point(min_size, min_size), center + cv::Point(diag_dist, diag_dist), color, 1 * base_thickness, cv::LINE_AA);
}

cv::Point2f RotatePoint(const cv::Point2f& inPoint, const cv::Point2f& center, const float& angDeg)
{
    cv::Point2f outPoint;

    const float angRad = angDeg / 180.0f * std::atan(1.0f) * 4;

    outPoint.x = std::cos(angRad) * (inPoint.x - center.x) - std::sin(angRad) * (inPoint.y - center.y);
    outPoint.y = std::sin(angRad) * (inPoint.x - center.x) + std::cos(angRad) * (inPoint.y - center.y);

    return outPoint + center;
}

void DrawTrain(cv::Mat& img, const Detection& d, const cv::Scalar& color)
{
    // A train has the same UI as a person, but sideways
    const cv::Point2f tl(d.x1, d.y1);
    const cv::Point2f br(d.x2, d.y2);
    const float width = d.x2 - d.x1;
    const float height = d.y2 - d.y1;
    const float min_size = std::min(width, height);

    const cv::Point2f tl_square = min_size == width ? tl : (tl + cv::Point2f((width - height) / 2.0f, 0));
    const cv::Point2f center = tl_square + cv::Point2f(min_size / 2, min_size / 2);

    const float base_thickness = std::min(2.0f, std::max(1.0f, 10.0f * min_size / std::max(img.cols, img.rows)));
    auto rotate = [&center](const cv::Point2f& p) { return RotatePoint(p, center, 45.0f); };

    // Corners
    const float corner_size = std::min(min_size, std::max(1.0f, 0.15f * min_size));
    // Top Left
    cv::line(img, rotate(tl_square), rotate(tl_square + cv::Point2f(corner_size, 0)), color, 4 * base_thickness, cv::LINE_AA);
    cv::line(img, rotate(tl_square), rotate(tl_square + cv::Point2f(0, corner_size)), color, 4 * base_thickness, cv::LINE_AA);
    //Top Right
    cv::line(img, rotate(tl_square + cv::Point2f(min_size, 0)), rotate(tl_square + cv::Point2f(min_size - corner_size, 0)), color, 4 * base_thickness, cv::LINE_AA);
    cv::line(img, rotate(tl_square + cv::Point2f(min_size, 0)), rotate(tl_square + cv::Point2f(min_size, corner_size)), color, 4 * base_thickness, cv::LINE_AA);
    // Bottom Right
    cv::line(img, rotate(tl_square + cv::Point2f(min_size, min_size)), rotate(tl_square + cv::Point2f(min_size - corner_size, min_size)), color, 4 * base_thickness, cv::LINE_AA);
    cv::line(img, rotate(tl_square + cv::Point2f(min_size, min_size)), rotate(tl_square + cv::Point2f(min_size, min_size - corner_size)), color, 4 * base_thickness, cv::LINE_AA);
    //Bottom Left
    cv::line(img, rotate(tl_square + cv::Point2f(0, min_size)), rotate(tl_square + cv::Point2f(corner_size, min_size)), color, 4 * base_thickness, cv::LINE_AA);
    cv::line(img, rotate(tl_square + cv::Point2f(0, min_size)), rotate(tl_square + cv::Point2f(0, min_size - corner_size)), color, 4 * base_thickness, cv::LINE_AA);


    // Side lines
    // We want N dashes and N+1 holes 40% their size between them
    size_t N = 11;
    float dash_size;
    do
    {
        N -= 2;
        dash_size = (min_size - 2 * corner_size) / (N + 0.4f * (N + 1));
    } while (dash_size < 15 && N > 1);

    for (size_t j = 0; j < N; j++)
    {
        // Top
        cv::line(img, rotate(tl_square + cv::Point2f(corner_size + j * dash_size + (j + 1) * 0.4f * dash_size, 0)),
            rotate(tl_square + cv::Point2f(corner_size + (j + 1) * 1.4f * dash_size, 0)), color, 2 * base_thickness, cv::LINE_AA);
        // Bottom
        cv::line(img, rotate(tl_square + cv::Point2f(corner_size + j * dash_size + (j + 1) * 0.4f * dash_size, min_size)),
            rotate(tl_square + cv::Point2f(corner_size + (j + 1) * 1.4f * dash_size, min_size)), color, 2 * base_thickness, cv::LINE_AA);
        // Left
        cv::line(img, rotate(tl_square + cv::Point2f(0, corner_size + j * dash_size + (j + 1) * 0.4f * dash_size)),
            rotate(tl_square + cv::Point2f(0, corner_size + (j + 1) * 1.4f * dash_size)), color, 2 * base_thickness, cv::LINE_AA);
        // Right
        cv::line(img, rotate(tl_square + cv::Point2f(min_size, corner_size + j * dash_size + (j + 1) * 0.4f * dash_size)),
            rotate(tl_square + cv::Point2f(min_size, corner_size + (j + 1) * 1.4f * dash_size)), color, 2 * base_thickness, cv::LINE_AA);
    }

    // Small dash in the middle
    // Top
    cv::line(img, rotate(tl_square + cv::Point2f(min_size / 2.0f, 0)), rotate(tl_square + cv::Point2f(min_size / 2.0f, dash_size / 3.0f)),
        color, 2 * base_thickness, cv::LINE_AA);
    // Bottom
    cv::line(img, rotate(tl_square + cv::Point2f(min_size / 2.0f, min_size)), rotate(tl_square + cv::Point2f(min_size / 2.0f, min_size - dash_size / 3.0f)),
        color, 2 * base_thickness, cv::LINE_AA);
    // Left
    cv::line(img, rotate(tl_square + cv::Point2f(0, min_size / 2.0f)), rotate(tl_square + cv::Point2f(dash_size / 3.0f, min_size / 2.0f)),
        color, 2 * base_thickness, cv::LINE_AA);
    // Right
    cv::line(img, rotate(tl_square + cv::Point2f(min_size, min_size / 2.0f)), rotate(tl_square + cv::Point2f(min_size - dash_size / 3.0f, min_size / 2.0f)),
        color, 2 * base_thickness, cv::LINE_AA);
}

void DrawPlane(cv::Mat& img, const Detection& d, const cv::Scalar& color)
{
    // A plane has a special triangular UI
    const cv::Point2f tl(d.x1, d.y1);
    const float width = d.x2 - d.x1;
    const float height = d.y2 - d.y1;
    const float min_size = std::min(width, height);

    const cv::Point2f tl_square = min_size == width ? tl : (tl + cv::Point2f((width - height) / 2.0f, 0));
    const cv::Point2f tm(tl_square + cv::Point2f(min_size / 2.0f, 0));
    const cv::Point2f bl(tl_square + cv::Point2f(0.0f, min_size));
    const cv::Point2f br(tl_square + cv::Point2f(min_size, min_size));

    const float base_thickness = std::min(2.0f, std::max(1.0f, 10.0f * min_size / std::max(img.cols, img.rows)));

    const float side_ratio = std::sqrt(5) / 2.0f;
    const float angle = std::atan(2.0f) * 180.0f / (4 * std::atan(1.0f));

    // Corners
    const float corner_size = std::min(min_size, std::max(1.0f, 0.15f * min_size));
    //Top
    cv::line(img, tm, RotatePoint(tm + cv::Point2f(-side_ratio * corner_size, 0), tm, -angle), color, 4 * base_thickness, cv::LINE_AA);
    cv::line(img, tm, RotatePoint(tm + cv::Point2f(side_ratio * corner_size, 0), tm, angle), color, 4 * base_thickness, cv::LINE_AA);
    // Bottom Right
    cv::line(img, br, RotatePoint(br + cv::Point2f(-side_ratio * corner_size, 0), br, angle), color, 4 * base_thickness, cv::LINE_AA);
    cv::line(img, br, br + cv::Point2f(-corner_size, 0), color, 4 * base_thickness, cv::LINE_AA);
    //Bottom Left
    cv::line(img, bl, RotatePoint(bl + cv::Point2f(side_ratio * corner_size, 0), bl, -angle), color, 4 * base_thickness, cv::LINE_AA);
    cv::line(img, bl, bl + cv::Point2f(corner_size, 0), color, 4 * base_thickness, cv::LINE_AA);

    // Side lines
    // We want N dashes and N+1 holes 40% their size between them
    size_t N = 11;
    float dash_size;
    do
    {
        N -= 2;
        dash_size = (min_size - 2 * corner_size) / (N + 0.4f * (N + 1));
    } while (dash_size < 15 && N > 1);

    for (size_t j = 0; j < N; j++)
    {
        // Bottom
        cv::line(img, bl + cv::Point2f(corner_size + j * dash_size + (j + 1) * 0.4f * dash_size, 0),
            bl + cv::Point2f(corner_size + (j + 1) * 1.4f * dash_size, 0), color, 2 * base_thickness, cv::LINE_AA);
        // Left
        cv::line(img, RotatePoint(bl + side_ratio * cv::Point2f(corner_size + j * dash_size + (j + 1) * 0.4f * dash_size, 0), bl, -angle),
            RotatePoint(bl + side_ratio * cv::Point2f(corner_size + (j + 1) * 1.4f * dash_size), bl, -angle), color, 2 * base_thickness, cv::LINE_AA);
        // Right
        cv::line(img, RotatePoint(br - side_ratio * cv::Point2f(corner_size + j * dash_size + (j + 1) * 0.4f * dash_size, 0), br, angle),
            RotatePoint(br - side_ratio * cv::Point2f(corner_size + (j + 1) * 1.4f * dash_size, 0), br, angle), color, 2 * base_thickness, cv::LINE_AA);
    }

    // Small dash on bottom line
    cv::line(img, bl + cv::Point2f(min_size / 2.0f, 0), bl + cv::Point2f(min_size / 2.0f, - dash_size / 3.0f),
        color, 2 * base_thickness, cv::LINE_AA);


    // Center circle
    const cv::Point2f circle_center = (tm + br + bl) / 3.0f;
    const float circle_radius = 0.15f * min_size;
    cv::circle(img, circle_center, circle_radius, color, 1 * base_thickness, cv::LINE_AA);

    // Lines next to the circle
    // Left
    cv::line(img, circle_center + cv::Point2f(-circle_radius, -circle_radius), circle_center + cv::Point2f(-circle_radius, circle_radius),
        color, 1 * base_thickness, cv::LINE_AA);
    cv::line(img, circle_center + cv::Point2f(-circle_radius, -circle_radius), circle_center + cv::Point2f(-circle_radius + dash_size / 3.0f, -circle_radius),
        color, 1 * base_thickness, cv::LINE_AA);
    cv::line(img, circle_center + cv::Point2f(-circle_radius, circle_radius), circle_center + cv::Point2f(-circle_radius + dash_size / 3.0f, circle_radius),
        color, 1 * base_thickness, cv::LINE_AA);
    // Right
    cv::line(img, circle_center + cv::Point2f(circle_radius, -circle_radius), circle_center + cv::Point2f(circle_radius, circle_radius),
        color, 1 * base_thickness, cv::LINE_AA);
    cv::line(img, circle_center + cv::Point2f(circle_radius, -circle_radius), circle_center + cv::Point2f(circle_radius - dash_size / 3.0f, -circle_radius),
        color, 1 * base_thickness, cv::LINE_AA);
    cv::line(img, circle_center + cv::Point2f(circle_radius, circle_radius), circle_center + cv::Point2f(circle_radius - dash_size / 3.0f, circle_radius),
        color, 1 * base_thickness, cv::LINE_AA);
}
