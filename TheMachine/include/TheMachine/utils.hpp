#pragma once

#include <opencv2/core.hpp>

struct Detection
{
	float x1, y1, x2, y2;
	float score;
	int cls;
};

void DrawRectangle(cv::Mat& img, const Detection& d, const cv::Scalar& color);

void DrawDetection(cv::Mat& img, const Detection& d, const cv::Scalar& color);

void DrawPerson(cv::Mat& img, const Detection& d, const cv::Scalar& color);

void DrawCar(cv::Mat& img, const Detection& d, const cv::Scalar& color);

cv::Point2f RotatePoint(const cv::Point2f& inPoint, const cv::Point2f& center, const float& angDeg);

void DrawTrain(cv::Mat& img, const Detection& d, const cv::Scalar& color);

void DrawPlane(cv::Mat& img, const Detection& d, const cv::Scalar& color);
