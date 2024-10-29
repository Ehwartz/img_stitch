//
// Created by Ehwartz on 03/18/2024.
//
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <format>
#include <map>
#include <tuple>
#include <algorithm>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

#ifndef IMAGE_STITCHING_DRAW_H
#define IMAGE_STITCHING_DRAW_H

cv::Mat draw_matches(std::vector<cv::Mat> images, double distance_threshold);

#endif //IMAGE_STITCHING_DRAW_H
