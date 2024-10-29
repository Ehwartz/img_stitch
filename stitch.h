//
// Created by Ehwartz on 03/10/2024.
//
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <format>
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

#ifndef IMAGE_STITCHING_STITCH_H
#define IMAGE_STITCHING_STITCH_H

void image_stitch(std::vector<cv::Mat> images,
                  const std::vector<cv::detail::ImageFeatures>& features,
                  const std::vector<cv::detail::MatchesInfo>& matches_infos,
                  const std::string& result_filename);

#endif //IMAGE_STITCHING_STITCH_H
