//
// Created by Ehwartz on 03/22/2024.
//
#include "draw.h"

cv::Mat draw_matches(std::vector<cv::Mat> images, double distance_threshold)
{

    auto img1 = images[0];
    auto img2 = images[1];

    std::vector<cv::KeyPoint> img1_keypoints, img2_keypoints;
    cv::Mat img1_descriptors, img2_descriptors;

    cv::Ptr<cv::Feature2D> detector = cv::ORB::create(500, 1.2f,
                                                      8, 31, 0, 2,
                                                      cv::ORB::HARRIS_SCORE, 31, 20);
    detector->detect(img1, img1_keypoints);
    detector->detect(img2, img2_keypoints);
    detector->compute(img1, img1_keypoints, img1_descriptors);
    detector->compute(img2, img2_keypoints, img2_descriptors);
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);

    std::vector<cv::DMatch> matches;
    matcher.match(img1_descriptors, img2_descriptors, matches);

    std::vector<cv::DMatch> filtered_matches;
    for (auto & match :matches)
    {
        if (match.distance < distance_threshold)
            filtered_matches.emplace_back(match);
    }
    cv::Mat img_output;
    drawMatches(img1, img1_keypoints, img2, img2_keypoints, filtered_matches, img_output);
    return img_output;

}