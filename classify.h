//
// Created by Ehwartz on 03/16/2024.
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

#ifndef IMAGE_STITCHING_CLASSIFY_H
#define IMAGE_STITCHING_CLASSIFY_H

class Node
{
public:
    explicit Node(int idx);

    ~Node() = default;

    int idx;
    std::vector<Node *> neighbors;

    void connect(Node *dst);
    bool connected = false;
};

class Graph
{
public:
    explicit Graph(int n, const std::vector<cv::detail::MatchesInfo>& matches_infos);

    ~Graph();
    std::vector<Node*> nodes;
    int n;
    cv::Mat A;
    std::vector<std::vector<Node*>> clusters;
    void clustering(double threshold);
    void traverse(Node *ptr, std::vector<Node*>&cluster,std::vector<Node*>&queue);
    void cout_clusters();
};

int find_node(Node *node, std::vector<Node*> nodes);


std::vector<std::tuple<std::vector<cv::Mat>,
                       std::vector<cv::detail::ImageFeatures>,
                       std::vector<cv::detail::MatchesInfo>>>
panorama_classify(std::vector<cv::Mat> images, double threshold);

#endif //IMAGE_STITCHING_CLASSIFY_H
