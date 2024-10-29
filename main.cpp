#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"

#include "stitch.h"
#include "classify.h"
#include "draw.h"

void classify_and_stitch_images(const std::string &images_path)
{
    std::vector<std::string> image_names;
    cv::glob(images_path + "/*.jpg", image_names, false);
//    for (const auto& name:image_names)
//    {
//        std::cout<<name<<std::endl;
//    }
    int num_images = (int) image_names.size();
    std::vector<cv::Mat> images(num_images);

    for (int i = 0; i < num_images; ++i)
    {
        images[i] = cv::imread(cv::samples::findFile(image_names[i]));
//        std::cout<<images[i].size<<std::endl;
    }
    auto panoramas = panorama_classify(images, 0.8);

    for (int i = 0; i < panoramas.size(); ++i)
    {
        auto &panorama = panoramas[i];
        auto &panorama_images = std::get<0>(panorama);
        auto &panorama_features = std::get<1>(panorama);
        auto &panorama_matches_infos = std::get<2>(panorama);
        image_stitch(panorama_images,
                     panorama_features,
                     panorama_matches_infos,
                     std::format("./stitch_results/panorama{}.jpg", i));
    }
}

void draw_example_matches(std::vector<std::string> image_names,
                          double distance_threshold, const std::string &result_filename)
{
    int num_images = (int) image_names.size();
    std::vector<cv::Mat> images(num_images);

    for (int i = 0; i < num_images; ++i)
    {
        images[i] = cv::imread(cv::samples::findFile(image_names[i]));
    }
    cv::Mat out_img = draw_matches(images, distance_threshold);
    cv::imwrite(result_filename, out_img);
};

int main()
{
    classify_and_stitch_images("./imgs");

    return 0;
}
