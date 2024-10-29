//
// Created by Ehwartz on 03/10/2024.
//
#include "stitch.h"
void image_stitch(std::vector<cv::Mat> images,
                  const std::vector<cv::detail::ImageFeatures>& features,
                  const std::vector<cv::detail::MatchesInfo>& matches_infos,
                  const std::string& result_filename)
{
    size_t num_images = images.size();

    cv::Ptr<cv::detail::Estimator> estimator = cv::makePtr<cv::detail::HomographyBasedEstimator>();

    std::vector<cv::detail::CameraParams> cameras_params;
    (*estimator)(features, matches_infos, cameras_params);

    for (auto & cameras_param : cameras_params)
        cameras_param.R.convertTo(cameras_param.R, CV_32F);

    cv::Ptr<cv::detail::BundleAdjusterBase> adjuster = cv::makePtr<cv::detail::BundleAdjusterRay>();
    (*adjuster)(features, matches_infos, cameras_params);

    for (auto & cameras_param : cameras_params)
        cameras_param.R.convertTo(cameras_param.R, CV_32F);

    std::vector<cv::Mat> wave_correct_R;
    for (auto &camera: cameras_params)
        wave_correct_R.push_back(camera.R);

    cv::detail::waveCorrect(wave_correct_R, cv::detail::WAVE_CORRECT_HORIZ);
    for (size_t i = 0; i < cameras_params.size(); ++i)
        cameras_params[i].R = wave_correct_R[i];


    std::vector<cv::Mat> masks(num_images);
    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(cv::Scalar::all(255));
    }

    std::vector<cv::Mat> masks_warp(num_images);
    std::vector<cv::Mat> images_warp(num_images);
    std::vector<cv::Point> corners(num_images);
    std::vector<cv::Size> sizes(num_images);
    cv::Ptr<cv::WarperCreator> warper_creator = cv::makePtr<cv::CylindricalWarper>();
    cv::Ptr<cv::detail::RotationWarper> warper = warper_creator->create(static_cast<float>(cameras_params[0].focal));
    for (int i = 0; i < num_images; ++i)
    {
        cv::Mat K;
        cameras_params[i].K().convertTo(K, CV_32F);
        corners[i] = warper->warp(images[i], K, cameras_params[i].R,
                                  cv::INTER_LINEAR, cv::BORDER_REFLECT,
                                  images_warp[i]);
        sizes[i] = images_warp[i].size();
        warper->warp(masks[i], K, cameras_params[i].R,
                     cv::INTER_NEAREST, cv::BORDER_CONSTANT,
                     masks_warp[i]);
    }
    cv::Ptr<cv::detail::Blender> blender = cv::detail::Blender::createDefault(cv::detail::Blender::NO, false);
    blender->prepare(corners, sizes);
    for (int i = 0; i < num_images; ++i)
    {
        images_warp[i].convertTo(images_warp[i], CV_16S);
        blender->feed(images_warp[i], masks_warp[i], corners[i]);
    }
    cv::Mat dst, dst_mask;
    blender->blend(dst, dst_mask);

    imwrite(result_filename, dst);
}



