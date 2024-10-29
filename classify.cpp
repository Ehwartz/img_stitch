//
// Created by Ehwartz on 03/16/2024.
//
#include "classify.h"

Node::Node(int idx)
{
    this->idx = idx;
}

void Node::connect(Node *dst)
{
    bool connected_to_dst = false;
    for (Node *neighbor: this->neighbors)
    {
        if (neighbor->idx == this->idx)
        {
            connected_to_dst = true;
            break;
        }
    }
    if (connected_to_dst)
        return;
    else
    {
        this->neighbors.emplace_back(dst);
        this->connected = true;
        dst->neighbors.emplace_back(this);
        dst->connected = true;
    }
}


Graph::Graph(int n, const std::vector<cv::detail::MatchesInfo> &matches_infos)
{
    this->n = n;
    for (int i = 0; i < n; ++i)
    {
        Node *node_ptr = new Node(i);
        this->nodes.emplace_back(node_ptr);
    }
    this->A = cv::Mat::zeros(this->n, this->n, CV_64FC(1));
    for (const cv::detail::MatchesInfo &match_info: matches_infos)
    {
        if (match_info.src_img_idx == -1)
            continue;
        else
            this->A.at<double>(match_info.src_img_idx, match_info.dst_img_idx) = match_info.confidence;
    }
}

Graph::~Graph()
{
    for (auto &node: this->nodes)
    {
        delete node;
    }
}

void Graph::clustering(double threshold)
{
    this->clusters.clear();
    for (Node *node: this->nodes)
    {
        node->connected = false;
        node->neighbors.clear();
    }
    for (int i = 0; i < this->n; ++i)
    {
        for (int j = i + 1; j < this->n; ++j)
        {
            if (this->A.at<double>(i, j) > threshold)
            {
                this->nodes[i]->connect(this->nodes[j]);
            }
        }
    }
    std::vector<Node *> queue;
    for (auto node: this->nodes)
    {
        queue.push_back(node);
    }
    while (!queue.empty())
    {
        std::vector<Node *> cluster;
        Node *ptr = queue[0];
        queue.erase(queue.begin());
        cluster.emplace_back(ptr);
        this->traverse(ptr, cluster, queue);
        this->clusters.emplace_back(cluster);
    }
}

void Graph::traverse(Node *ptr, std::vector<Node *> &cluster, std::vector<Node *> &queue)
{
    for (auto neighbor: ptr->neighbors)
    {
        int idx = find_node(neighbor, queue);
        if (idx == -1)
            continue;
        else
        {
            queue.erase(queue.begin() + idx);
            cluster.emplace_back(neighbor);
            this->traverse(neighbor, cluster, queue);
        }
    }
}

void Graph::cout_clusters()
{
    for (int i = 0; i < this->clusters.size(); ++i)
    {
        for (int j = 0; j < this->clusters[i].size(); ++j)
        {
            std::cout << std::format("cluster:{}\tnode:{}\t", i, j) << this->clusters[i][j] << "\t"
                      << this->clusters[i][j]->idx << std::endl;
        }
    }
}

int find_node(Node *node, std::vector<Node *> nodes)
{
    for (int i = 0; i < nodes.size(); ++i)
    {
        if (node == nodes[i])
            return i;
    }
    return -1;
}

std::vector<std::tuple<std::vector<cv::Mat>,
                       std::vector<cv::detail::ImageFeatures>,
                       std::vector<cv::detail::MatchesInfo>>>
panorama_classify(std::vector<cv::Mat> images, double threshold)
{
    int num_images = (int) images.size();
    std::vector<cv::detail::ImageFeatures> features(num_images);
    std::vector<cv::Size> images_sizes(num_images);

    cv::Ptr<cv::Feature2D> features_finder = cv::SIFT::create();
    for (int i = 0; i < num_images; ++i)
    {
        images_sizes[i] = images[i].size();
        cv::detail::computeImageFeatures(features_finder, images[i], features[i]);
    }
    cv::Ptr<cv::Formatter> formatter=cv::Formatter::get(cv::Formatter::FMT_DEFAULT);
    formatter->set64fPrecision(3);
    formatter->set32fPrecision(3);
    std::vector<cv::detail::MatchesInfo> matches_infos;
    cv::Ptr<cv::detail::FeaturesMatcher> matcher = cv::makePtr<cv::detail::BestOf2NearestMatcher>(false, 0.3f, 6, 6);
    (*matcher)(features, matches_infos);
//    for (auto& matches_info: matches_infos)
//    {
//        std::cout<<std::endl;
//    }

    Graph graph(num_images, matches_infos);
    std::cout << formatter->format(graph.A)<< std::endl;
    graph.clustering(threshold);

    graph.cout_clusters();

    int n_panoramas = (int) graph.clusters.size();
    std::vector<std::tuple<std::vector<cv::Mat>,
                           std::vector<cv::detail::ImageFeatures>,
                           std::vector<cv::detail::MatchesInfo>>>
                           panoramas(n_panoramas);

//    std::vector<std::vector<cv::Mat>> panoramas(n_panoramas);
//    for (auto match_infos : matches_infos)
//    {
//        std::cout<<match_infos.src_img_idx<<"   "<<match_infos.dst_img_idx<<std::endl;
//    }

    for (int i = 0; i < n_panoramas; ++i)
    {
        std::vector<int> indices;
        for (auto node: graph.clusters[i])
        {
            indices.push_back(node->idx);
        }
        int n_images = (int) graph.clusters[i].size();
        std::vector<cv::Mat> panorama_images(n_images);
        std::vector<cv::detail::ImageFeatures> panorama_features(n_images);
        std::vector<cv::detail::MatchesInfo> panorama_matches_infos(n_images * n_images);
        for (int img_idx=0; img_idx< n_images; ++img_idx)
        {
            panorama_images[img_idx] = images[indices[img_idx]];
            panorama_features[img_idx] = features[indices[img_idx]];
        }

        for (int src_idx = 0; src_idx < n_images; ++src_idx)
        {
            for (int dst_idx = 0; dst_idx < n_images; ++dst_idx)
            {
                int idx = indices[src_idx] * num_images + indices[dst_idx];
                cv::detail::MatchesInfo match_infos = matches_infos[idx];
                if (src_idx == dst_idx)
                {
                    match_infos.src_img_idx = -1;
                    match_infos.dst_img_idx = -1;
                }
                else
                {
                    match_infos.src_img_idx = src_idx;
                    match_infos.dst_img_idx = dst_idx;
                }
                panorama_matches_infos[src_idx * n_images + dst_idx]=match_infos;
            }
        }
        panoramas[i]=std::make_tuple(panorama_images, panorama_features, panorama_matches_infos) ;
    }
    return panoramas;
}



