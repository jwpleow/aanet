#include <torch/script.h>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <memory>
#include <string>


// test model

void ProcessImage(cv::Mat input, torch::Tensor output, torch::Device device)
{
    cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
    input.convertTo(input, CV_32FC3, 1./255.0);
    // pad image to correct size
    cv::Mat padded = cv::Mat::zeros(384, 672, CV_32FC3);
    input.copyTo(padded(cv::Rect(0, 0, input.cols, input.rows)));
    cv::imshow("padded", padded);

    // convert to torch tensor
    output = torch::zeros({padded.rows, padded.cols, padded.channels()});
    output = torch::from_blob(padded.data, {1, 3, padded.rows, padded.cols}, at::kFloat).clone().to(device);

    // normalize image
    // IMAGENET_MEAN = [0.485, 0.456, 0.406]
    // IMAGENET_STD = [0.229, 0.224, 0.225]
    output[0][0] = output[0][0].sub(0.485).div(0.229); // not exactly right?
    output[0][1] = output[0][1].sub(0.456).div(0.224);
    output[0][2] = output[0][2].sub(0.406).div(0.225);
    // std::cout << output << "\n";
    // at::transpose(output, 1, 2);
    // at::transpose(output, 1, 3);
}

int main(int argc, const char *argv[])
{
    torch::Device device(torch::kCUDA);
    std::cout << "Hi!\n";
    if (argc != 2)
    {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module model;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        model = torch::jit::load(argv[1]);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
        return -1;
    }
    model.to(device);

    cv::Mat left = cv::imread("../leftRectified.png");
    cv::Mat right = cv::imread("../rightRectified.png");
    if (left.empty() || right.empty())
    {
        std::cout << "imread failed!\n";
    }

    std::cout << "size of left image: " << left.size() << "\n";
    std::cout << "size of right image: " << right.size() << "\n";
    cv::imshow("Left", left);
    cv::imshow("Right", right);


    torch::Tensor leftT, rightT;
    ProcessImage(left, leftT, device);
    ProcessImage(right, rightT, device);

    std::vector<torch::jit::IValue> input;
    // input.push_back(leftT);
    // input.push_back(rightT);
    auto lef = torch::ones({1, 3, 384, 672}).to(device);
    auto righ = torch::ones({1, 3, 384, 672}).to(device);
    input.push_back(lef);
    input.push_back(righ);

    auto output = model.forward(input);
    std::cout << output << "\n";

    cv::waitKey(100000);
}