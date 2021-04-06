#include <torch/script.h>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

#include <iostream>
#include <memory>
#include <string>


// test model
void printTensorProperties(const torch::Tensor &tens)
{
    std::cout << "Tensor Properties\n"
              << "Dimensions: " << tens.dim() << "\n"
              << "Datatype: " << tens.dtype() << "\n"
              << "Device: " << tens.device() << "\n"
              << "Size: " << tens.sizes() << "\n"
              << "Number of Elements: " << tens.numel() << "\n";
}

void ProcessImage(const cv::Mat& input, torch::Tensor& output, torch::Device& device)
{
    cv::Mat temp;
    cv::cvtColor(input, temp, cv::COLOR_BGR2RGB);
    temp.convertTo(temp, CV_32FC3, 1. / 255.0);
    // pad image to correct size
    cv::Mat padded = cv::Mat::zeros(384, 672, CV_32FC3);
    temp.copyTo(padded(cv::Rect(0, 0, temp.cols, temp.rows)));
    cv::imshow("padded", padded);

    // convert to torch tensor
    output = torch::zeros({padded.rows, padded.cols, padded.channels()});
    output = torch::from_blob(padded.data, {1, 3, padded.rows, padded.cols}, at::kFloat).clone();
    output = output.to(device);

    // normalize image
    // IMAGENET_MEAN = [0.485, 0.456, 0.406]
    // IMAGENET_STD = [0.229, 0.224, 0.225]
    output[0][0] = output[0][0].sub(0.485).div(0.229);
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

    // std::cout << "size of left image: " << left.size() << "\n";
    // std::cout << "size of right image: " << right.size() << "\n";
    // cv::imshow("Left", left);
    // cv::imshow("Right", right);


    torch::Tensor leftT, rightT;
    ProcessImage(left, leftT, device);
    ProcessImage(right, rightT, device);

    printTensorProperties(leftT);
    std::vector<torch::jit::IValue> input;
    input.push_back(leftT);
    input.push_back(rightT);
    
    // predict
    auto output = model.forward(input).toTensor().cpu().squeeze();
    printTensorProperties(output);

    // convert tensor to cv::Mat. remember that torch is CxHxW while openCV is HxWxC
    cv::Mat disparity_output = cv::Mat::zeros(384, 672, CV_32F);
    std::memcpy(disparity_output.data, output.data_ptr(), sizeof(float) * output.numel());

    // crop to correct size
    auto crop_size = cv::Rect(0, 0, left.cols, left.rows);
    cv::Mat disparity = disparity_output(crop_size);

    // visualise
    cv::Mat disparityVis;
    disparityVis = disparity / 200.;


    // check minmax vals
    double minVal, maxVal;
    cv::minMaxLoc(disparityVis, &minVal, &maxVal);
    std::cout << "min max " << minVal << " " << maxVal << "\n";

    cv::imshow("Disparity", disparityVis);

    cv::waitKey(100000);
}