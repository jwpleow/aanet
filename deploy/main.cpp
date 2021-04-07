#include <torch/script.h>
#include <torch/nn/functional.h>
#include <torch/torch.h>

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
    std::cout << "Dimensions: " << tens.dim() << "\n"
              << "Datatype: " << tens.dtype() << "\n"
              << "Device: " << tens.device() << "\n"
              << "Size: " << tens.sizes() << "\n"
              << "Number of Elements: " << tens.numel() << "\n";
}

/*
    output is a torch::Tensor of size [1, C, H, W]
*/
void ProcessImage(const cv::Mat& input, torch::Tensor& output, torch::Device& device)
{
    // if using an RGB image make sure to remember to transpose/cvtColor to RGB 

    // convert to torch tensor
    auto temp = torch::zeros({input.rows, input.cols, input.channels()});
    temp = torch::from_blob(input.data, {1, input.rows, input.cols, 3}, at::kByte).clone();
    temp = temp.to(at::kFloat);
    temp = temp.permute({0, 3, 1, 2}); // rearrange to BxCxHxW

    // normalize image
    // IMAGENET_MEAN = [0.485, 0.456, 0.406]
    // IMAGENET_STD = [0.229, 0.224, 0.225]
    temp = temp.div(255.0);
    temp[0][0] = temp[0][0].sub(0.485).div(0.229);
    temp[0][1] = temp[0][1].sub(0.456).div(0.224);
    temp[0][2] = temp[0][2].sub(0.406).div(0.225);

    // pad image to correct size
    static int ori_height = input.rows;
    static int ori_width = input.cols;
    static float factor = 96.0f;
    static int img_height = static_cast<int>(std::ceil(static_cast<float>(ori_height) / factor) * factor);
    static int img_width = static_cast<int>(std::ceil(static_cast<float>(ori_width) / factor) * factor);
    
    output = torch::zeros({1, input.channels(), img_height, img_width});
    if (ori_height < img_height || ori_width < img_width)
    {
        static int top_pad = img_height - ori_height;
        static int right_pad = img_width - ori_width;
        output = torch::nn::functional::pad(temp, torch::nn::functional::PadFuncOptions({0, right_pad, top_pad, 0}));
    }

    output = output.to(device);
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
    cv::imshow("Left", left);
    cv::imshow("Right", right);

    torch::Tensor leftT, rightT;
    ProcessImage(left, leftT, device);
    ProcessImage(right, rightT, device);
    
    std::cout << "Left image input properties: \n";
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
    disparityVis = disparity / 200.; // divided by ~200 since maxDisp is 192 and the actual output is up to ~197

    cv::imshow("Disparity", disparityVis);

    cv::waitKey(100000);
}