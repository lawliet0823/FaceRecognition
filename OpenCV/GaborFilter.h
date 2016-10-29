#include <opencv2\opencv.hpp>
#include <vector>

using namespace cv;

class GaborFilter {

public:
	GaborFilter();
	static Mat getImageGaborKernel(Size ksize, double sigma, double theta, double nu, double gamma = 1, int ktype = CV_32F);
	static Mat getRealGaborKernel(Size ksize, double sigma, double theta, double nu, double gamma = 1, int ktype = CV_32F);
	static Mat getPhase(Mat &real, Mat &img);
	static Mat getMagnitude(Mat &real, Mat &img);
	static Mat getFilterRealPart(Mat &src, Mat &real);
	static Mat getFilterImagePart(Mat &src, Mat &img);
	static void getFilterRealImagePart(Mat &src, Mat &real, Mat &img, Mat &outReal, Mat &outImg);

	void Init(Size ksize = Size(19, 19), double sigma = 2 * CV_PI, double gamma = 1, int ktype = CV_32FC1);

private:
	vector<Mat> gaborRealKernels;
	vector<Mat> gaborImageKernels;
	bool isInited;
};