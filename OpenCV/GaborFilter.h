#include <opencv2\opencv.hpp>
#include <ccomplex>
#include <vector>

using namespace cv;

class GaborFilter{

public:
	GaborFilter(){}
	void GaborFilterBank(int u,int v,int m,int n);
	void GaborFeature(Mat image,vector<Mat> gaborArray,int u,int v,int d1,int d2);

private:
	float fmax_;
	float gama_;
	float eta_;
	vector<Mat> gaborArray_;
	Complex<float> imaginaryPart_;
};