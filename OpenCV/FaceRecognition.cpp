
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2\opencv.hpp>
#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <flandmark_detector.h>
#include <FaceAlignment.h>
#include <lbp.hpp>
#include <histogram.hpp>
#include <svm.h>

#include <dirent.h>

using namespace cv;
using namespace lbp;
using namespace std;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

// Face model
const String face_cascade_name = "haarcascade_frontalface_alt.xml";
FLANDMARK_Model * model = flandmark_init("flandmark_model.dat");

// Face Recognition Function
vector<Rect> faceDetection(Mat image);
double* landmarkDetection(Mat image,vector<Rect> faces);
//vector<Mat> landmarkDetection(Mat image,vector<Rect> faces);
Mat lbpFeature(Mat image);

inline bool islegalFileName(const char* tmpName);
inline bool isImgOutofBound(int x,int y,int bound);

int main(void){

	map<string,int> multiImgMap;			// preserve how many pictures per person
	map<double,string> testMapInfo;			// preserve person label
	stringstream sfile_name;

	// SVM Setup
	struct svm_parameter _param;
	struct svm_problem _prob;

	// SVM parameter setup
	_param.svm_type = C_SVC;
	_param.kernel_type = LINEAR;
	_param.degree = 3;
	_param.gamma = 0.00055;			// 1/num_features
	_param.coef0 = 0;
	_param.nu = 0.5;
	_param.cache_size = 100;
	_param.C = 1;
	_param.eps = 1e-3;
	_param.p = 0.1;
	_param.shrinking = 1;
	_param.probability = 0;
	_param.nr_weight = 0;
	_param.weight_label = NULL;
	_param.weight = NULL;

	// training sample number
	_prob.l = 7292;
	_prob.y = Malloc(double,_prob.l);
	svm_node **node = Malloc(svm_node*,_prob.l);

	int total_num = 0;	// training total number
	double label = 1;

	string lineString;
	fstream fileReader("name.txt");
	while(getline(fileReader,lineString)){
		istringstream isstream(lineString);
		string name;
		int value;
		isstream >> name >> value;
		multiImgMap.insert(make_pair(name,value));
	}

	DIR *mainDir = NULL;
	DIR *nameDir = NULL;
	struct dirent *pent = NULL;


	mainDir = opendir("temp");
	if(mainDir == NULL){
		cout << "Fail to read LFW dataset" << endl;
		exit(3);
	}

	while(pent = readdir(mainDir)){
		if(pent == NULL){
			cout << "Can't find name directory" << endl;
			exit(3);
		}

		if(islegalFileName(pent->d_name)){
			string _temp(pent->d_name);
			//cout << _temp << endl;
			map<string,int>::iterator it = multiImgMap.find(_temp);
			if(it== multiImgMap.end())
				continue;
			int dirImgNum = it->second;
			if(dirImgNum>2){
				testMapInfo.insert(make_pair(label,_temp));
				//cout << _temp << "	" << label << endl;
			}
			++label;
		}
	}
	/*
	int count = 1;

	sfile_name.str("");
	sfile_name.clear();
	sfile_name << "lfw" << "/" << pent->d_name;

	const string &temp = sfile_name.str();
	const char *imagePath = temp.c_str();

	nameDir = opendir(imagePath);
	while(pent = readdir(nameDir)){

	// Examine file name and control training number
	if(islegalFileName(pent->d_name) && dirImgNum >= count){
	if(dirImgNum>1 && dirImgNum == count)
	break;
	int feature_index = 0;
	svm_node *node_space = Malloc(svm_node,1793);

	sfile_name.str("");
	sfile_name.clear();
	sfile_name << temp << "/" << pent->d_name;
	cout << sfile_name.str() << endl;
	Mat image = imread(sfile_name.str(),0);
	if(image.empty())
	cout << "Image read fail";
	vector<Rect> faces = faceDetection(image);
	double *landmarks = landmarkDetection(image,faces);
	if(faces.size() == 0 || *landmarks < 0){
	for(int i=0;i<1792;i++){
	node_space[i].index = i;
	node_space[i].value = 0;
	}
	node_space[1792].index = -1;
	node[total_num] = node_space;
	_prob.y[total_num] = label;
	++total_num;
	++count;
	image.release();
	continue;
	}
	for(size_t landmark_element=2;landmark_element<15;landmark_element+=2){
	Mat lbpFeatureVector;
	Rect rect(static_cast<int>(landmarks[landmark_element]) - 20,static_cast<int>(landmarks[landmark_element+1]) - 20,40,40);
	if(isImgOutofBound(static_cast<int>(landmarks[landmark_element]),static_cast<int>(landmarks[landmark_element+1]),250)){
	for(int histogram_num = 0;histogram_num<256;histogram_num++){
	node_space[feature_index].index = feature_index;
	node_space[feature_index].value = 0;
	feature_index++;
	}
	continue;
	}
	//cout << landmarks[landmark_element] - 20 << "	" << landmarks[landmark_element+1]-20 << endl;
	Mat crop_ROI = image(rect);
	Mat lbpImage = lbpFeature(crop_ROI);
	histogram(lbpImage,lbpFeatureVector,256);
	for(int histogram_num = 0;histogram_num<256;histogram_num++){
	node_space[feature_index].index = feature_index;
	node_space[feature_index].value = lbpFeatureVector.at<int>(0,histogram_num);
	feature_index++;
	}
	lbpFeatureVector.release();
	crop_ROI.release();
	lbpImage.release();
	}
	node_space[1792].index = -1;
	node[total_num] = node_space; 
	_prob.y[total_num] = label;
	++total_num;
	++count;
	image.release();
	}
	}
	++label;
	}
	}

	_prob.x = node;
	svm_model *model = svm_train(&_prob,&_param);
	svm_save_model("model.txt",model);
	*/
	svm_model *model = svm_load_model("model.txt");

	double correct_num = 0;

	mainDir = opendir("temp");

	// LFW dataset directory - Human Name
	while(pent = readdir(mainDir)){
		if(pent == NULL){
			cout << "Can't find name directory" << endl;
			exit(3);
		}

		if(islegalFileName(pent->d_name)){
			string humanName(pent->d_name);
			map<string,int>::iterator it = multiImgMap.find(humanName);

			// Directory has only one picture => continue
			if(it== multiImgMap.end() || it->second == 1){
				continue;
			}

			// Record total file number in directory
			int dirImgNum = it->second;
			int count = 1;

			sfile_name.str("");
			sfile_name.clear();
			sfile_name << "temp" << "/" << pent->d_name;
			const string &temp = sfile_name.str();
			const char *imagePath = temp.c_str();

			nameDir = opendir(imagePath);
			while(pent = readdir(nameDir)){

				// Examine file name and control training number
				if(islegalFileName(pent->d_name)){
					//cout << dirImgNum << "	" << count << endl;
					if(dirImgNum != count){
						++count;
						continue;
					}
					//cout << "in" << endl;
					int feature_index = 0;
					svm_node *test_node = Malloc(svm_node,1793);

					sfile_name.str("");
					sfile_name.clear();
					sfile_name << temp << "/" << pent->d_name;
					cout << sfile_name.str() << endl;
					Mat image = imread(sfile_name.str(),0);

					if(image.empty())
						cout << "Image read fail";
					vector<Rect> faces = faceDetection(image);
					double *landmarks = landmarkDetection(image,faces);
					if(faces.size() == 0 || *landmarks < 0){
						for(int i=0;i<1792;i++){
							test_node[i].index = i;
							test_node[i].value = 0;
						}
						test_node[1792].index = -1;
						double retval = svm_predict(model,test_node);
						map<double,string>::iterator it = testMapInfo.find(retval);
						if(it == testMapInfo.end()){
							cout << "Find nothing" << endl;
							break;
						}
						cout << it->second << endl;
						if(strcmp(it->second.c_str(),humanName.c_str())!=0){
							++correct_num;
						}
						image.release();
						continue;
					}
					for(size_t landmark_element=2;landmark_element<15;landmark_element+=2){
						Mat lbpFeatureVector;
						Rect rect(static_cast<int>(landmarks[landmark_element]) - 20,static_cast<int>(landmarks[landmark_element+1]) - 20,40,40);
						//cout << landmarks[landmark_element] - 20 << "	" << landmarks[landmark_element+1]-20 << endl;
						if(isImgOutofBound(static_cast<int>(landmarks[landmark_element]),static_cast<int>(landmarks[landmark_element+1]),250)){
							for(int histogram_num = 0;histogram_num<256;histogram_num++){
								test_node[feature_index].index = feature_index;
								test_node[feature_index].value = 0;
								feature_index++;
							}
							continue;
						}
						Mat crop_ROI = image(rect);
						Mat lbpImage = lbpFeature(crop_ROI);
						histogram(lbpImage,lbpFeatureVector,256);
						for(int histogram_num = 0;histogram_num<256;histogram_num++){
							test_node[feature_index].index = feature_index;
							test_node[feature_index].value = lbpFeatureVector.at<int>(0,histogram_num);
							feature_index++;
						}
						lbpFeatureVector.release();
						crop_ROI.release();
						lbpImage.release();
					}
					test_node[1792].index = -1;
					double retval = svm_predict(model,test_node);
					map<double,string>::iterator it = testMapInfo.find(retval);
					if(it == testMapInfo.end()){
						cout << "Find nothing" << endl;
						break;
					}
					cout << it->second << endl;
					if(strcmp(it->second.c_str(),humanName.c_str())!=0){
						++correct_num;
					}
					image.release();
				}
			}
		}
	}

	cout << correct_num << endl;

	/*
	Mat image = imread("AOA.png",0);
	double *landmarks = landmarkDetection(image,faceDetection(image));
	if(*landmarks<0)
	cout << "fail" << endl;
	*/

	waitKey(0);
	system("pause");
	return 0;
}

// Face Detection: return face bound
vector<Rect> faceDetection(Mat image){

	CascadeClassifier face_cascade;
	vector<Rect> faces;

	if(!face_cascade.load(face_cascade_name)){
		printf("Error Loading");
	}

	face_cascade.detectMultiScale(image,faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
	/*
	Point leftUpCorner(faces[0].x,faces[0].y);
	Point rightDownCorner(faces[0].x+faces[0].width,faces[0].y+faces[0].height);
	rectangle(image,leftUpCorner,rightDownCorner,Scalar(255,0,255),4,8,0);
	imshow("Hello",image);
	*/
	return faces;
}

// flandmark version
double* landmarkDetection(Mat image,vector<Rect> faces){
	IplImage *img_grayscale = &IplImage(image);
	//IplImage *img_grayscale = cvCreateImage(cvSize(i_image->width, i_image->height), IPL_DEPTH_8U, 1);
	//cvCvtColor(i_image,img_grayscale,CV_BGR2GRAY);
	double *landmarks = (double*)malloc(2*model->data.options.M*sizeof(double));
	for(size_t faces_num = 0;faces_num<faces.size();faces_num++){
		int bbox[] = {faces[faces_num].x,faces[faces_num].y,faces[faces_num].x+faces[faces_num].width,faces[faces_num].y+faces[faces_num].height};
		flandmark_detect(img_grayscale,bbox,model,landmarks);
		//cvRectangle(i_image, cvPoint(bbox[0], bbox[1]), cvPoint(bbox[2], bbox[3]), CV_RGB(255,0,0) );
		//cvRectangle(i_image, cvPoint(model->bb[0], model->bb[1]), cvPoint(model->bb[2], model->bb[3]), CV_RGB(0,0,255) );
		//cvCircle(img_grayscale, cvPoint((int)landmarks[0], (int)landmarks[1]), 3, CV_RGB(0, 0,255), CV_FILLED);
		/*
		for (int i = 2; i < 2*model->data.options.M; i += 2)
		{
		cvCircle(img_grayscale, cvPoint(int(landmarks[i]), int(landmarks[i+1])), 3, CV_RGB(255,0,0), CV_FILLED);
		}
		*/
	}
	//cvShowImage("Hello",img_grayscale);
	return landmarks;
}

/*
vector<Mat> landmarkDetection(Mat image,vector<Rect> faces){

// Face Landmark detection pre-setup
//ShapeRegressor regressor;
//regressor.Load("model.txt");
vector<BoundingBox> test_bounding_box;
vector<Mat> landmark;

for(size_t i=0;i<faces.size();i++){
// Face Detection visualization

Point leftUpCorner(faces[i].x,faces[i].y);
Point rightDownCorner(faces[i].x+faces[i].width,faces[i].y+faces[i].height);
rectangle(image,leftUpCorner,rightDownCorner,Scalar(255,0,255),4,8,0);


// Face Detection Bound
BoundingBox temp;
temp.start_x = faces[i].x;
temp.start_y = faces[i].y;
temp.width = faces[i].width;
temp.height = faces[i].height;
temp.centroid_x = faces[i].x + faces[i].width/2;
temp.centroid_y = faces[i].y + faces[i].height/2;
test_bounding_box.push_back(temp);

Mat_<int> current_shape = regressor.Predict(image,test_bounding_box[i],20);
landmark.push_back(current_shape);
// Face image landmark visualization

for(int j = 0;j < 29;j++){
cout << current_shape(j,0) << "		" << current_shape(j,1) << endl;
circle(image,Point2d(current_shape(j,0),current_shape(j,1)),3,Scalar(255,0,0),-1,8,0);
}
imshow("Face",image);
}
return landmark;
//imshow(window_name,image);
}
*/
Mat lbpFeature(Mat image){
	Mat dst = image;
	//GaussianBlur(image,dst,Size(7,7),5,3,BORDER_CONSTANT);
	Mat lbpImage;
	ELBP(dst,lbpImage,1,8);
	return lbpImage;
}

// Examine directory file not inlcude . ..
inline bool islegalFileName(const char* tmpName){
	if(strcmp(tmpName,".")!=0 && strcmp(tmpName,"..")!=0)
		return true;
	else
		return false;
}

inline bool isImgOutofBound(int x,int y,int bound){
	if(x-20<0 || x+40>bound || y-20<0 || y+40 >bound)
		return true;
	else
		return false;
}

