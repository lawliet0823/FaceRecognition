#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <opencv\cv.hpp>
#include <opencv\highgui.h>
#include <opencv\cxcore.h>

#include <flandmark_detector.h>
#include <FaceAlignment.h>
#include <lbp.hpp>
#include <histogram.hpp>
#include <svm.h>

#include <pca.h>
#include <random>


using namespace cv;
using namespace lbp;
using namespace std;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define min(a,b) (a) > (b) ? b : a

// Face model
const String face_cascade_name = "haarcascade_frontalface_alt_tree.xml";
FLANDMARK_Model *landmarkModel = flandmark_init("flandmark_model.dat");

map<string, int> readFile(string fileName);
Rect getCropImageBound(double center_x, double center_y, int crop_size);

double getSkewness(double *landmarks);
Mat rotateImage(Mat src, double angle);
double *landmarkRotation(double *landmarks, double angle, double midX, double midY);

// Face Recognition Function
vector<Rect> faceDetection(Mat image);
double* landmarkDetection(Mat image, Rect rect_face);
Mat lbpFeature(Mat image);

inline bool isImgOutofBound(int x, int y, int crop_size, int img_width, int img_height);

vector<Rect> faces;
vector<Rect> faces_t;

int main(void) {

	const int feature_total_num = 3 * 7 * 16 * 59 + 1;
	const int cell_size = 36;

	// SVM Setup
	struct svm_parameter _param;
	struct svm_problem _prob;

	// SVM parameter setup
	_param.svm_type = C_SVC;
	_param.kernel_type = LINEAR;
	_param.degree = 3;
	_param.gamma = 0.0001;			// 1/num_features
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
	_prob.l = 20;
	_prob.y = Malloc(double, _prob.l);
	svm_node **node = Malloc(svm_node*, _prob.l);

	map<string, int> trainMap;
	map<string, int> testMap;
	trainMap = readFile("Train.txt");
	testMap = readFile("Test.txt");

	char filename1[] = "FaceLandmarkFail.txt";
	char filename2[] = "BoundFail.txt";
	char outputFile[] = "negative.txt";
	fstream fp1, fp2, nfp;


	fp1.open(filename1, ios::out);
	if (!fp1) {
		cout << "Fail to open file: " << filename1 << endl;
	}
	fp2.open(filename2, ios::out);
	if (!fp2) {
		cout << "Fail to open file: " << filename2 << endl;
	}
	nfp.open(outputFile, ios::out);
	if (!nfp) {
		cout << "Fail to open file: " << outputFile << endl;
	}

	//regressor.Load("landmark_model_new.txt");
	int parameter_choice = 1;

	if (parameter_choice == 0) {

		// count training number
		int train_num_count = 0;

		map<string, int>::iterator it = trainMap.begin();
		for (; it != trainMap.end(); it++) {
			string file_Location = it->first;
			int label = it->second;
			cout << file_Location << "	" << label << endl;
			Mat image = imread(file_Location, 0);
			//equalizeHist(image, image);

			if (image.empty()) {
				cout << file_Location << " Image read fail" << endl;
			}

			int feature_index = 0;
			svm_node *node_space = Malloc(svm_node, feature_total_num);

			// Face Detection
			faces.swap(vector<Rect>());
			faces = faceDetection(image);
			if (faces.size() == 0) {
				//fp1 << file_Location << "	:face" << endl;
				cout << "Face Detection 1 Error!!!" << endl;
				for (int i = 0; i < feature_total_num - 1; i++) {
					node_space[i].index = i;
					node_space[i].value = 0;
				}
				node_space[feature_total_num - 1].index = -1;
				node[train_num_count] = node_space;
				_prob.y[train_num_count] = label;
				++train_num_count;
				continue;
			}

			image = image(Rect(faces.at(0).x - 20, faces.at(0).y - 20, faces.at(0).width + 40, faces.at(0).height + 40));
			resize(image, image, Size(200, 200));


			faces_t.swap(vector<Rect>());
			faces_t = faceDetection(image);
			if (faces_t.size() == 0) {
				cout << "Face Detection 2 Error!!!" << endl;
				for (int i = 0; i < feature_total_num - 1; i++) {
					node_space[i].index = i;
					node_space[i].value = 0;
				}
				node_space[feature_total_num - 1].index = -1;
				node[train_num_count] = node_space;
				_prob.y[train_num_count] = label;
				++train_num_count;
				continue;
			}

			double *landmarks = landmarkDetection(image, faces_t.at(0));
			if (*landmarks < 0) {
				cout << "landmark detection error" << endl;
				//fp1 << file_Location << "	:mark" << endl;
				for (int i = 0; i < feature_total_num - 1; i++) {
					node_space[i].index = i;
					node_space[i].value = 0;
				}
				node_space[feature_total_num - 1].index = -1;
				node[train_num_count] = node_space;
				_prob.y[train_num_count] = label;
				++train_num_count;
				continue;
			}
			else {
				image = rotateImage(image, getSkewness(landmarks));
				landmarks = landmarkRotation(landmarks, -getSkewness(landmarks), image.cols / 2, image.rows / 2);
				//landmarks = landmarkRotation(landmarks, getSkewness(landmarks), image.cols, image.rows);
				// scale image
				for (size_t scale_num = 0; scale_num < 3; scale_num++) {
					Mat scaleImage;
					resize(image, scaleImage, Size(), 1 - 0.2*scale_num, 1 - 0.2*scale_num, 1);
					// can't find faces and landmarks

					for (size_t landmark_element = 2; landmark_element < 15; landmark_element += 2) {
						// crop image is out of bound
						if (isImgOutofBound(static_cast<int>(landmarks[landmark_element] * (1 - 0.2*scale_num)),
							static_cast<int>(landmarks[landmark_element + 1] * (1 - 0.2*scale_num)), cell_size, scaleImage.cols, scaleImage.rows)) {
							fp2 << file_Location << "	" << scale_num << endl;
							for (int histogram_num = 0; histogram_num < 16 * 59; histogram_num++) {
								node_space[feature_index].index = feature_index;
								node_space[feature_index].value = 0;
								++feature_index;
							}
							continue;
						}
						//cout << landmarks[landmark_element] - 20 << "	" << landmarks[landmark_element+1]-20 << endl;
						else {
							Mat calLBPFeature;

							Rect cropRect = getCropImageBound(landmarks[landmark_element] * (1 - 0.2*scale_num),
								landmarks[landmark_element + 1] * (1 - 0.2*scale_num), cell_size);
							Mat crop_Img = scaleImage(cropRect);

							int sample_count = 0;
							for (size_t cell_y = 0; cell_y < cell_size; cell_y += 9) {
								for (size_t cell_x = 0; cell_x < cell_size; cell_x += 9) {
									Mat lbpFeatureVector;	// Feature Vector
									Rect cellRect(cell_x, cell_y, 9, 9);
									Mat crop_cell = crop_Img(cellRect);

									Mat lbpImage = lbpFeature(crop_cell);
									//histogram(lbpImage, lbpFeatureVector, 256);
									uni_histogram(lbpImage, lbpFeatureVector);
									for (int histogram_num = 0; histogram_num < 59; histogram_num++) {
										//mat_pca.at<int>(sample_count, histogram_num) = lbpFeatureVector.at<int>(0, histogram_num);
										//cout << lbpFeatureVector.at<int>(0, histogram_num) << "		";

										node_space[feature_index].index = feature_index;
										node_space[feature_index].value = lbpFeatureVector.at<int>(0, histogram_num);  // (Scale)
										++feature_index;

										//cout << histogram_num << "	" << lbpFeatureVector.at<int>(0, histogram_num) << endl;
									}
									//cout << endl;
									//lbpFeatureVector.release();
									//crop_cell.release();
									//lbpImage.release();
								}
							}
							/*
							PCA pca(mat_pca, Mat(), CV_PCA_DATA_AS_ROW);
							cout << pca.eigenvectors.size().width << "		" << pca.eigenvectors.size().height << endl;
							for (size_t i = 0; i < pca.eigenvectors.size().width; i++) {
							cout << pca.eigenvectors.at<float>(13, i) << "	";
							}
							cout << endl;
							*/
							//crop_Img.release();
						}
						//Rect rect(static_cast<int>(landmarks[landmark_element]) - 20, static_cast<int>(landmarks[landmark_element + 1]) - 20, 40, 40);
					}
					//scaleImage.release();
				}
				node_space[feature_total_num - 1].index = -1;
				node[train_num_count] = node_space;
				_prob.y[train_num_count] = label;
				++train_num_count;
				/*
				for (int element = 0; element < feature_total_num; element++) {
					nfp << node_space[element].value << "	";
				}
				nfp << endl;
				*/
			}
			//image.release();
		}
		fp1.close();
		fp2.close();
		//nfp.close();
		_prob.x = node;
		svm_model *model = svm_train(&_prob, &_param);
		svm_save_model("model.txt", model);
	}


	if (parameter_choice == 1) {
		svm_model *model = svm_load_model("model.txt");	// load svm model
		if (model == NULL) {
			cout << "Can't load model" << endl;
		}
		cout << "finish loading" << endl;

		double correct_num = 0;
		int count = 1;
		double total_num = 0;
		map<string, int>::iterator it = testMap.begin();
		for (; it != testMap.end(); it++) {
			string file_Location = it->first;
			int label = it->second;
			cout << file_Location << "	" << label << endl;
			Mat image = imread(file_Location, 0);
			if (label == count) {
				total_num++;
				if (image.empty()) {
					cout << "Image read fail" << endl;
				}
				//equalizeHist(image, image);

				int feature_index = 0;
				svm_node *node_space = Malloc(svm_node, feature_total_num);
				faces.swap(vector<Rect>());
				faces = faceDetection(image);
				if (faces.size() == 0) {
					cout << "Face Detection 1 Error!!!" << endl;
					//fp1 << file_Location << "	:face" << endl;
					for (int i = 0; i < feature_total_num - 1; i++) {
						node_space[i].index = i;
						node_space[i].value = 0;
					}
					node_space[feature_total_num - 1].index = -1;
					double retval = svm_predict(model, node_space);
					if (retval == 1) {
						correct_num++;
					}
					continue;
				}

				if (faces.at(0).x - 20 < 0 || faces.at(0).y - 20 < 0 || (faces.at(0).x + faces.at(0).width) + 40 > image.cols || (faces.at(0).y + faces.at(0).height) + 40 > image.rows) {
					cout << "Out of bound" << endl;
					continue;
				}
				image = image(Rect(faces.at(0).x - 20, faces.at(0).y - 20, faces.at(0).width + 40, faces.at(0).height + 40));
				resize(image, image, Size(200, 200));

				faces_t.swap(vector<Rect>());
				faces_t = faceDetection(image);
				if (faces_t.size() == 0) {
					cout << "Face Detection 2 Error!!!" << endl;
					for (int i = 0; i < feature_total_num - 1; i++) {
						node_space[i].index = i;
						node_space[i].value = 0;
					}
					node_space[feature_total_num - 1].index = -1;
					double retval = svm_predict(model, node_space);
					if (retval == 1) {
						correct_num++;
					}
					continue;
				}

				double *landmarks = landmarkDetection(image, faces_t[0]);
				if (*landmarks < 0) {
					fp1 << file_Location << "	:mark" << endl;
					for (int i = 0; i < feature_total_num - 1; i++) {
						node_space[i].index = i;
						node_space[i].value = 0;
					}
					node_space[feature_total_num - 1].index = -1;
					double retval = svm_predict(model, node_space);
					if (retval == 1) {
						correct_num++;
					}
					cout << "landmark detection error" << endl;
					continue;
				}
				else {
					image = rotateImage(image, getSkewness(landmarks));
					landmarks = landmarkRotation(landmarks, -getSkewness(landmarks), image.cols / 2, image.rows / 2);
					// scale image
					for (size_t scale_num = 0; scale_num < 3; scale_num++) {
						Mat scaleImage;
						resize(image, scaleImage, Size(), 1 - 0.2*scale_num, 1 - 0.2*scale_num, 1);
						// can't find faces and landmarks
						for (size_t landmark_element = 2; landmark_element < 15; landmark_element += 2) {
							// crop image is out of bound
							if (isImgOutofBound(static_cast<int>(landmarks[landmark_element] * (1 - 0.2*scale_num)),
								static_cast<int>(landmarks[landmark_element + 1] * (1 - 0.2*scale_num)), cell_size, scaleImage.cols, scaleImage.rows)) {
								fp2 << file_Location << "	" << scale_num << endl;
								for (int histogram_num = 0; histogram_num < 16 * 59; histogram_num++) {
									node_space[feature_index].index = feature_index;
									node_space[feature_index].value = 0;
									++feature_index;
								}
								continue;
							}
							//cout << landmarks[landmark_element] - 20 << "	" << landmarks[landmark_element+1]-20 << endl;
							else {
								Rect cropRect = getCropImageBound(landmarks[landmark_element] * (1 - 0.2*scale_num),
									landmarks[landmark_element + 1] * (1 - 0.2*scale_num), cell_size);
								Mat crop_Img = scaleImage(cropRect);
								for (size_t cell_y = 0; cell_y < cell_size; cell_y += 9) {
									for (size_t cell_x = 0; cell_x < cell_size; cell_x += 9) {
										Mat lbpFeatureVector;
										Rect cellRect(cell_x, cell_y, 9, 9);
										Mat crop_cell = crop_Img(cellRect);
										Mat lbpImage = lbpFeature(crop_cell);
										//histogram(lbpImage, lbpFeatureVector, 256);
										uni_histogram(lbpImage, lbpFeatureVector);
										for (int histogram_num = 0; histogram_num < 59; histogram_num++) {
											node_space[feature_index].index = feature_index;
											node_space[feature_index].value = lbpFeatureVector.at<int>(0, histogram_num);  // (Scale)
											++feature_index;
										}
										//lbpFeatureVector.release();
										//crop_cell.release();
										//lbpImage.release();
									}
								}
								crop_Img.release();
							}
							//Rect rect(static_cast<int>(landmarks[landmark_element]) - 20, static_cast<int>(landmarks[landmark_element + 1]) - 20, 40, 40);
						}
						//scaleImage.release();

					}
					node_space[feature_total_num - 1].index = -1;
					double retval = svm_predict(model, node_space);
					if (retval == 1) {
						correct_num++;
					}
				}
				//image.release();
			}
			else {
				if (correct_num >= 3) {
					cout << "Find!!!" << endl;
					break;
				}
				correct_num = 0;
				total_num = 0;
				count++;
				it--;
			}
		}
		fp1.close();
		fp2.close();
		cout << correct_num << endl;
	}

	system("pause");
	return 0;
}

map<string, int> readFile(string fileName) {
	map<string, int> fileMap;
	string lineString;
	fstream fileReader(fileName);
	while (getline(fileReader, lineString)) {
		istringstream isstream(lineString);
		string file_Location;
		int label;
		isstream >> file_Location >> label;
		//cout << file_Location << "	" << label << endl;
		fileMap.insert(make_pair(file_Location, label));
	}
	return fileMap;
}

double getSkewness(double *landmarks) {
	double mean_x_value = (landmarks[2] + landmarks[4] + landmarks[10] + landmarks[12]) / 4.0;
	double mean_y_value = (landmarks[3] + landmarks[5] + landmarks[11] + landmarks[13]) / 4.0;
	double theta = atan(((landmarks[2] * landmarks[3] + landmarks[4] * landmarks[5] + landmarks[10] * landmarks[11] + landmarks[12] * landmarks[13]) - 4 * mean_x_value*mean_y_value)
		/ (pow(landmarks[2], 2) + pow(landmarks[4], 2) + pow(landmarks[10], 2) + pow(landmarks[12], 2) - 4 * pow(mean_x_value, 2)));
	return theta * 180 / 3.1415926;
}

Mat rotateImage(Mat src, double angle) {
	Mat dst;
	Point2f pt(src.cols / 2, src.rows / 2);
	Mat r = getRotationMatrix2D(pt, angle, 1.0);
	warpAffine(src, dst, r, Size(src.cols, src.rows));
	return dst;
}

double *landmarkRotation(double *landmarks, double angle, double midX, double midY) {
	float s = sin(angle*3.1415926 / 180);
	float c = cos(angle*3.1415926 / 180);
	double *returnLandmarks = new double[16];
	for (size_t i = 0; i < 15; i += 2) {
		float x = 0;
		float y = 0;
		x = landmarks[i] - midX;
		y = landmarks[i + 1] - midY;
		float xnew = x*c - y*s;
		float ynew = x*s + y*c;
		x = xnew + midX;
		y = ynew + midY;
		returnLandmarks[i] = x;
		returnLandmarks[i + 1] = y;
	}
	return returnLandmarks;
}

Rect getCropImageBound(double center_x, double center_y, int crop_size) {
	Rect rect(static_cast<int>(center_x) - crop_size / 2, static_cast<int>(center_y) - crop_size / 2, crop_size, crop_size);
	return rect;
}

// Face Detection: return face bound
vector<Rect> faceDetection(Mat image) {

	CascadeClassifier face_cascade;
	vector<Rect> faces;

	if (!face_cascade.load(face_cascade_name)) {
		printf("Error Loading");
	}

	face_cascade.detectMultiScale(image, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cvSize(90, 90));
	return faces;
}

// flandmark version
double* landmarkDetection(Mat image, Rect rect_face) {
	IplImage *img_grayscale = &IplImage(image);
	double *landmarks = (double*)malloc(2 * landmarkModel->data.options.M * sizeof(double));
	int bbox[] = { rect_face.x,rect_face.y,rect_face.x + rect_face.width,rect_face.y + rect_face.height };
	flandmark_detect(img_grayscale, bbox, landmarkModel, landmarks);
	return landmarks;
}

// get feature lbp feature
Mat lbpFeature(Mat image) {
	Mat dst;
	GaussianBlur(image, dst, Size(5, 5), 5, 3, BORDER_CONSTANT);
	Mat lbpImage = ELBP(dst, 2, 8);
	return lbpImage;
}

inline bool isImgOutofBound(int x, int y, int crop_size, int img_width, int img_height) {
	if (x - crop_size / 2 < 0 || x + crop_size / 2 > img_width || y - crop_size / 2 < 0 || y + crop_size / 2 > img_height)
		return true;
	else
		return false;
}

