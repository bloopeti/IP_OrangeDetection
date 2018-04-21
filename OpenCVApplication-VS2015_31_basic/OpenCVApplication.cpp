// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <queue>
#include <time.h>
#include <iostream>
#include <math.h>
using namespace std;

Mat openImageGrayscale();
Mat openImageColor();
vector<int> calculateHistogram(Mat src);
Mat RGBtoGrayscale(Mat src);
Mat grayscaleToBW(Mat src, int threshold);
Mat autoThreshold(Mat src);
Mat anisotropicDiffusion(Mat &output, int width, int height); //need to find correct param
vector<KeyPoint> blobDetect(Mat src);
int numberOfBlobs(vector<KeyPoint> blobs);
Mat shadowReduction(Mat src);
Mat convolution(Mat src);
//Mat erosion(Mat src);
//Mat erode(Mat src);
//bool erosionCondition(Mat src, int ix, int jx);

Mat openImageGrayscale()
{
	char fname[MAX_PATH];
	Mat src;
	if (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		return src;
	}
	return src;
}

Mat openImageColor()
{
	char fname[MAX_PATH];
	Mat src;
	if (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_COLOR);
		return src;
	}
	return src;
}

vector<int> calculateHistogram(Mat src)
{
	int rows = src.rows;
	int cols = src.cols;
	vector<int> histogramVec(256);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			int k = src.at<uchar>(i, j);
			histogramVec[k]++;
		}
	return histogramVec;
}

Mat RGBtoGrayscale(Mat src)
{
	int rows = src.rows;
	int cols = src.cols;

	Mat g = Mat::zeros(rows, cols, CV_8UC1);
//	Mat gb = Mat::zeros(rows, cols, CV_8UC1);

	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			Vec3b pixel;
			pixel = src.at<Vec3b>(i, j);

			 uchar x = (pixel.val[0] + pixel.val[1] + pixel.val[2]) / 3;
			 if (x < 0)
				 g.at<uchar>(i, j) = 0;
			 else if (x > 255)
				 g.at<uchar>(i, j) = 255;
			 else
				 g.at<uchar>(i, j) = x;
//			gb.at<uchar>(i, j) = 0.0722 * pixel.val[0] + 0.7152 * pixel.val[1] + 0.2126 * pixel.val[2];
		}
	return g;
}

Mat grayscaleToBW(Mat src, int threshold)
{
	int rows = src.rows;
	int cols = src.cols;
	Mat result = Mat::zeros(rows, cols, CV_8UC1);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			if (src.at<uchar>(i, j) < threshold)
				result.at<uchar>(i, j) = 0;
			else
				result.at<uchar>(i, j) = 255;
		}
	return result;
}

Mat autoThreshold(Mat src)
{
	vector<int> hist = calculateHistogram(src);
	int hm = 0;
	int hM = 0;
	for (int i = 0; i < hist.size(); i++)
	{
		if (hist[i] > hist[hM])
			hM = i;
		else if (hist[i] < hist[hm])
			hm = i;
	}
	int thresh1 = (hm + hM) / 2;
	int thresh0 = thresh1;

	int meanInt1 = 0;
	int meanInt2 = 0;
	do
	{
		meanInt1 = 0;
		meanInt2 = 0;
		int sum = 0;
		for (int i = 0; i < thresh1; i++)
		{
			meanInt1 += i*hist.at(i);
			sum += hist.at(i);
		}
		if (sum != 0)
			meanInt1 /= sum;
		sum = 0;
		for (int i = thresh1; i < hist.size(); i++)
		{
			meanInt2 += i*hist.at(i);
			sum += hist.at(i);
		}
		if (sum != 0)
			meanInt2 /= sum;
		thresh0 = thresh1;
		thresh1 = (meanInt1 + meanInt2) / 2;
	} while (abs(thresh1 - thresh0) > 5);

	Mat binarized = grayscaleToBW(src, thresh1);
	return binarized;
}

//need to find correct param
Mat anisotropicDiffusion(Mat &src, int width, int height) 
{
	const double lambda = 1.0 / 7.0;
	const double k = 30;
	const int iter = 1;

	float ahN[3][3] = { { 0, 1, 0 },{ 0, -1, 0 },{ 0, 0, 0 } };
	float ahS[3][3] = { { 0, 0, 0 },{ 0, -1, 0 },{ 0, 1, 0 } };
	float ahE[3][3] = { { 0, 0, 0 },{ 0, -1, 1 },{ 0, 0, 0 } };
	float ahW[3][3] = { { 0, 0, 0 },{ 1, -1, 0 },{ 0, 0, 0 } };
	float ahNE[3][3] = { { 0, 0, 1 },{ 0, -1, 0 },{ 0, 0, 0 } };
	float ahSE[3][3] = { { 0, 0, 0 },{ 0, -1, 0 },{ 0, 0, 1 } };
	float ahSW[3][3] = { { 0, 0, 0 },{ 0, -1, 0 },{ 1, 0, 0 } };
	float ahNW[3][3] = { { 1, 0, 0 },{ 0, -1, 0 },{ 0, 0, 0 } };

	Mat hN = Mat(3, 3, CV_32FC1, &ahN);
	Mat hS = Mat(3, 3, CV_32FC1, &ahS);
	Mat hE = Mat(3, 3, CV_32FC1, &ahE);
	Mat hW = Mat(3, 3, CV_32FC1, &ahW);
	Mat hNE = Mat(3, 3, CV_32FC1, &ahNE);
	Mat hSE = Mat(3, 3, CV_32FC1, &ahSE);
	Mat hSW = Mat(3, 3, CV_32FC1, &ahSW);
	Mat hNW = Mat(3, 3, CV_32FC1, &ahNW);

	//mat initialisation
	Mat nablaN, nablaS, nablaW, nablaE, nablaNE, nablaSE, nablaSW, nablaNW;
	Mat cN, cS, cW, cE, cNE, cSE, cSW, cNW;

	//depth of filters
	int ddepth = -1;

	//center pixel distance
	double dx = 1, dy = 1, dd = sqrt(2);
	double idxSqr = 1.0 / (dx * dx), idySqr = 1.0 / (dy * dy), iddSqr = 1 / (dd * dd);

	Mat result = src.clone();

	for (int i = 0; i < iter; i++) {
		//filters 
		filter2D(result, nablaN, ddepth, hN);
		filter2D(result, nablaS, ddepth, hS);
		filter2D(result, nablaW, ddepth, hW);
		filter2D(result, nablaE, ddepth, hE);
		filter2D(result, nablaNE, ddepth, hNE);
		filter2D(result, nablaSE, ddepth, hSE);
		filter2D(result, nablaSW, ddepth, hSW);
		filter2D(result, nablaNW, ddepth, hNW);

		//exponential flux
		cN = nablaN / k;
		cN = cN.mul(cN);
		cN = 1.0 / (1.0 + cN);
		//exp(-cN, cN);

		cS = nablaS / k;
		cS = cS.mul(cS);
		cS = 1.0 / (1.0 + cS);
		//exp(-cS, cS);

		cW = nablaW / k;
		cW = cW.mul(cW);
		cW = 1.0 / (1.0 + cW);
		//exp(-cW, cW);

		cE = nablaE / k;
		cE = cE.mul(cE);
		cE = 1.0 / (1.0 + cE);
		//exp(-cE, cE);

		cNE = nablaNE / k;
		cNE = cNE.mul(cNE);
		cNE = 1.0 / (1.0 + cNE);
		//exp(-cNE, cNE);

		cSE = nablaSE / k;
		cSE = cSE.mul(cSE);
		cSE = 1.0 / (1.0 + cSE);
		//exp(-cSE, cSE);

		cSW = nablaSW / k;
		cSW = cSW.mul(cSW);
		cSW = 1.0 / (1.0 + cSW);
		//exp(-cSW, cSW);

		cNW = nablaNW / k;
		cNW = cNW.mul(cNW);
		cNW = 1.0 / (1.0 + cNW);
		//exp(-cNW, cNW);

		result = result + lambda * (idySqr * cN.mul(nablaN) + idySqr * cS.mul(nablaS) +
			idxSqr * cW.mul(nablaW) + idxSqr * cE.mul(nablaE) +
			iddSqr * cNE.mul(nablaNE) + iddSqr * cSE.mul(nablaSE) +
			iddSqr * cNW.mul(nablaNW) + iddSqr * cSW.mul(nablaSW));
	}
	return result;
}

vector<KeyPoint> blobDetect(Mat src)
{
	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;

	// Change thresholds
	params.minThreshold = 0;
	params.maxThreshold = 255;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 15;

	// Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity = 0.1;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.87;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;


	// Storage for blobs
	vector<KeyPoint> keypoints;


#if CV_MAJOR_VERSION < 3   // If you are using OpenCV 2

	// Set up detector with params
	SimpleBlobDetector detector(params);

	// Detect blobs
	detector.detect(im, keypoints);
#else 

	// Set up detector with params
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	// Detect blobs
	detector->detect(src, keypoints);
#endif 

	return keypoints;
}

int numberOfBlobs(vector<KeyPoint> blobs)
{
	return blobs.size();
}

// convert BGR to LAB. increase luminosity. convert back to BGR
Mat shadowReduction(Mat src)
{
	cv::Mat lab_image;

	cv::cvtColor(src, lab_image, CV_BGR2Lab);

	// Extract the L channel
	std::vector<cv::Mat> lab_planes(3);
	cv::split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

	// apply the CLAHE algorithm to the L channel
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	clahe->setClipLimit(2); // changed parameter 4 to 2
	cv::Mat dst;
	clahe->apply(lab_planes[0], dst);

	// Merge the the color planes back into an Lab image
	dst.copyTo(lab_planes[0]);
	cv::merge(lab_planes, lab_image);

	// convert back to RGB
	cv::Mat image_clahe;
	cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);
/*
	cv::imshow("image original", src);
	cv::imshow("image CLAHE", image_clahe);
	waitKey();
*/
	return image_clahe;
}

Mat convolution(Mat src)
{  
	/// Initialize arguments for the filter
	Point anchor = Point(-1, -1);
	double delta = 0;
	int ddepth = -1;
	int kernel_size = 3; // 3x3 matrix
	double k[3][3] = { {-1., -1., -1.},{-1., 8., -1.},{-1., -1., -1.} };
	Mat kernel = Mat(kernel_size, kernel_size, CV_64F, k);// .inv();// / (float)(kernel_size*kernel_size); // 3x3 kernel 1 on middle col, 0 else

	

	Mat result;
	/// Apply filter
	filter2D(src, result, ddepth, kernel, anchor, delta, BORDER_DEFAULT);

	//imshow("res", result);
	//imshow("src", src);
	//waitKey();

	return result;
}
//
//Mat erosion(Mat src)
//{
//	/// Initialize arguments for the filter
//	Point anchor = Point(-1, -1);
//	double delta = 0;
//	int ddepth = -1;
//	int kernel_size = 3; // 3x3 matrix
//	int k[3][3] = { { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 } };
//	Mat kernel = Mat(kernel_size, kernel_size, CV_64F, k).inv();
//
//
//
//	Mat result;
//	/// Apply filter
////	result = erode(src);
//	erode(src, result, kernel);
//
//	imshow("res", result);
//	imshow("src", src);
//	waitKey();
//
//	return result;
//}
//
//Mat erode(Mat src)
//{
//	int rows = src.rows;
//	int cols = src.cols;
//
//	Mat e = src.clone();
//
//	for (int i = 1; i < rows - 1; i++)
//		for (int j = 1; j < cols - 1; j++)
//		{
//			if (erosionCondition(src, i, j))
//				e.at<uchar>(i, j) = 255;
//		}
//	return e;
//}
//
//bool erosionCondition(Mat src, int ix, int jx) // 8 grid
//{
//	int rows = src.rows;
//	int cols = src.cols;
//	int i, j;
//
//	if (src.at<uchar>(ix, jx) == 0)
//	{
//		if (ix < rows && (jx + 1) < cols) // right
//		{
//			i = ix;
//			j = jx + 1;
//			if (src.at<uchar>(i, j) != 0)
//				return true;
//		}
//		if ((ix - 1) > 0 && (jx + 1) < cols) // up right
//		{
//			i = ix - 1;
//			j = jx + 1;
//			if (src.at<uchar>(i, j) != 0)
//				return true;
//		}
//		if ((ix - 1) > 0 && jx  < cols) // up
//		{
//			i = ix - 1;
//			j = jx;
//			if (src.at<uchar>(i, j) != 0)
//				return true;
//		}
//		if ((ix - 1) > 0 && (jx - 1) > 0) // up left
//		{
//			i = ix - 1;
//			j = jx - 1;
//			if (src.at<uchar>(i, j) != 0)
//				return true;
//		}
//		if (ix < rows && (jx - 1) > 0) // left
//		{
//			i = ix;
//			j = jx - 1;
//			if (src.at<uchar>(i, j) != 0)
//				return true;
//		}
//		if ((ix + 1) < cols && (jx - 1) > 0) // down left
//		{
//			i = ix + 1;
//			j = jx - 1;
//			if (src.at<uchar>(i, j) != 0)
//				return true;
//		}
//		if ((ix + 1) < rows && jx < cols) // down
//		{
//			i = ix + 1;
//			j = jx;
//			if (src.at<uchar>(i, j) != 0)
//				return true;
//		}
//		if ((ix + 1) < rows && (jx + 1) < cols) // down right
//		{
//			i = ix + 1;
//			j = jx + 1;
//			if (src.at<uchar>(i, j) != 0)
//				return true;
//		}
//	}
//	return false;
//}
//

void doIt()
{
	Mat src = openImageColor();
	Mat result;
	result = shadowReduction(src);
	//result = convolution(src);
	//result = RGBtoGrayscale(result);
	//result = autoThreshold(result);
	//imshow("res", result);
	//imshow("src", src);
	//waitKey();
	int i = numberOfBlobs(blobDetect(src));
	printf("blobs:%d", i);
	getchar();
	getchar();
}

int main()
{
	srand(time(NULL));

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
		case 1:
			doIt();
			break;
		}
	}
	while (op!=0);
	return 0;
}