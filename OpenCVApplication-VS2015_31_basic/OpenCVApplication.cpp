// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <queue>
#include <time.h>
#include <iostream>
#include <math.h>
#include <chrono>
#include <random>
using namespace std;
using namespace cv;

Mat openImageGrayscale();
Mat openImageColor();
vector<int> calculateHistogram(Mat src);
Mat RGBtoGrayscale(Mat src);
Mat grayscaleToBW(Mat src, int threshold);
Mat autoThreshold(Mat src);
Mat anisotropicDiffusion(Mat src, int option = 2, int iter = 15, double lambda = 1.0 / 7.0, double kappa = 30.0); //need to find correct param
vector<KeyPoint> blobDetect(Mat src);
int numberOfBlobs(vector<KeyPoint> blobs);
Mat shadowReduction(Mat src);
Mat convolution(Mat src);
void kMeansSegmentationExample();
Mat kMeansSegmentation(Mat src);
float dist(Point2f p1, Point2f p2);
vector<Point2f> generateClusterCenters(int k, int rows, int cols);
vector<Vec3b> generateColors(int k);
vector<Vec3b> fetchColors(Mat src, vector<Point2f> clusterCenters, int k);
Mat assignByDistance(Mat src, vector<Point2f> clusterCenters, int k);
Mat assignByColor(Mat src, vector<Vec3b> colors, int k);
vector<Vec3b> calculateClusterColorMeans(Mat src, Mat L, int k);
vector<Point2f> calculateClusterCenters(Mat src, Mat L, vector<Vec3b> colorMeans, int k);
Vec3i colorDifference(Vec3b col1, Vec3b col2);
float averageColorDifference(Vec3b col1, Vec3b col2);
float euclideanColorDifference(Vec3b col1, Vec3b col2);
int nrOfChangedClusters(Mat Lold, Mat Lnew);
Mat fullKMeans(Mat src, int k, int maxIterations);
Mat colorByCluster(Mat L, vector<Vec3b> colorMeans, int k);
Mat grayscaleFunctionOnColorImage(Mat src, int option);

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

// References:
//   P.Perona and J.Malik.
//   Scale - Space and Edge Detection Using Anisotropic Diffusion.
//   IEEE Transactions on Pattern Analysis and Machine Intelligence,
// 12(7) : 629 - 639, July 1990.
//
//   G.Grieg, O.Kubler, R.Kikinis, and F.A.Jolesz.
//   Nonlinear Anisotropic Filtering of MRI Data.
//   IEEE Transactions on Medical Imaging,
// 11(2) : 221 - 232, June 1992.
//
//   MATLAB implementation based on Peter Kovesi's anisodiff(.):
//   P.D.Kovesi.MATLAB and Octave Functions for Computer Vision and Image Processing.
//   School of Computer Science & Software Engineering,
//   The University of Western Australia.Available from :
//   <http ://www.csse.uwa.edu.au/~pk/research/matlabfns/>.
//
//   C++ implementation based on
//	 Daniel Simoes Lopes
//   ICIST
//   Instituto Superior Tecnico - Universidade Tecnica de Lisboa
//   danlopes(at) civil ist utl pt
//   http ://www.civil.ist.utl.pt/~danlopes
//   https ://www.mathworks.com/matlabcentral/fileexchange/14995-anisotropic-diffusion--perona---malik-?focused=5090800&tab=function
//   May 2007 original version.
//
//   and
//
//   Ishank Gulati BTech student at Malaviya National Institute of Technology
//   Anisotropic (Perona-Malik) Diffusion
//   18 Dec 2015
//   http ://ishankgulati.github.io/posts/Anisotropic-(Perona-Malik)-Diffusion/

//need to find correct param
//   conventional anisotropic diffusion(Perona & Malik) upon a gray scale
//   image.A 2D network structure of 8 neighboring nodes is considered for
//   diffusion conduction.
//
//       ARGUMENT DESCRIPTION :
//               src - gray scale image(MxN).
//               iter - number of iterations.
//               lambda - integration constant(0 <= lambda <= 1 / 7).
//                          Usually, due to numerical stability this
//                          parameter is set to its maximum value.
//               kappa - gradient modulus threshold that controls the conduction.
//               option - conduction coefficient functions proposed by Perona & Malik:
// 1 - c(x, y, t) = exp(-(nablaI / kappa). ^ 2),
//                              privileges high-contrast edges over low-contrast ones.
// 2 - c(x, y, t) = 1. / (1 + (nablaI / kappa). ^ 2),
//                              privileges wide regions over smaller ones.
//
//       OUTPUT DESCRIPTION :
//                result - (diffused)image with the largest scale - space parameter.
Mat anisotropicDiffusion(Mat src, int option, int iter, double lambda, double kappa)
{
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
	result.convertTo(result, CV_32FC1, 1.0 / 255.0);

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
		/*
		//option 2
				//exponential flux
				cN = nablaN / kappa;
				cN = cN.mul(cN);
				cN = 1.0 / (1.0 + cN);
				//exp(-cN, cN);

				cS = nablaS / kappa;
				cS = cS.mul(cS);
				cS = 1.0 / (1.0 + cS);
				//exp(-cS, cS);

				cW = nablaW / kappa;
				cW = cW.mul(cW);
				cW = 1.0 / (1.0 + cW);
				//exp(-cW, cW);

				cE = nablaE / kappa;
				cE = cE.mul(cE);
				cE = 1.0 / (1.0 + cE);
				//exp(-cE, cE);

				cNE = nablaNE / kappa;
				cNE = cNE.mul(cNE);
				cNE = 1.0 / (1.0 + cNE);
				//exp(-cNE, cNE);

				cSE = nablaSE / kappa;
				cSE = cSE.mul(cSE);
				cSE = 1.0 / (1.0 + cSE);
				//exp(-cSE, cSE);

				cSW = nablaSW / kappa;
				cSW = cSW.mul(cSW);
				cSW = 1.0 / (1.0 + cSW);
				//exp(-cSW, cSW);

				cNW = nablaNW / kappa;
				cNW = cNW.mul(cNW);
				cNW = 1.0 / (1.0 + cNW);
				//exp(-cNW, cNW);
		*/
		// Diffusion function.
		if (option == 1)
		{
			exp(-(nablaN / kappa).mul(nablaN / kappa), cN);
			exp(-(nablaS / kappa).mul(nablaS / kappa), cS);
			exp(-(nablaW / kappa).mul(nablaW / kappa), cW);
			exp(-(nablaE / kappa).mul(nablaE / kappa), cE);
			exp(-(nablaNE / kappa).mul(nablaNE / kappa), cNE);
			exp(-(nablaSE / kappa).mul(nablaSE / kappa), cSE);
			exp(-(nablaSW / kappa).mul(nablaSW / kappa), cSW);
			exp(-(nablaNW / kappa).mul(nablaNW / kappa), cNW);
		}
		else if (option == 2)
		{
			cN = 1. / (1 + (nablaN / kappa).mul(nablaN / kappa));
			cS = 1. / (1 + (nablaS / kappa).mul(nablaS / kappa));
			cW = 1. / (1 + (nablaW / kappa).mul(nablaW / kappa));
			cE = 1. / (1 + (nablaE / kappa).mul(nablaE / kappa));
			cNE = 1. / (1 + (nablaNE / kappa).mul(nablaNE / kappa));
			cSE = 1. / (1 + (nablaSE / kappa).mul(nablaSE / kappa));
			cSW = 1. / (1 + (nablaSW / kappa).mul(nablaSW / kappa));
			cNW = 1. / (1 + (nablaNW / kappa).mul(nablaNW / kappa));
		}

		result = result + lambda * (
			idySqr * cN.mul(nablaN) + idySqr * cS.mul(nablaS) +
			idxSqr * cW.mul(nablaW) + idxSqr * cE.mul(nablaE) +
			iddSqr * cNE.mul(nablaNE) + iddSqr * cSE.mul(nablaSE) +
			iddSqr * cNW.mul(nablaNW) + iddSqr * cSW.mul(nablaSW));
	}
	result.convertTo(result, CV_8UC1, 255.0);
	return result;
}

// Reference:
// Blob Detection Using OpenCV(Python, C++)
// FEBRUARY 17, 2015 BY SATYA MALLICK
// https ://www.learnopencv.com/blob-detection-using-opencv-python-c/
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

	// Filter by Color
	params.filterByColor = true;
	params.blobColor = 255;


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
	Mat srcclone = src.clone();
	if (srcclone.type() == CV_32FC1)
		srcclone.convertTo(srcclone, CV_8UC1, 255.0);
	cv::Mat lab_image;
	cv::cvtColor(srcclone, lab_image, CV_BGR2Lab);


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

void kMeansSegmentationExample()
{
	const int MAX_CLUSTERS = 5;
	Scalar colorTab[] =
	{
		Scalar(0, 0, 255),
		Scalar(0,255,0),
		Scalar(255,100,100),
		Scalar(255,0,255),
		Scalar(0,255,255)
	};
	Mat img(500, 500, CV_8UC3);
	RNG rng(12345);
	for (;;)
	{
		int k, clusterCount = rng.uniform(2, MAX_CLUSTERS + 1);
		int i, sampleCount = rng.uniform(1, 1001);
		Mat points(sampleCount, 1, CV_32FC2), labels;
		clusterCount = MIN(clusterCount, sampleCount);
		Mat centers;
		/* generate random sample from multigaussian distribution */
		for (k = 0; k < clusterCount; k++)
		{
			Point center;
			center.x = rng.uniform(0, img.cols);
			center.y = rng.uniform(0, img.rows);
			Mat pointChunk = points.rowRange(k*sampleCount / clusterCount,
				k == clusterCount - 1 ? sampleCount :
				(k + 1)*sampleCount / clusterCount);
			rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.y), Scalar(img.cols*0.05, img.rows*0.05));
		}
		randShuffle(points, 1, &rng);
		kmeans(points, clusterCount, labels,
			TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
			3, KMEANS_PP_CENTERS, centers);
		img = Scalar::all(0);
		for (i = 0; i < sampleCount; i++)
		{
			int clusterIdx = labels.at<int>(i);
			Point ipt = points.at<Point2f>(i);
			circle(img, ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA);
		}
		imshow("clusters", img);
		char key = (char)waitKey();
		if (key == 27 || key == 'q' || key == 'Q') // 'ESC'
			break;
	}
}

Mat kMeansSegmentation(Mat src)
{
	Mat result = src.clone();
	Mat centers;
	
	Mat input = src.reshape(0, 1);
	input.convertTo(input, CV_32FC1, 1.0 / 255.0);
	Mat res2 = input.clone();


	kmeans(input, 4, res2,
		TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
		3, KMEANS_PP_CENTERS, centers);

	Scalar colorTab[] =
	{
		Scalar(0, 0, 255),
		Scalar(0,255,0),
		Scalar(255,100,100),
		Scalar(255,0,255),
		Scalar(0,255,255)
	};

	result = Scalar::all(0);
	int size = input.rows*input.cols;
	for (int i = 0; i < size; i++)
	{
		int clusterIdx = res2.at<int>(i);
		Point ipt = input.at<Point2f>(i);
		circle(result, ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA);
	}

	imshow("result", result);
//	imshow("centers", centers);
	waitKey();

	return result;
}

float dist(Point2f p1, Point2f p2)
{
	float result = 0.0;
	result = (float)(p2.x - p1.x)*(float)(p2.x - p1.x);
	result += (float)(p2.y - p1.y)*(float)(p2.y - p1.y);
	result = sqrt(result);
	return result;
}

vector<Point2f> generateClusterCenters(int k, int rows, int cols)
{
	vector<Point2f> result;
	default_random_engine generator_rows;
	default_random_engine generator_cols;
	uniform_int_distribution<int> distribution_rows(0, rows);
	uniform_int_distribution<int> distribution_cols(0, cols);
	for (int i = 0; i < k; i++)
	{
		int randRow = distribution_rows(generator_rows);
		int randCol = distribution_cols(generator_cols);
		Point2f newPoint(randCol, randRow); // is inverted
		result.push_back(newPoint);
	}
	return result;
}

vector<Vec3b> generateColors(int k)
{
	vector<Vec3b> result;
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator(seed);
	uniform_int_distribution<int> distribution(0, 255);
	for (int i = 0; i < k; i++)
	{
		int randR = distribution(generator);
		int randG = (distribution(generator) + distribution(generator)) % 255 ;
		int randB = (distribution(generator) * distribution(generator)) % 255;
		Vec3b newColor(randB, randG, randR); //inverted
		result.push_back(newColor);
	}
	return result;
}

vector<Vec3b> set4BaseColors() // orange, blue, brown, green
{
	vector<Vec3b> result;
	Vec3b ORANGE = { 0,140,255 }; // oranges
	Vec3b SKYBLUE = { 235,206,135 }; // sky
	Vec3b BROWN = { 19,69,139 }; // branches
	Vec3b GREEN = { 34,139,34 }; // leaves/grass
	result.push_back(ORANGE);
	result.push_back(SKYBLUE);
	result.push_back(BROWN);
	result.push_back(GREEN);
	return result;
}

vector<Vec3b> fetchColors(Mat src, vector<Point2f> clusterCenters, int k)
{
	vector<Vec3b> result(k);
	for (int i = 0; i < k; i++)
	{
		Vec3b pixel = src.at<Vec3b>(clusterCenters[i].y, clusterCenters[i].x);
		result.push_back(pixel);
	}
	return result;
}

Mat assignByDistance(Mat src, vector<Point2f> clusterCenters, int k)
{
	int rows = src.rows;
	int cols = src.cols;
	Mat L = Mat(rows, cols, CV_8UC1); // labels; what cluster does each pixel belong to
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			int min = 0;
			Point2f current(i, j);
			float minDist = dist(current, clusterCenters[0]);
			for (int c = 1; c < k; c++)
			{
				float d = dist(current, clusterCenters[c]);
				if (d < minDist)
				{
					minDist = d;
					min = c;
				}
			}
			L.at<uchar>(i, j) = min;
		}
	return L;
}

Mat assignByColor(Mat src, vector<Vec3b> colors, int k)
{
	int rows = src.rows;
	int cols = src.cols;
	Mat L = Mat(rows, cols, CV_8UC1); // labels; what cluster does each pixel belong to
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			int min = 0;
			Vec3b current = src.at<Vec3b>(i, j);
			float minDiff = euclideanColorDifference(current, colors[0]);
			for (int c = 1; c < k; c++)
			{
				float d = euclideanColorDifference(current, colors[c]);
				if (d < minDiff)
				{
					minDiff = d;
					min = c;
				}
			}
			L.at<uchar>(i, j) = min;
		}
	return L;
}

vector<Vec3b> calculateClusterColorMeans(Mat src, Mat L, int k)
{
	vector<Vec3b> clusterColorMeans(k);
	vector<Vec3d> clusterColorSums(k);
	vector<int> cluserPixelCount(k);
	int rows = src.rows;
	int cols = src.cols;
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			int currentCluster = L.at<uchar>(i, j);
			clusterColorSums[currentCluster] += (Vec3d)src.at<Vec3b>(i, j);
			cluserPixelCount[currentCluster]++;
		}
	for (int i = 0; i < k; i++)
	{
		clusterColorMeans[i] = clusterColorSums[i] / cluserPixelCount[i];
	}
	return clusterColorMeans;
}

vector<Vec3b> calculateColorMeans(Mat src, Mat L, vector<Vec3b> colorMeans, int k)//to move reference
{
	vector<Point2f> clusterCenters(k);
	vector<Vec3b> newColorMeans(k);

	int rows = src.rows;
	int cols = src.cols;
	vector<float> clusterMinVariation(k);
	fill(clusterMinVariation.begin(), clusterMinVariation.end(), -1);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			int curr = L.at<uchar>(i, j);
			float colDiff = euclideanColorDifference(src.at<Vec3b>(i, j), colorMeans[curr]);
			if (clusterMinVariation[curr] != -1)
			{
				if (colDiff < clusterMinVariation[curr])
				{
					clusterMinVariation[curr] = colDiff;
					clusterCenters[curr] = Point2f(i, j);
					newColorMeans[curr] = src.at<Vec3b>(i, j);
				}
			}
			else
			{
				clusterMinVariation[curr] = colDiff;
				clusterCenters[curr] = Point2f(i, j);
				newColorMeans[curr] = src.at<Vec3b>(i, j);
			}
		}
	return newColorMeans;
}

vector<Point2f> calculateClusterCenters(Mat src, Mat L, vector<Vec3b> colorMeans, int k)
{
	vector<Point2f> clusterCenters(k);

	int rows = src.rows;
	int cols = src.cols;
	vector<float> clusterMinVariation(k);
	fill(clusterMinVariation.begin(), clusterMinVariation.end(), -1);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			int curr = L.at<uchar>(i, j);
			if (clusterMinVariation[curr] != -1)
			{
				if (clusterMinVariation[curr] > euclideanColorDifference(src.at<Vec3b>(i, j), colorMeans[curr]))
				{
					clusterMinVariation[curr] = euclideanColorDifference(src.at<Vec3b>(i, j), colorMeans[curr]);
					clusterCenters[curr] = Point2f(i, j);
				}
			}
			else
			{
				clusterMinVariation[curr] = euclideanColorDifference(src.at<Vec3b>(i, j), colorMeans[curr]);
				clusterCenters[curr] = Point2f(i, j);
			}
		}
	return clusterCenters;
}

Vec3i colorDifference(Vec3b col1, Vec3b col2)
{
	Vec3i result;
	result[0] = col1[0] - col2[0];
	result[1] = col1[1] - col2[1];
	result[2] = col1[2] - col2[2];
	return result;
}

float averageColorDifference(Vec3b col1, Vec3b col2)
{
	float result = 0;
	Vec3i colDif = colorDifference(col1, col2);
	result += colDif[0] + colDif[1] + colDif[2];
	result /= 3;
	result = abs(result);
	return result;
}

float euclideanColorDifference(Vec3b col1, Vec3b col2)
{
	float result = 0.0;
	result = (float)(col1[0] - col2[0])*(float)(col1[0] - col2[0]);
	result += (float)(col1[1] - col2[1])*(float)(col1[1] - col2[1]);
	result += (float)(col1[2] - col2[2])*(float)(col1[2] - col2[2]);
	result = sqrt(result);
	return result;
}

int nrOfChangedClusters(Mat Lold, Mat Lnew)
{
	int rows = Lold.rows;
	int cols = Lold.cols;
	int result = 0;
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			if (Lold.at<uchar>(i, j) != Lnew.at<uchar>(i, j))
				result++;
		}
	return result;
}

Mat fullKMeans(Mat src, int k, int maxIterations)
{
	int rows = src.rows;
	int cols = src.cols;
	//vector<Point2f> centersNew = generateClusterCenters(k, rows, cols);
	//vector<Point2f> centersOld(centersNew);
	vector<Vec3b> colorMeansNew = set4BaseColors();
	vector<Vec3b> colorMeansOld(colorMeansNew);
	Mat L = Mat(rows, cols, CV_8UC1);
	int iterations = -1;
	printf("%d,%d,%d   %d,%d,%d   %d,%d,%d   %d,%d,%d\n%d\n",
		colorMeansNew[0][0], colorMeansNew[0][1], colorMeansNew[0][2],
		colorMeansNew[1][0], colorMeansNew[1][1], colorMeansNew[1][2],
		colorMeansNew[2][0], colorMeansNew[2][1], colorMeansNew[2][2],
		colorMeansNew[3][0], colorMeansNew[3][1], colorMeansNew[3][2],
		iterations);
	do
	{
		iterations++;
		L = assignByColor(src, colorMeansNew, k);
		//colorMeans = calculateClusterColorMeans(src, L, k);
		colorMeansOld = colorMeansNew;
		colorMeansNew = calculateColorMeans(src, L, colorMeansOld, k);
		printf("%d,%d,%d   %d,%d,%d   %d,%d,%d   %d,%d,%d\n%d\n",
			colorMeansNew[0][0], colorMeansNew[0][1], colorMeansNew[0][2],
			colorMeansNew[1][0], colorMeansNew[1][1], colorMeansNew[1][2],
			colorMeansNew[2][0], colorMeansNew[2][1], colorMeansNew[2][2],
			colorMeansNew[3][0], colorMeansNew[3][1], colorMeansNew[3][2],
			iterations);
	} while ((colorMeansOld != colorMeansNew) && (iterations < maxIterations));
	Mat result = colorByCluster(L, colorMeansNew, k);
	return result;
}

Mat colorByCluster(Mat L, vector<Vec3b> colorMeans, int k)
{
	int rows = L.rows;
	int cols = L.cols;
	Mat result(rows, cols, CV_8UC3);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			int curr = L.at<uchar>(i, j);
			result.at<Vec3b>(i, j) = colorMeans[curr];
		}
	return result;
}

Mat grayscaleFunctionOnColorImage(Mat src, int option)
{
	vector<Mat> rgbChannels(3);
	//split(src, rgbChannels);
	int rows = src.rows;
	int cols = src.cols;
	rgbChannels[0] = Mat(rows, cols, CV_8UC1);
	rgbChannels[1] = Mat(rows, cols, CV_8UC1);
	rgbChannels[2] = Mat(rows, cols, CV_8UC1);

	for(int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			Vec3b pixel = src.at<Vec3b>(i, j);
			rgbChannels[0].at<uchar>(i, j) = pixel[0];
			rgbChannels[1].at<uchar>(i, j) = pixel[1];
			rgbChannels[2].at<uchar>(i, j) = pixel[2];
		}

	vector<Mat> results(3);
	results[0] = Mat(rows, cols, CV_8UC1);
	results[1] = Mat(rows, cols, CV_8UC1);
	results[2] = Mat(rows, cols, CV_8UC1);
	if (option == 1)
	{
		results[0] = anisotropicDiffusion(rgbChannels[0], 2, 1);
		results[1] = anisotropicDiffusion(rgbChannels[1], 2, 1);
		results[2] = anisotropicDiffusion(rgbChannels[2], 2, 1);
	}
	else if (option == 2)
	{
		results[0] = grayscaleToBW(rgbChannels[0], 255);
		results[1] = grayscaleToBW(rgbChannels[1], 140);
		results[2] = grayscaleToBW(rgbChannels[2], 0);
	}


	//imshow("resB", results[0]);
	//waitKey();
	//imshow("resG", results[1]);
	//waitKey();
	//imshow("resR", results[2]);
	//waitKey();

	Mat result(src.rows, src.cols, CV_8UC3);
	//merge(results, result);
	////for (int i = 0; i < rows; i++)
	////	for (int j = 0; j < cols; j++)
	////	{
	////		Vec3b pixel;
	////		pixel[0] = results[0].at<uchar>(i, j);
	////		pixel[1] = results[1].at<uchar>(i, j);
	////		pixel[2] = results[2].at<uchar>(i, j);
	////		result.at<Vec3b>(i, j) = pixel;
	////	}

	//imshow("resCombined", result);
	//waitKey();
	return result;
}

void l6KMeansClustering(Mat img)
{
	int one = img.rows;
	int two = img.cols;
	Mat label(one, two, CV_8UC1, Scalar(255));
	std::default_random_engine gen;
	std::uniform_int_distribution<int> distributionX(0, img.cols);
	int randintX;
	std::uniform_int_distribution<int> distributionY(0, img.rows);
	int randintY;
	std::uniform_int_distribution<int> dist_img(0, 255);

	int k;
	printf("Number of clusters: ");
	scanf("%d", &k);

	vector<Point> seeds;
	vector<int> sumx(k, 0);
	vector<int> sumy(k, 0);
	vector<int> count(k, 0);

	for (int i = 0; i < k; i++)
	{
		randintX = distributionX(gen);
		randintY = distributionY(gen);
		seeds.push_back(Point(randintX, randintY));
	}

	for (Point vec : seeds)
		std::cout << vec << std::endl;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			if (img.at<uchar>(i, j) == 0)
			{
				//Find out to which cluster it belongs
				int min = 9999;
				int lbl;
				int dist = 0;
				for (int seedIndex = 0; seedIndex < k; seedIndex++)
				{
					dist = sqrt((i - seeds[seedIndex].x)*(i - seeds[seedIndex].x) + (j - seeds[seedIndex].y)*(j - seeds[seedIndex].y));
					if (dist < min)
					{
						min = dist;
						lbl = seedIndex;
					}
					printf("Entered in for %d %d %d %d\n", seedIndex, lbl, dist, min);
				}
				//Now I know to which is the closest
				label.at<uchar>(i, j) = lbl;
				//printf("%d\n", lbl);

				//Update seed coordinates
				sumx[lbl] = sumx[lbl] + seeds[lbl].x;
				sumy[lbl] = sumy[lbl] + seeds[lbl].x;
				count[lbl]++;
				printf("1: %d %d\n", seeds[lbl], seeds[lbl]);
				seeds[lbl].x = sumx[lbl] / count[lbl];
				seeds[lbl].y = sumy[lbl] / count[lbl];
				printf("2: %d %d\n", seeds[lbl], seeds[lbl]);
			}

		}
	Mat color(one, two, CV_8UC3, Scalar(0, 0, 0));
	const int K = k;
	Vec3b colors[4];
	for (int i = 0; i < K; i++)
	{
		colors[i] = { (uchar)dist_img(gen), (uchar)dist_img(gen), (uchar)dist_img(gen) };
		//printf("%u, %u, %u\n", colors[i](0), colors[i](1), colors[i](2));
	}

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			if (label.at<uchar>(i, j) != 255)
			{
				color.at<Vec3b>(i, j) = colors[label.at<uchar>(i, j)];
				//printf("%u, %u, %u\n", color.at<Vec3b>(i, j)(0), color.at<Vec3b>(i, j)(1), color.at<Vec3b>(i, j)(2));
			}

	Mat hh(img.cols, img.rows, CV_8UC3, Scalar(0, 0, 0));
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			{
				//Find out to which cluster it belongs
				int min = 9999;
				int lbl;
				int dist = 0;
				for (int seedIndex = 0; seedIndex < k; seedIndex++)
				{
					dist = sqrt((j - seeds[seedIndex].x)*(j - seeds[seedIndex].x) + (i - seeds[seedIndex].y)*(i - seeds[seedIndex].y));
					if (dist < min)
					{
						min = dist;
						lbl = seedIndex;
					}
					printf("Entered in for %d %d %d %d\n", seedIndex, lbl, dist, min);
				}
				//Now I know to which is the closest
				label.at<uchar>(i, j) = lbl;
				printf("%d\n", lbl);
			}
			hh.at<Vec3b>(i, j) = colors[label.at<uchar>(i, j)];

		}

	for (int i = 0; i < k; i++)
		circle(color, seeds[i], 3, (Scalar)colors[i], 1, 8, 0);
	imshow("Image", img);
	imshow("Label", label);
	imshow("Color", color);
	imshow("Hh", hh);
	waitKey(0);
}

void doIt()
{
	Mat src = openImageColor();
	Mat result = src.clone();
	//result = grayscaleFunctionOnColorImage(result, 1);
	imshow("resPerona", result);
	//result = shadowReduction(result);
	imshow("resShadowReduction", result);
	//result = convolution(result);
	imshow("resConvolution", result);
	//result = RGBtoGrayscale(result);
	//imshow("resGrayscale", result);
	//result = autoThreshold(result);
	//imshow("resThreshold", result);
	result = fullKMeans(result, 4, 100);

	//imshow("resKmeans", result);
	int i = numberOfBlobs(blobDetect(result));

	imshow("src", src);
	imshow("resFinal", result);
	printf("blobs:%d", i);

	waitKey();
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