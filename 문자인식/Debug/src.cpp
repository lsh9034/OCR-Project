#include <opencv.hpp>
#include <cmath>
using namespace std;
using namespace cv;
/*int main(void)
{

	cv::VideoCapture capture(0);

	if (!capture.isOpened()) {
		std::cerr << "Could not open camera" << std::endl;
		return 0;
	}

	// create a window
	cv::namedWindow("webcam", 1);

	// -------------------------------------------------------------------------
	// face detection configuration
	cv::CascadeClassifier face_classifier;
	//face_classifier.load("E://opencv//opencv//build//etc//haarcascades//haarcascade_frontalface_default.xml");
	face_classifier.load("C://OpenCV2.4.11//sources//data//haarcascades//haarcascade_frontalface_default.xml");
	while (true) {
		bool frame_valid = true;

		cv::Mat frame_original;
		cv::Mat frame;

		try {
			capture >> frame_original; // get a new frame from webcam
			cv::resize(frame_original, frame, cv::Size(frame_original.cols / 2,
				frame_original.rows / 2), 0, 0, CV_INTER_NN); // downsample 1/2x
		}
		catch (cv::Exception& e) {
			std::cerr << "Exception occurred. Ignoring frame... " << e.err
				<< std::endl;
			frame_valid = false;
		}

		if (frame_valid) {
			try {
				// convert captured frame to gray scale & equalize
				cv::Mat grayframe;
				cv::cvtColor(frame, grayframe, CV_BGR2GRAY);
				cv::equalizeHist(grayframe, grayframe);

				// -------------------------------------------------------------
				// face detection routine

				// a vector array to store the face found
				std::vector<cv::Rect> faces;

				face_classifier.detectMultiScale(grayframe, faces,
					1.1, // increase search scale by 10% each pass
					3,   // merge groups of three detections
					0 | CV_HAAR_SCALE_IMAGE,         //CV_HAAR_FIND_BIGGEST_OBJECT
					cv::Size(30, 30));

				// -------------------------------------------------------------
				// draw the results
				for (int i = 0; i<faces.size(); i++) {
					cv::Point lb(faces[i].x + faces[i].width,
						faces[i].y + faces[i].height);
					cv::Point tr(faces[i].x, faces[i].y);

					cv::rectangle(frame, lb, tr, cv::Scalar(0, 255, 0), 3, 4, 0);
				}

				// print the output
				cv::imshow("webcam", frame);

			}
			catch (cv::Exception& e) {
				std::cerr << "Exception occurred. Ignoring frame... " << e.err
					<< std::endl;
			}
		}
		if (cv::waitKey(30) >= 0) break;
	}

	// VideoCapture automatically deallocate camera object
	return 0;

	/*
	VideoCapture capture(0);
	if (!capture.isOpened()) {
	std::cerr << "Could not open camera" << std::endl;
	return 0;
	}

	Size size = Size((int)capture.get(CAP_PROP_FRAME_WIDTH),(int)capture.get(CAP_PROP_FRAME_HEIGHT));

	imshow("frame", NULL);
	waitKey(1000);
	namedWindow("frame", WINDOW_AUTOSIZE);

	int delay = 30;
	int frameNum = -1;
	Mat frame, grayImage, edgeImage;

	for (;;)
	{
	capture >> frame;
	if (frame.empty())break;
	cout << "frameNum:" << ++frameNum << endl;

	cvtColor(frame, grayImage, COLOR_BGR2GRAY);
	Canny(grayImage, edgeImage, 80, 150, 3);

	flip(edgeImage, edgeImage, 1);
	flip(frame, frame, 1);
	imshow("frame", frame);
	imshow("edgeImage", edgeImage);
	int ckey = waitKey(delay);
	if (ckey == 27)break;
	}

	return 0;
	
}*/

int max_c(int r = 0, int g = 0, int b = 0)
{
	if (r > g && r > b) return r;
	if (g > r && g > b) return g;
	return b;
}

int min_c(int r = 0, int g = 0, int b = 0)
{
	if (r < g && r < b) return r;
	if (g < r && g < b) return g;
	return b;
}

int main()
{
	Mat img(500, 500, CV_8UC3);
	//img = imread("C:\\Users\\philip\\Desktop\\OpenCV\\lena.jpg", IMREAD_COLOR);
	//Mat img2(img.size(), CV_8UC3);
	//imshow("i", img);
	//cvtColor(img2, img2, COLOR_BGR2HSV);
	//Point p1(100, 100), p2(100, 101), p3(100, 102), p4(100, 103), p5(100, 104);
	//Scalar black(0, 0, 0), blue(255, 0, 0), green(0, 255, 0), red(0, 0, 255);
	//line(img, p1, p2, black);
	//line(img, p2, p3, red);
	//line(img, p3, p4, green);
	//line(img, p4, p5, blue);
	/*
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			int h, s, v1, v2;
			int b = img.at<uchar>(i, j * 3);
			int g = img.at<uchar>(i, j * 3 + 1);
			int r = img.at<uchar>(i, j * 3 + 2);
			v1 = max_c(r,g,b);
			v2 = min_c(r, g, b);
			if (v1) s = (int)(((double)(v1 - v2) / (double)v1) * (double)255);
			if (v1 == r) h = (int)((double)60 * (double)(g - b) / (double)(v1 - v2));
			else if (v1 == r) h = 120 + (int)((double)60 * (double)(b - r) / (double)(v1 - v2));
			else h = 240 + (int)((double)60 * (double)(r - g) / (double)(v1 - v2));
			img2.at<uchar>(i, j * 3) = h;
			img2.at<uchar>(i, j * 3 + 1) = s;
			img2.at<uchar>(i, j * 3 + 2) = v1;
		}*/
	for (int v = 0; v < 1; v++)
	{
		for (int s = 0; s < 255; s++)
		{
			for (int h = 0; h < 255; h++)
			{
				line(img, Point(s, h), Point(s + 1, h), Scalar(h, s, v), 1);
			}
		}
		cvtColor(img, img, COLOR_BGR2HSV);
		imshow("img", img);
		waitKey(10);
	} 
	//imshow("im", img);
	//imshow("img", img2);
	//cout << img2.channels() << endl;
	cout << "done" << endl;
	waitKey(0);
	return 0;
}