#include <iostream>
#include <opencv2/opencv.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/objdetect/objdetect.hpp>

using namespace std;
using namespace cv;

CascadeClassifier face_cascade;
CascadeClassifier eye_cascade;

vector<Rect> faces;
Mat dst;

int detectEye(Mat& im, Mat& tpl, Rect& rect);
void trackEye(Mat& im, Mat& tpl, Rect& rect);

int main(int argc, char** argv)
{
	face_cascade.load("../cascade_xmls/haarcascade_frontalface_alt2.xml");
	eye_cascade.load("../cascade_xmls/haarcascade_eye.xml");

	VideoCapture cap(0);

	if (face_cascade.empty() || eye_cascade.empty() || !cap.isOpened()){
		cerr << "Something load failed!" << endl;
		return -1;
	}

	cap.set(CAP_PROP_FRAME_WIDTH, 320);
	cap.set(CAP_PROP_FRAME_HEIGHT, 240);

	Mat frame, eye_tpl;
	Rect eye_bb;

	while (waitKey(15) != 'q')
	{
		cap >> frame;

		if (frame.empty()){
			cerr << "Frame is empty!" << endl;
			break;
		}

		flip(frame, frame, 1); // 좌우대칭

		Mat resized, gray;
		cvtColor(frame, gray, COLOR_BGR2GRAY); // 계산을 위해 그레이 스케일로 변환

		if (eye_bb.width == 0 && eye_bb.height == 0)
		{
			// 초기 검출
			detectEye(gray, eye_tpl, eye_bb);
		}
		else
		{
			// Tracking stage with template matching
			trackEye(gray, eye_tpl, eye_bb);

			// Draw bounding rectangle for the eye
			// rectangle(frame, eye_bb, CV_RGB(0,255,0));

			rectangle(frame, faces[0], CV_RGB(0,255,0));
			dst = gray(faces[0]);

            Mat tempImg = frame(eye_bb);
            resize(tempImg, resized, Size(500, 500));
		}
 
		imshow("video", frame);

        if(!resized.empty()){
			cvtColor(resized, resized, COLOR_BGR2GRAY);
            imshow("resized", resized);
        }
	}

	imshow("dst", dst);
	waitKey();
    destroyAllWindows();
} 

int detectEye(Mat& im, Mat& tpl, Rect& rect)
{
	vector<Rect> faces, eyes;
	// vector<Rect> eyes;
	face_cascade.detectMultiScale(im, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30,30)); // 이 부분에서 파라미터 수정 요구

	for (int i = 0; i < faces.size(); i++)
	{
		Mat face = im(faces[i]);
		eye_cascade.detectMultiScale(face, eyes, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(20,20));

		if (eyes.size())
		{
			rect = eyes[0] + Point(faces[i].x, faces[i].y);
			tpl  = im(rect);
		}
	}

	return eyes.size();
}

void trackEye(Mat& im, Mat& tpl, Rect& rect)
{
	Size size(rect.width * 2, rect.height * 2);
	Rect window(rect + size - Point(size.width/2, size.height/2));
	
	window &= Rect(0, 0, im.cols, im.rows);

	Mat dst(window.width - tpl.rows + 1, window.height - tpl.cols + 1, CV_32FC1);
	matchTemplate(im(window), tpl, dst, TM_SQDIFF_NORMED);

	double minval, maxval;
	Point minloc, maxloc;
	minMaxLoc(dst, &minval, &maxval, &minloc, &maxloc);

	if (minval <= 0.2)
	{
		rect.x = window.x + minloc.x;
		rect.y = window.y + minloc.y;
	}
	else
		rect.x = rect.y = rect.width = rect.height = 0;
}