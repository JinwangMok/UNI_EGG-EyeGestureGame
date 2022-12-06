#include <opencv2/opencv.hpp>
#include <iostream>

#define FACE_STOREGE_SIZE 30
#define FACE_ERROR_MARGIN 30

using namespace std;
using namespace cv;

CascadeClassifier face_classifier;
CascadeClassifier eye_classifier;

int main(){
    face_classifier.load("./cascade_xmls/haarcascade_frontalface_alt2.xml");
	eye_classifier.load("./cascade_xmls/haarcascade_eye.xml");

    VideoCapture cap(0);

	if (face_classifier.empty() || eye_classifier.empty() || !cap.isOpened()){
		cerr << "Something load failed!" << endl;
		return -1;
	}

    Mat frame;
    Rect face;
    queue<Rect> latest_faces; // index -1 : latest face rect
    uint16_t face_error_count = 0;

    while(true){
        Mat gray_frame;
        vector<Rect> faces;

        cap >> frame;
        if(frame.empty()){ break; }

        flip(frame, frame, 1);                                          // 좌우반전
        cvtColor(frame, gray_frame, COLOR_BGR2GRAY);                    // 계산을 위해 그레이 스케일로 변환

        face_classifier.detectMultiScale(gray_frame, faces, 1.1, 9);     // scaleFactor=1.1, minNeighbors=9

        /* 얼굴 선택기(예외 발생 시 가장 최근 선택된 얼굴로 강제 대체) */
        if(faces.size()==1){
            face = faces.at(0);
            if(latest_faces.size() < FACE_STOREGE_SIZE){
                latest_faces.push(face);
            }else{
                latest_faces.pop();
                latest_faces.push(face);
            }
        }else{
            /* FACE_ERROR_MARGIN만큼 오류 처리 후에도 얼굴이 존재하지 않는다면 강제 초기화 */
            if(face_error_count < FACE_ERROR_MARGIN){
                if(latest_faces.size() > 1){
                    face = latest_faces.back();
                    face_error_count++;
                }else{
                    face = Rect();
                }
            }else{
                queue<Rect> eraser;
                swap(eraser, latest_faces); // 얼굴 저장 배열 초기화
                face = Rect();
                face_error_count = 0;
            }
        }

        rectangle(frame, face, Scalar(255, 0, 255), 2);

        Mat faceROI = frame(face);
        vector<Rect> eyes;
        eye_classifier.detectMultiScale(faceROI, eyes, 1.1, 8); // scaleFactor=1.1, minNeighbors=8

        for (Rect eye : eyes) {
            Point center(eye.x + eye.width / 2, eye.y + eye.height / 2);
            circle(faceROI, center, eye.width / 2, Scalar(255, 0, 0), 2, LINE_AA);
        }

        imshow("frame", frame);

        if(waitKey(10)==27){ break; }
    }
    destroyAllWindows();
}