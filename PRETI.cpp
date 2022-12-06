#include <opencv2/opencv.hpp>
#include <iostream>

#define FACE_STOREGE_SIZE 30
#define FACE_ERROR_MARGIN 30
#define FACE_MIN_NEIGHBORS 9
#define EYES_MIN_NEIGHBORS 15


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

    Mat frame, face, leftEyeROI, rightEyeROI;
    Rect faceROI;
    vector<Rect> eyesFromLeft, eyesFromRight;
    queue<Rect> latest_faces; // index -1 : latest face rect
    uint16_t face_error_count = 0;
    Point leftCenter, rightCenter, focusPoint;
    double sightWeight = 0;

    //TODO: 얼굴 검출 안된 경우 예외 처리(지금은 강제 종료됨)
    while(true){
        Mat gray_frame;
        vector<Rect> faces;

        cap >> frame;
        if(frame.empty()){ break; }

        flip(frame, frame, 1);                                          // 좌우반전
        cvtColor(frame, gray_frame, COLOR_BGR2GRAY);                    // 계산을 위해 그레이 스케일로 변환

        face_classifier.detectMultiScale(gray_frame, faces, 1.1, FACE_MIN_NEIGHBORS);     // scaleFactor=1.1, minNeighbors=9

        /* 얼굴 선택기(예외 발생 시 가장 최근 선택된 얼굴로 강제 대체) */
        // TODO: 프레임 간의 얼굴 영역 마진 처리
        if(faces.size()==1){
            faceROI = faces.at(0);
            if(latest_faces.size() < FACE_STOREGE_SIZE){
                latest_faces.push(faceROI);
            }else{
                latest_faces.pop();
                latest_faces.push(faceROI);
            }
        }else{
            /* FACE_ERROR_MARGIN만큼 오류 처리 후에도 얼굴이 존재하지 않는다면 강제 초기화 */
            if(face_error_count < FACE_ERROR_MARGIN){
                if(latest_faces.size() > 1){
                    faceROI = latest_faces.back();
                    face_error_count++;
                }else{
                    faceROI = Rect();
                }
            }else{
                queue<Rect> eraser;
                swap(eraser, latest_faces); // 얼굴 저장 배열 초기화
                faceROI = Rect();
                face_error_count = 0;
            }
        }

        // rectangle(frame, face, Scalar(255, 0, 255), 2); 얼굴 상자 표시
        
        /* 얼굴 영역 상하 분할 후 상부 선택 */
        faceROI.height = cvRound(faceROI.height/2);
        leftEyeROI = frame(Rect(faceROI.x, faceROI.y, cvRound(faceROI.width/2), faceROI.height));
        rightEyeROI = frame(Rect(faceROI.x+cvRound(faceROI.width/2), faceROI.y, cvRound(faceROI.width/2), faceROI.height));
        
        eye_classifier.detectMultiScale(leftEyeROI, eyesFromLeft, 1.1, EYES_MIN_NEIGHBORS); // scaleFactor=1.1
        eye_classifier.detectMultiScale(rightEyeROI, eyesFromRight, 1.1, EYES_MIN_NEIGHBORS); // scaleFactor=1.1
        
        // TODO: 안구 검출 안되는 경우(ex. 눈 감을 때) 예외 처리. PPT 알고리즘 5번 참고
        for (Rect eye : eyesFromLeft) {
            leftCenter = Point(eye.x + eye.width / 2, eye.y + eye.height / 2);
            circle(frame(faceROI), leftCenter, 3, Scalar(0, 255, 255), -1, LINE_AA);
        }

        for (Rect eye : eyesFromRight) {
            rightCenter = Point((eye.x + eye.width / 2)+leftEyeROI.cols, eye.y + eye.height / 2);
            circle(frame(faceROI), rightCenter, 3, Scalar(0, 255, 255), -1, LINE_AA);
        }
        /* 초점 및 시야각 가중치 계산 */
        focusPoint = Point(((rightCenter.x-leftCenter.x)/2)+leftCenter.x, // ((r-l)/2) + l
                            leftCenter.y<rightCenter.y?((leftCenter.y-rightCenter.y)/2)+leftCenter.y:((rightCenter.y-leftCenter.y)/2)+rightCenter.y); 
        sightWeight = norm(rightCenter-leftCenter);
        
        /* 가중치 변화 확인 */
        String norm_val = to_string(sightWeight);
        putText(frame, norm_val, Point(10, 30), 2, 1, Scalar(0, 0, 255));

        circle(frame(faceROI), focusPoint, 3, Scalar(0, 255, 255), -1, LINE_AA);
        line(frame(faceROI), leftCenter, rightCenter, Scalar(0, 0, 255), 1, LINE_AA);

        imshow("frame", frame);

        if(waitKey(10)==27){ break; }
    }
    destroyAllWindows();
}