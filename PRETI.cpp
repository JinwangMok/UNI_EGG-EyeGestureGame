#include <opencv2/opencv.hpp>
#include <iostream>

#define FACE_STOREGE_SIZE 30
#define FACE_ERROR_MARGIN 30
#define FACE_MIN_NEIGHBORS 9
#define EYES_MIN_NEIGHBORS 15
#define DISPLAY_WIDTH 3840
#define DISPLAY_HEIGHT 2160

using namespace std;
using namespace cv;

CascadeClassifier face_classifier;
CascadeClassifier eye_classifier;

//TODO: Call by Ref로 각 프레임마다 수행하는 코드 단위로 함수 분할 필요


int main(){
    face_classifier.load("./cascade_xmls/haarcascade_frontalface_alt2.xml");
	eye_classifier.load("./cascade_xmls/haarcascade_eye.xml");

    VideoCapture cap(0);

	if (face_classifier.empty() || eye_classifier.empty() || !cap.isOpened()){
		cerr << "Something load failed!" << endl;
		return -1;
	}
    
    /* 디스플레이 크기는 OS마다 다르다고 함. 확인 필요! */
    Mat display(DISPLAY_HEIGHT, DISPLAY_WIDTH, CV_8UC3, Scalar::all(255));
    Point ORIGIN(cvRound(display.cols/2), cvRound(display.rows/2));

    Mat frame, leftEyeROI, rightEyeROI;
    Rect faceROI;
    vector<Rect> eyesFromLeft, eyesFromRight;
    queue<Rect> latest_faces; // index -1 : latest face rect
    uint16_t faceErrorCount = 0, eyeErrorCount = 0;
    Point leftCenter, rightCenter, focusPoint;
    double sightWeight = 0;

    //TODO: 얼굴 검출 안된 경우 예외 처리(지금은 강제 종료됨)
    while(true){
        Mat gray_frame;
        vector<Rect> faces;

        cap >> frame;
        if(frame.empty()){ break; }

        flip(frame, frame, 1);// 좌우반전
        cvtColor(frame, gray_frame, COLOR_BGR2GRAY);// 계산을 위해 그레이 스케일로 변환

        face_classifier.detectMultiScale(gray_frame, faces, 1.1, FACE_MIN_NEIGHBORS);// scaleFactor=1.1, minNeighbors=9

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
            if(faceErrorCount < FACE_ERROR_MARGIN){
                if(latest_faces.size() > 1){
                    faceROI = latest_faces.back();
                    faceErrorCount++;
                }else{
                    faceROI = Rect();
                }
            }else{
                queue<Rect> eraser;
                swap(eraser, latest_faces); // 얼굴 저장 배열 초기화
                faceROI = Rect();
                faceErrorCount = 0;
            }
        }

        // rectangle(frame, face, Scalar(255, 0, 255), 2); 얼굴 상자 표시
        
        /* 얼굴 영역 상하 분할 후 상부 선택 */
        faceROI.height = cvRound(faceROI.height/2);
        leftEyeROI = frame(Rect(faceROI.x, faceROI.y, cvRound(faceROI.width/2), faceROI.height));
        rightEyeROI = frame(Rect(faceROI.x+cvRound(faceROI.width/2), faceROI.y, cvRound(faceROI.width/2), faceROI.height));
        
        eye_classifier.detectMultiScale(leftEyeROI, eyesFromLeft, 1.1, EYES_MIN_NEIGHBORS); // scaleFactor=1.1
        eye_classifier.detectMultiScale(rightEyeROI, eyesFromRight, 1.1, EYES_MIN_NEIGHBORS); // scaleFactor=1.1
        
        /* 양안의 동공 검출 */
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
        // String norm_val = to_string(sightWeight);
        // putText(frame, norm_val, Point(10, 30), 2, 1, Scalar(0, 0, 255));

        circle(frame(faceROI), focusPoint, 3, Scalar(0, 255, 255), -1, LINE_AA);
        line(frame(faceROI), leftCenter, rightCenter, Scalar(0, 0, 255), 1, LINE_AA);

        imshow("frame", frame);

        /* 디스플레이에 커서 표현 */
        // 아래의 코드는 임시. 추후 직후 프레임과의 초점의 차이로 계산해야함.
        // TODO: 초기값 설정과 직전 프레임간의 오차 계산을 통한 정밀한 커서 표현. 눈으로 다른 곳을 보는 경우 고려 요망.
        circle(display, ORIGIN-(focusPoint*sightWeight/70), 10, Scalar(0, 0, 255), -1, LINE_AA);
        cout << ORIGIN-(focusPoint*sightWeight/100) << endl << endl;
        imshow("Main", display);

        if(waitKey(10)==27){ break; }
    }
    destroyAllWindows();
}