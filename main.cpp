#include <opencv2/opencv.hpp>
#include <iostream>
#include "config.h"

using namespace std;
using namespace cv;

// CASCADE CLASSIFIER
CascadeClassifier face_cascade;
CascadeClassifier eye_cascade;

// CAMERA
VideoCapture cap;

/*----- game -----*/
Mat game_frame;

/*----- player -----*/
Mat player_focus(Size(DEFAULT_WIDTH, DEFAULT_HEIGHT), CV_8UC3);

void GAME::GAME_INIT(){
    /* NOT SUPPORT DISPLAY SETTING FROM USER */
    game_frame = imread(GAME_IMG_FILE, IMREAD_COLOR);
    if (game_frame.empty()) { cout << "Game image loading Error occured." << endl; return; }
    ANSWER = Rect(ANSWER_POINT1, ANSWER_POINT2);
    game_frame(ANSWER) = 0; // Masking Answer
}

void GAME::GAME_PLAY(Point& CURSOR, Point& CURSOR_EX){
    /* GET FOCUS COORDINATE */
    // if (CURSOR.x == 0 && CURSOR.y == 0) {   // CURSOR ==0 -> return
    //     return -1;
    // }
    // else if (CURSOR_ex.x == 0 && CURSOR_ex.y == 0) {    // CURSOR-ex == 0 -> reutrn
    //     return -1;
    // }
    // else {
    //     cv::Point diff = CURSOR - CURSOR_ex;
    //     circle(game_frame, CURSOR - diff*30, 3, CURSOR_COLOR, 2, FILLED);
    // }
}

void PLAYER::open_cascade(){
    face_cascade.load(CASCADE_FACE);
	eye_cascade.load(CASCADE_EYE);

    cout << BORDER_LINE << endl;
    if (face_cascade.empty()) { cout << "Error occured during loading face cascade" << endl; return; };
    if (eye_cascade.empty()) { cout << "Error occured during loading face cascade" << endl; return; };

    cout << "Cascade load successed." << endl;
    cout << BORDER_LINE << endl;
}

void PLAYER::open_camera(){
    cap.open(0);

    cout << BORDER_LINE << endl;
    if (!cap.isOpened()) { cout << "CAM OPEN FAILED" << endl; return; }

    cap.set(CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT);

    cout << "Camera opened successfully." << endl;
    cout << BORDER_LINE << endl;
}

void PLAYER::lamping_time() {

    for (int i = 0; i < LAMPING_TIME; i++) {
        cap >> player_focus;
    }
    cout << "Lamping TIME " << endl << BORDER_LINE << endl;
}

void PLAYER::detect_Eyes(Mat& PLAYER_FOCUS, Mat& GAME_FRAME, queue<Rect>& DETECTED_FACES_QUEUE, queue<Point>& DETECTED_LEFT_EYE_QUEUE, queue<Point>& DETECTED_RIGHT_EYE_QUEUE){
    /*----- INITIALIZATION -----*/
    // FLIPPING & CONVERT2GRAY 
    Mat grayscale;  // PALYER_FOCUS -> GRAYSCALE
    // FACE DETECTION
    vector<Rect> faces;
    Rect face_ROI;
    uint16_t faceErrorCount = 0;
    Point face_ROI_coordinate;
    uint16_t face_width, face_height;
    // SPLIT BOTH EYES ROI
    Rect left_eye_ROI, right_eye_ROI;
    // EYE DETECTION
    vector<Rect> detected_left_eyes, detected_right_eyes;
    Point left_eye_coordinate, right_eye_coordinate;
    uint16_t left_eye_errorCount = 0, right_eye_errorCount = 0;
    // ETC
    Point focus;
    uint16_t eye_count = 0;      // eye ditected count

    /*----- FLIPPING & CONVERT2GRAY -----*/
    flip(PLAYER_FOCUS, PLAYER_FOCUS, 1);  // reverse left, right
    cvtColor(PLAYER_FOCUS, grayscale, COLOR_BGR2GRAY); // convert BGR -> GRAY
    // equalizeHist(grayscale, grayscale); // 보기에는 오히려 눈과 피부색의 명암비 차이가 줄어들어보임
    
    /*----- FACE DETECTION -----*/
    face_cascade.detectMultiScale(grayscale, faces, 1.1, CASCADE_FACE_MIN_NEIGHBORS);
    
    // TODO:프레임 간의 얼굴 영역 마진 처리
    if(faces.size()==1){            // DETECT SINGLE FACE
        face_ROI = faces.at(0);
        if(DETECTED_FACES_QUEUE.size() > CASCADE_FACE_STORAGE_SIZE){
            DETECTED_FACES_QUEUE.pop();
        }
        DETECTED_FACES_QUEUE.push(face_ROI);
    }else{                          // DETECT MULTIPLE FACES
        /* CASCADE_FACE_ERROR_COUNT만큼 오류 처리 후에도 얼굴이 존재하지 않거나 2개 이상이면 가장 최근 얼굴로 강제 초기화 */
        if(faceErrorCount < CASCADE_FACE_ERROR_COUNT){
            if(DETECTED_FACES_QUEUE.size() > 1){
                face_ROI = DETECTED_FACES_QUEUE.back();
                faceErrorCount++;
            }else{
                face_ROI = Rect();
            }
        }else{
            queue<Rect> eraser;
            swap(eraser, DETECTED_FACES_QUEUE); // Erase entire queue.
            face_ROI = Rect();
            faceErrorCount = 0;
        }
    }

    /*----- SPLIT BOTH EYES ROI -----*/
    face_ROI_coordinate = Point(face_ROI.x, face_ROI.y);
    face_width = cvRound(face_ROI.width);
    face_height = cvRound(face_ROI.height/2);
    left_eye_ROI = Rect(face_ROI_coordinate.x, face_ROI_coordinate.y, face_width/2, face_height);
    right_eye_ROI = Rect(face_ROI_coordinate.x+face_width/2, face_ROI_coordinate.y, face_width/2, face_height);
    
    /*----- EYE DETECTION -----*/
    eye_cascade.detectMultiScale(PLAYER_FOCUS(left_eye_ROI), detected_left_eyes, 1.1, CASCADE_EYES_MIN_NEIGHBORS);
    eye_cascade.detectMultiScale(PLAYER_FOCUS(right_eye_ROI), detected_right_eyes, 1.1, CASCADE_EYES_MIN_NEIGHBORS);

    // NOTE: ~~원래 HoughCircles 검출을 추가로 검사하고자 하였으나 성능 문제로 보류. 추후 논문 작성 시 실험 예정.~~ -> 일단 실험 예정. 정확한 원의 중심을 찾기 위함.
    // TODO: 안구 검출 안되는 경우(ex. 눈 감을 때) 예외 처리 필요. PPT 5번 참고
    // LEFT EYE
    if(detected_left_eyes.size()==1){// DETECT SINGLE LEFT EYE
        left_eye_coordinate = Point((detected_left_eyes[0].x + (detected_left_eyes[0].width/2)),
                                    (detected_left_eyes[0].y + (detected_left_eyes[0].height/2)));
        if(DETECTED_LEFT_EYE_QUEUE.size() > CASCADE_LEFT_EYE_ERROR_COUNT){
            DETECTED_LEFT_EYE_QUEUE.pop();
        }
        DETECTED_LEFT_EYE_QUEUE.push(left_eye_coordinate);
    }else{                          // DETECT MULTIPLE LEFT EYES
        /* CASCADE_LEFT_EYE_ERROR_COUNT만큼 오류 처리 후에도 얼굴이 존재하지 않거나 2개 이상이면 가장 최근 얼굴로 강제 초기화 */
        if(left_eye_errorCount < CASCADE_LEFT_EYE_ERROR_COUNT){
            if(DETECTED_LEFT_EYE_QUEUE.size() > 1){
                left_eye_coordinate = DETECTED_LEFT_EYE_QUEUE.back();
                left_eye_errorCount++;
            }else{
                left_eye_coordinate = Point();
            }
        }else{
            queue<Point> eraser;
            swap(eraser, DETECTED_LEFT_EYE_QUEUE); // Erase entire queue.
            left_eye_coordinate = Point();
            left_eye_errorCount = 0;
        }
    }
    // RIGHT EYE
    if(detected_right_eyes.size()==1){// DETECT SINGLE RIGHT EYE
        right_eye_coordinate = Point((detected_right_eyes[0].x + (detected_right_eyes[0].width/2)),
                                    (detected_right_eyes[0].y + (detected_right_eyes[0].height/2)));
        if(DETECTED_RIGHT_EYE_QUEUE.size() > CASCADE_RIGHT_EYE_ERROR_COUNT){
            DETECTED_RIGHT_EYE_QUEUE.pop();
        }
        DETECTED_RIGHT_EYE_QUEUE.push(right_eye_coordinate);
    }else{                          // DETECT MULTIPLE RIGHT EYES
        /* CASCADE_RIGHT_EYE_ERROR_COUNT만큼 오류 처리 후에도 얼굴이 존재하지 않거나 2개 이상이면 가장 최근 얼굴로 강제 초기화 */
        if(right_eye_errorCount < CASCADE_RIGHT_EYE_ERROR_COUNT){
            if(DETECTED_LEFT_EYE_QUEUE.size() > 1){
                right_eye_coordinate = DETECTED_RIGHT_EYE_QUEUE.back();
                right_eye_errorCount++;
            }else{
                right_eye_coordinate = Point();
            }
        }else{
            queue<Point> eraser;
            swap(eraser, DETECTED_RIGHT_EYE_QUEUE); // Erase entire queue.
            right_eye_coordinate = Point();
            right_eye_errorCount = 0;
        }
    }

    cout << BORDER_LINE << endl << "[DETECTED]" << endl;
    cout << "Left Eye Coordinate: " << "x=" << left_eye_ROI.x+left_eye_coordinate.x << "y=" << left_eye_ROI.y+left_eye_coordinate.y << endl;
    cout << "Right Eye Coordinate: " << "x=" << right_eye_ROI.x+right_eye_coordinate.x << "y=" << right_eye_ROI.y+right_eye_coordinate.y << endl;
    cout << BORDER_LINE << endl << endl;
}

int main(int argc, char** argv){
    queue<Rect> detected_faces_queue;
    queue<Point> detected_left_eye_queue, detected_right_eye_queue; // NOTE: queue index -1 is latest one.
    GAME game;
    game.GAME_INIT();

    PLAYER player;
    player.open_cascade();
    player.open_camera();
    player.lamping_time();
    player.detect_Eyes(player_focus, game_frame, detected_faces_queue, detected_left_eye_queue, detected_right_eye_queue);
    // waitKey(0);
    // destroyAllWindows();
    // face_classifier.load("./cascade_xmls/haarcascade_frontalface_alt2.xml");
	// eye_classifier.load("./cascade_xmls/haarcascade_eye.xml");

    // VideoCapture cap(0);

	// if (face_classifier.empty() || eye_classifier.empty() || !cap.isOpened()){
	// 	cerr << "Something load failed!" << endl;
	// 	return -1;
	// }
    
    // /* 디스플레이 크기는 OS마다 다르다고 함. 확인 필요! */
    // Mat display(DISPLAY_HEIGHT, DISPLAY_WIDTH, CV_8UC3, Scalar::all(255));
    // Point ORIGIN(cvRound(display.cols/2), cvRound(display.rows/2));

    // Mat frame, leftEyeROI, rightEyeROI;
    // Rect faceROI;
    // vector<Rect> eyesFromLeft, eyesFromRight;
    // queue<Rect> latest_faces; // index -1 : latest face rect
    // uint16_t faceErrorCount = 0, eyeErrorCount = 0;
    // Point leftCenter, rightCenter, focusPoint;
    // double sightWeight = 0;

    // //TODO: 얼굴 검출 안된 경우 예외 처리(지금은 강제 종료됨)
    // while(true){
    //     Mat gray_frame;
    //     vector<Rect> faces;

    //     cap >> frame;
    //     if(frame.empty()){ break; }

    //     flip(frame, frame, 1);// 좌우반전
    //     cvtColor(frame, gray_frame, COLOR_BGR2GRAY);// 계산을 위해 그레이 스케일로 변환

    //     face_classifier.detectMultiScale(gray_frame, faces, 1.1, FACE_MIN_NEIGHBORS);// scaleFactor=1.1, minNeighbors=9

    //     /* 얼굴 선택기(예외 발생 시 가장 최근 선택된 얼굴로 강제 대체) */
    //     // TODO: 프레임 간의 얼굴 영역 마진 처리
    //     if(faces.size()==1){
    //         faceROI = faces.at(0);
    //         if(latest_faces.size() < FACE_STOREGE_SIZE){
    //             latest_faces.push(faceROI);
    //         }else{
    //             latest_faces.pop();
    //             latest_faces.push(faceROI);
    //         }
    //     }else{
    //         /* FACE_ERROR_MARGIN만큼 오류 처리 후에도 얼굴이 존재하지 않는다면 강제 초기화 */
    //         if(faceErrorCount < FACE_ERROR_MARGIN){
    //             if(latest_faces.size() > 1){
    //                 faceROI = latest_faces.back();
    //                 faceErrorCount++;
    //             }else{
    //                 faceROI = Rect();
    //             }
    //         }else{
    //             queue<Rect> eraser;
    //             swap(eraser, latest_faces); // 얼굴 저장 배열 초기화
    //             faceROI = Rect();
    //             faceErrorCount = 0;
    //         }
    //     }

    //     // rectangle(frame, face, Scalar(255, 0, 255), 2); 얼굴 상자 표시
        
    //     /* 얼굴 영역 상하 분할 후 상부 선택 */
    //     faceROI.height = cvRound(faceROI.height/2);
    //     leftEyeROI = frame(Rect(faceROI.x, faceROI.y, cvRound(faceROI.width/2), faceROI.height));
    //     rightEyeROI = frame(Rect(faceROI.x+cvRound(faceROI.width/2), faceROI.y, cvRound(faceROI.width/2), faceROI.height));
        
    //     eye_classifier.detectMultiScale(leftEyeROI, eyesFromLeft, 1.1, EYES_MIN_NEIGHBORS); // scaleFactor=1.1
    //     eye_classifier.detectMultiScale(rightEyeROI, eyesFromRight, 1.1, EYES_MIN_NEIGHBORS); // scaleFactor=1.1
        
    //     /* 양안의 동공 검출 */
    //     // TODO: HoughCircles 사용해야 함. HoubhCircles(InputArray, OutputArray, int method, double dp_ratio, double minDist최소거리, 
    //     //                                           int param1=100캐니에지 높은 임계값, int param=2축적 배열에서 원 검출을 위한 임계값, intminRadius, int maxRadius) >> 9-5.cpp참고
    //     // ⭐️12.08 16:17) 허프 검출이든, 하르 검출이든 둘중에 하나만 하자고 통합함. 그 중 하르 검출로 의견을 모았음. 대신, IPIU 논문쓸 때 두개를 실험해서 내는 걸로!
    //     // TODO: 안구 검출 안되는 경우(ex. 눈 감을 때) 예외 처리. PPT 알고리즘 5번 참고
    //     for (Rect eye : eyesFromLeft) {
    //         leftCenter = Point(eye.x + eye.width / 2, eye.y + eye.height / 2);
    //         circle(frame(faceROI), leftCenter, 3, Scalar(0, 255, 255), -1, LINE_AA);
    //     }

    //     for (Rect eye : eyesFromRight) {
    //         rightCenter = Point((eye.x + eye.width / 2)+leftEyeROI.cols, eye.y + eye.height / 2);
    //         circle(frame(faceROI), rightCenter, 3, Scalar(0, 255, 255), -1, LINE_AA);
    //     }

    //     /* 초점 및 시야각 가중치 계산 */
    //     focusPoint = Point(((rightCenter.x-leftCenter.x)/2)+leftCenter.x, // ((r-l)/2) + l
    //                         leftCenter.y<rightCenter.y?((leftCenter.y-rightCenter.y)/2)+leftCenter.y:((rightCenter.y-leftCenter.y)/2)+rightCenter.y); 
    //     sightWeight = norm(rightCenter-leftCenter);
        
    //     /* 가중치 변화 확인 */
    //     // String norm_val = to_string(sightWeight);
    //     // putText(frame, norm_val, Point(10, 30), 2, 1, Scalar(0, 0, 255));

    //     circle(frame(faceROI), focusPoint, 3, Scalar(0, 255, 255), -1, LINE_AA);
    //     line(frame(faceROI), leftCenter, rightCenter, Scalar(0, 0, 255), 1, LINE_AA);

    //     imshow("frame", frame);

    //     /* 디스플레이에 커서 표현 */
    //     // 아래의 코드는 임시. 추후 직후 프레임과의 초점의 차이로 계산해야함.
    //     // TODO: 초기값 설정과 직전 프레임간의 오차 계산을 통한 정밀한 커서 표현. 눈으로 다른 곳을 보는 경우 고려 요망.
    //     circle(display, ORIGIN-(focusPoint*sightWeight/70), 10, Scalar(0, 0, 255), -1, LINE_AA);
    //     cout << ORIGIN-(focusPoint*sightWeight/100) << endl << endl;
    //     imshow("Main", display);

    //     if(waitKey(10)==27){ break; }
    // }
    // destroyAllWindows();
}