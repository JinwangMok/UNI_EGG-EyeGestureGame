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
// void PLAYER::get_face_color(Mat& PLAYER_FOCUS, Rect& FACE_ROI){
//     // NOTE: 여기에서 얼굴에 해당하는 영역의 히스토그램을 통해 가장 주요한 색상을 추출하고 멤버로 저장한다.
//     Mat face_for_hist, hist[3];
//     vector<Mat> face_bgr;

//     face_for_hist = PLAYER_FOCUS(FACE_ROI).clone();
//     split(face_for_hist, face_bgr);

//     for(int i = 0; i < 3; i++){
//         int channels[] = { i };
//         int bins = 128;
//         int histSize[] = { bins };
//         float range[] = { 0, 256 };
//         const float* ranges[] = { range };
//         calcHist(&face_bgr[i], 1, channels, Mat(), hist[i], 1, histSize, ranges);
//         cout << hist[i] << endl;
//     }
    
// }

void PLAYER::detect_Eyes(Mat& PLAYER_FOCUS, Mat& GAME_FRAME, Point* EYES_COORDINATE, queue<Rect>& DETECTED_FACES_QUEUE, queue<Rect>& DETECTED_LEFT_EYE_QUEUE, queue<Rect>& DETECTED_RIGHT_EYE_QUEUE){
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
    Rect left_eye_rect, right_eye_rect;
    uint16_t left_eye_errorCount = 0, right_eye_errorCount = 0;
    //CALCULATE ACCULATE COORDINATE
    Point left_eye_coordinate, right_eye_coordinate;
    

    //UPDATE VARIABLES
    uint16_t left_eye_x, left_eye_y, right_eye_x, right_eye_y;

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
            this->initialize_members();
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
        // left_eye_coordinate = Point((detected_left_eyes[0].x + (detected_left_eyes[0].width/2)),
        //                             (detected_left_eyes[0].y + (detected_left_eyes[0].height/2)));
        left_eye_rect = detected_left_eyes[0];
        if(DETECTED_LEFT_EYE_QUEUE.size() > CASCADE_LEFT_EYE_ERROR_COUNT){
            DETECTED_LEFT_EYE_QUEUE.pop();
        }
        // DETECTED_LEFT_EYE_QUEUE.push(left_eye_coordinate);
        DETECTED_LEFT_EYE_QUEUE.push(left_eye_rect);
    }else{                          // DETECT MULTIPLE LEFT EYES
        /* CASCADE_LEFT_EYE_ERROR_COUNT만큼 오류 처리 후에도 얼굴이 존재하지 않거나 2개 이상이면 가장 최근 얼굴로 강제 초기화 */
        if(left_eye_errorCount < CASCADE_LEFT_EYE_ERROR_COUNT){
            if(DETECTED_LEFT_EYE_QUEUE.size() > 1){
                // left_eye_coordinate = DETECTED_LEFT_EYE_QUEUE.back();
                left_eye_rect = DETECTED_LEFT_EYE_QUEUE.back();
                left_eye_errorCount++;
            }else{
                // left_eye_coordinate = Point();
                left_eye_rect = Rect();
            }
        }else{
            queue<Rect> eraser;
            swap(eraser, DETECTED_LEFT_EYE_QUEUE); // Erase entire queue.
            this->initialize_members();
            // left_eye_coordinate = Point();
            left_eye_rect = Rect();
            left_eye_errorCount = 0;
        }
    }
    // RIGHT EYE
    if(detected_right_eyes.size()==1){// DETECT SINGLE RIGHT EYE
        // right_eye_coordinate = Point((detected_right_eyes[0].x + (detected_right_eyes[0].width/2)),
        //                             (detected_right_eyes[0].y + (detected_right_eyes[0].height/2)));
        right_eye_rect = detected_right_eyes[0];
        if(DETECTED_RIGHT_EYE_QUEUE.size() > CASCADE_RIGHT_EYE_ERROR_COUNT){
            DETECTED_RIGHT_EYE_QUEUE.pop();
        }
        // DETECTED_RIGHT_EYE_QUEUE.push(right_eye_coordinate);
        DETECTED_RIGHT_EYE_QUEUE.push(right_eye_rect);
    }else{                          // DETECT MULTIPLE RIGHT EYES
        /* CASCADE_RIGHT_EYE_ERROR_COUNT만큼 오류 처리 후에도 얼굴이 존재하지 않거나 2개 이상이면 가장 최근 얼굴로 강제 초기화 */
        if(right_eye_errorCount < CASCADE_RIGHT_EYE_ERROR_COUNT){
            if(DETECTED_LEFT_EYE_QUEUE.size() > 1){
                // right_eye_coordinate = DETECTED_RIGHT_EYE_QUEUE.back();
                right_eye_rect = DETECTED_RIGHT_EYE_QUEUE.back();
                right_eye_errorCount++;
            }else{
                // right_eye_coordinate = Point();
                right_eye_rect = Rect();
            }
        }else{
            queue<Rect> eraser;
            swap(eraser, DETECTED_RIGHT_EYE_QUEUE); // Erase entire queue.
            this->initialize_members();
            // right_eye_coordinate = Point();
            right_eye_rect = Rect();
            right_eye_errorCount = 0;
        }
    }
    /* CALCULATE ACCULATE COORDINATE */ 
    // 여기에서 Point left_eye_coordinate, right_eye_coordinate 구해야함!
    // face_ROI, left_eye_rect, right_eye_rect
    Mat left_eye_grayscale, right_eye_grayscale;
    Mat left_eye_binary, right_eye_binary;
    // 좌표 이동
    left_eye_rect += Point(left_eye_ROI.x, left_eye_ROI.y);
    right_eye_rect += Point(right_eye_ROI.x, right_eye_ROI.y);
    // 영역 선택
    left_eye_grayscale = PLAYER_FOCUS(left_eye_rect).clone();
    right_eye_grayscale = PLAYER_FOCUS(right_eye_rect).clone();
    // 그레이스케일 변환
    cvtColor(left_eye_grayscale, left_eye_grayscale, COLOR_BGR2GRAY);
    cvtColor(right_eye_grayscale, right_eye_grayscale, COLOR_BGR2GRAY);
    // 명암비 증가 (threshold=25, weight=2.f)
    left_eye_grayscale += (left_eye_grayscale - EYE_CONTRAST_THRESHOLD) * EYE_CONTRAST_WEIGHT;
    right_eye_grayscale += (right_eye_grayscale - EYE_CONTRAST_THRESHOLD) * EYE_CONTRAST_WEIGHT;
    // 이진화
    threshold(left_eye_grayscale, left_eye_binary, 0, 255, THRESH_BINARY | THRESH_OTSU | THRESH_BINARY_INV);
    threshold(right_eye_grayscale, right_eye_binary, 0, 255, THRESH_BINARY | THRESH_OTSU | THRESH_BINARY_INV);
    // 모폴로지
    morphologyEx(left_eye_binary, left_eye_binary, MORPH_CLOSE, Mat());
    morphologyEx(right_eye_binary, right_eye_binary, MORPH_CLOSE, Mat());

    // 좌동공의 우측끝 탐색(테스트중)
    Point left_eye_center(left_eye_binary.cols/2, left_eye_binary.rows/2);
    // cout << left_eye_binary.at<bool>(left_eye_center) << endl;

    uint16_t margin_count = 0;
    // TODO: while문 내부 수정 필요! binary_inverse.
    // while(true){
    //     if(!left_eye_binary.at<bool>(left_eye_center)){
    //         // white pixel 0
    //         if(margin_count > 3){ // 10 pixel 동안 하얀색이면
    //             break;
    //         }else{
    //             left_eye_center.x++;//단순 증가 보다 추후 조건후 증가가 필요할듯
    //             margin_count++;
    //         }
    //     }else{
    //         //black pixel 1
    //         left_eye_center.x++;
    //     }
    // }

    cvtColor(left_eye_binary, left_eye_binary, COLOR_GRAY2BGR);
    circle(left_eye_binary, left_eye_center, 1, Scalar(0, 0, 255), -1, LINE_AA);


    imshow("left_bin", left_eye_binary);

    /* UPDATE VARIABLES */
    // left_eye_x = left_eye_ROI.x+left_eye_coordinate.x;
    // left_eye_y = left_eye_ROI.y+left_eye_coordinate.y;
    // right_eye_x = right_eye_ROI.x+right_eye_coordinate.x;
    // right_eye_y = right_eye_ROI.y+right_eye_coordinate.y;

    // *(EYES_COORDINATE) = Point(left_eye_x, left_eye_y);     // LEFT EYE
    // *(EYES_COORDINATE+1) = Point(right_eye_x, right_eye_y); // RIGHT EYE

    // this->set_focus_point(Point(((right_eye_x-left_eye_x)/2)+left_eye_x,
    //                             left_eye_y<right_eye_y?((left_eye_y-right_eye_y)/2)+left_eye_y:((right_eye_y-left_eye_y)/2)+right_eye_y));
    // this->set_perspective_weight(norm(Point(right_eye_x, right_eye_y)-Point(left_eye_x, left_eye_y)));

    // cout << BORDER_LINE << endl << "[DETECTED]" << endl;
    // cout << "Left Eye Coordinate: " << "x=" << left_eye_ROI.x+left_eye_coordinate.x << "y=" << left_eye_ROI.y+left_eye_coordinate.y << endl;
    // cout << "Right Eye Coordinate: " << "x=" << right_eye_ROI.x+right_eye_coordinate.x << "y=" << right_eye_ROI.y+right_eye_coordinate.y << endl;
    // cout << BORDER_LINE << endl << endl;
}

int main(int argc, char** argv){
    /*----- variables -----*/
    queue<Rect> detected_faces_queue, detected_left_eye_queue, detected_right_eye_queue;
    Point eyes_coordinate[2]; //[0]:Left, [1]:Right

    GAME game;
    game.GAME_INIT();

    PLAYER player;
    player.open_cascade();
    player.open_camera();
    player.lamping_time();
    // player.detect_Eyes(player_focus, game_frame, eyes_coordinate, detected_faces_queue, detected_left_eye_queue, detected_right_eye_queue);
        
    double pw = player.get_perspective_weight();
    Point fp = player.get_focus_point();
    
    circle(player_focus, fp, 3, Scalar(0, 0, 255), -1, LINE_AA);

    imshow("Player", player_focus);
    while(true){
        cap >> player_focus;
        player.detect_Eyes(player_focus, game_frame, eyes_coordinate, detected_faces_queue, detected_left_eye_queue, detected_right_eye_queue);
        
        double pw = player.get_perspective_weight();
        Point fp = player.get_focus_point();
        
        // circle(player_focus, fp, 3, Scalar(0, 0, 255), -1, LINE_AA);
        cvtColor(player_focus, player_focus, COLOR_BGR2GRAY);
        imshow("Player", player_focus);
        if(waitKey(10)==27){ break; }
    }
    // waitKey(0);
    destroyAllWindows();
}