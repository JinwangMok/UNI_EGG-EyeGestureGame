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
Point answer_pt1, answer_pt2;

/*----- player -----*/
Mat player_webCAM(Size(DEFAULT_WIDTH, DEFAULT_HEIGHT), CV_8UC3);

void GAME::GAME_INIT() {
    /* NOT SUPPORT DISPLAY SETTING FROM USER */
    game_frame = imread(GAME_IMG_FILE, IMREAD_COLOR);
    Size game_size = game_frame.size();
    answer_pt1 = Point(cvRound(game_size.width * 0.614), cvRound(game_size.height * 0.368));
    answer_pt2 = Point(cvRound(game_size.width * 0.625), cvRound(game_size.height * 0.405));
    if (game_frame.empty()) { cout << "Game image loading Error occured." << endl; return; }
    ANSWER = Rect(answer_pt1, answer_pt2);
    rectangle(game_frame, ANSWER, GAME_ANSWER_ZONE, 3, LINE_AA);
}

bool GAME::GAME_PLAY(Point& CURSOR, int GESTURE_CODE) {
    switch (GESTURE_CODE) {
    case GESTURE_STOP:
        return false;
        break;
    case GESTURE_UP:
        CURSOR.y -= CURSOR_STEP;
        break;
    case GESTURE_DOWN:
        CURSOR.y += CURSOR_STEP;
        break;
    case GESTURE_LEFT:
        CURSOR.x -= CURSOR_STEP;
        break;
    case GESTURE_RIGHT:
        CURSOR.x += CURSOR_STEP;
        break;
    case GESTURE_ERROR:
        return false;
        break;
    default:
        break;
    }

    if (CURSOR.x <= answer_pt2.x && CURSOR.x >= answer_pt1.x &&
        CURSOR.y <= answer_pt2.y && CURSOR.y >= answer_pt1.y) {

        while (true) {
            putText(game_end, end_command,  Point(game_end.cols/4, game_end.rows/2), FONT_HERSHEY_COMPLEX, 0.6, FONT_COLOR);
            imshow(GAME_END, game_end);
            if (waitKey(10) == 27) { break; }
        }
        return true;
    }
    else {
        return false;
    }
}

void PLAYER::open_cascade() {
    face_cascade.load(CASCADE_FACE);
    eye_cascade.load(CASCADE_EYE);

    cout << BORDER_LINE << endl;
    if (face_cascade.empty()) { cout << "Error occured during loading face cascade" << endl; return; };
    if (eye_cascade.empty()) { cout << "Error occured during loading face cascade" << endl; return; };

    cout << "Cascade load successed." << endl;
    cout << BORDER_LINE << endl;
}

void PLAYER::open_camera() {
    cap.open(0);

    cout << BORDER_LINE << endl;
    if (!cap.isOpened()) { cout << "CAM OPEN FAILED" << endl; return; }

    cout << "Camera opened successfully." << endl;
    cout << BORDER_LINE << endl;
}

void PLAYER::lamping_time() {

    for (int i = 0; i < LAMPING_TIME; i++) {
        cap >> player_webCAM;
    }
    cout << "Lamping TIME " << endl << BORDER_LINE << endl;
}

int PLAYER::detect_gesture(Mat& PLAYER) {
    /*----- Variables -----*/
    // FLIPPING & CONVERT2GRAY 
    Mat grayscale;  // PALYER_FOCUS -> GRAYSCALE
    // FACE DETECTION
    vector<Rect> faces;
    Rect face_ROI;
    Point face_ROI_coordinate;
    uint16_t face_width, face_height;
    // SPLIT BOTH EYES ROI
    Rect left_eye_ROI, right_eye_ROI;
    // EYE DETECTION
    vector<Rect> detected_left_eyes, detected_right_eyes;
    Rect left_eye_rect, right_eye_rect;
    Point left_eye_center, right_eye_center;
    bool left_eye_closed = false, right_eye_closed = false;

    /*----- FLIPPING & CONVERT2GRAY -----*/
    flip(PLAYER, PLAYER, 1);  // reverse left, right
    cvtColor(PLAYER, grayscale, COLOR_BGR2GRAY); // convert BGR -> GRAY

    /*----- FACE DETECTION -----*/
    face_cascade.detectMultiScale(grayscale, faces, 1.1, CASCADE_FACE_MIN_NEIGHBORS);

    if (faces.size() == 1) {            // DETECT SINGLE FACE
        face_ROI = faces.at(0);
        face_ROI_coordinate = Point(face_ROI.x, face_ROI.y);
        face_width = cvRound(face_ROI.width);
        face_height = cvRound(face_ROI.height / 2);
    }
    else {                          // DETECT MULTIPLE FACES OR NOTTING
        return GESTURE_STOP;
    }

    rectangle(PLAYER, face_ROI, FACE_COLOR, 3, LINE_AA);

    /*----- SPLIT BOTH EYES ROI -----*/
    left_eye_ROI = Rect(face_ROI_coordinate.x, face_ROI_coordinate.y, face_width / 2, face_height);
    right_eye_ROI = Rect(face_ROI_coordinate.x + face_width / 2, face_ROI_coordinate.y, face_width / 2, face_height);

    /*----- EYE DETECTION -----*/
    eye_cascade.detectMultiScale(PLAYER(left_eye_ROI), detected_left_eyes, 1.1, CASCADE_EYES_MIN_NEIGHBORS);
    eye_cascade.detectMultiScale(PLAYER(right_eye_ROI), detected_right_eyes, 1.1, CASCADE_EYES_MIN_NEIGHBORS);

    // LEFT EYE
    if (detected_left_eyes.size() == 1) {// DETECT SINGLE LEFT EYE
        left_eye_rect = detected_left_eyes[0];
        left_eye_center = Point((left_eye_rect.x + (left_eye_rect.width / 2)),
            (left_eye_rect.y + (left_eye_rect.height / 2)));
    }
    else if (detected_left_eyes.size() == 0) {
        left_eye_closed = true;
    }
    else {
        return GESTURE_STOP; // MULTIPLE LEFT EYES DETECTED ERROR
    }

    // RIGHT EYE
    if (detected_right_eyes.size() == 1) {// DETECT SINGLE LEFT EYE
        right_eye_rect = detected_right_eyes[0];
        right_eye_center = Point((right_eye_rect.x + (right_eye_rect.width / 2)),
            (right_eye_rect.y + (right_eye_rect.height / 2)));
    }
    else if (detected_right_eyes.size() == 0) {
        right_eye_closed = true;
    }
    else {
        return GESTURE_STOP; // MULTIPLE RIGHT EYES DETECTED ERROR
    }
    Point left_eye_point(left_eye_ROI.x + left_eye_center.x, left_eye_ROI.y + left_eye_center.y);
    Point right_eye_point(right_eye_ROI.x + right_eye_center.x, right_eye_ROI.y + right_eye_center.y);
    circle(PLAYER, left_eye_point, 5, EYE_COLOR, -1, LINE_AA);
    circle(PLAYER, right_eye_point, 5, EYE_COLOR, -1, LINE_AA);

    /* RETURN GESTURES */
    if (left_eye_closed && !right_eye_closed) {
        return GESTURE_RIGHT; // 왼쪽 눈 윙크 시 오른쪽으로 이동
    }
    else if (!left_eye_closed && right_eye_closed) {
        return GESTURE_LEFT;  // 오른쪽 눈 윙크 시 왼쪽으로 이동
    }
    else if (!left_eye_closed && !right_eye_closed) { // 양쪽 눈 모두 뜬 상태
        if (left_eye_center.y + GESTURE_MARGIN < right_eye_center.y) {        // 화면 상에서 오른쪽 눈이 더 아래에 있을 때(==오른쪽으로 머리를 기울일 때)
            return GESTURE_DOWN;
        }
        else if (right_eye_center.y + GESTURE_MARGIN < left_eye_center.y) {  // 화면 상에서 왼쪽 눈이 더 아래에 있을 때(==왼쪽으로 머리를 기울일 때)
            return GESTURE_UP;
        }
        else {
            return GESTURE_STOP;
        }
    }

    // ELSE
    return GESTURE_ERROR;
}

int main(int argc, char** argv) {
    /*----- variables -----*/
    int gesture;
    bool game_finished = false;

    GAME game;
    game.GAME_INIT();

    PLAYER player;
    player.open_cascade();
    player.open_camera();
    player.lamping_time();

    Point cursor(game_frame.cols / 2, game_frame.rows / 2);
    Mat game_window;
    while (true) {
        cap >> player_webCAM;
        gesture = player.detect_gesture(player_webCAM);

        game_finished = game.GAME_PLAY(cursor, gesture);
        game_window = game_frame.clone();
        circle(game_window, cursor, 20, CURSOR_COLOR, -1, LINE_AA);
        String gesture_str;
        switch (gesture) {

        case 0:
            gesture_str = "STOP";
            break;
        case 1:
            gesture_str = "UP";
            break;
        case 2:
            gesture_str = "DOWN";
            break;
        case 3:
            gesture_str = "LEFT";
            break;
        case 4:
            gesture_str = "RIGHT";
            break;
        default:
            break;
        }

        putText(player_webCAM, gesture_str, Point(10, 30), 4, 1, Scalar(0, 0, 255));

        imshow("Player", player_webCAM);
        imshow("Game", game_window);
        if (waitKey(10) == 27 || game_finished) { break; }
    }

    destroyAllWindows();
}
