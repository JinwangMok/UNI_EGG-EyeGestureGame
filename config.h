#ifndef __CONFIGURATION_H__
#define __CONFIGURATION_H__

/*------ for MJ ------- */
#ifdef _WIN32
#include <Windows.h>
/*------ CAMERA_SETTINGS ------- */
#define DEFAULT_WIDTH 1280
#define DEFAULT_HEIGHT 720

/*------ CASCADE_DIR -------*/
#define CASCADE_FACE "haarcascade_frontalface_alt2.xml"
#define CASCADE_EYE "haarcascade_eye.xml"

#endif

//------ for JW ------- //
#ifdef __APPLE__    
#include <unistd.h>
//------ CAMERA_SETTINGS ------- //
#define DEFAULT_WIDTH 1280
#define DEFAULT_HEIGHT 720

//------ CASCADE_DIR -------//
#define CASCADE_FACE "./cascade_xmls/haarcascade_frontalface_alt2.xml"
#define CASCADE_EYE "./cascade_xmls/haarcascade_eye.xml"

#endif

/*------ WINDOW_NAME ------ */
#define GAME_IMG   "GAME"
#define WEBCAM_IMG   "Webcam"
#define GAME_END   "GAME_END"

/*------  GAME ------*/
#define GAME_IMG_FILE "GAME_IMAGE.jpg"
#define CURSOR_STEP 5
cv::Mat game_end(cv::Size(640, 480), CV_8UC3, cv::Scalar(200, 200, 200));

/*------  GESTURE ------*/
#define GESTURE_ERROR -1
#define GESTURE_STOP 0
#define GESTURE_UP 1
#define GESTURE_DOWN 2
#define GESTURE_LEFT 3
#define GESTURE_RIGHT 4
#define GESTURE_MARGIN 10

/*------ PARAMETERS ------*/
#define LAMPING_TIME 20
#define CASCADE_FACE_MIN_NEIGHBORS 9
#define CASCADE_EYES_MIN_NEIGHBORS 15

#define end_command  "GAME ENDED, press ESC to out."

/*------ COLOR ------*/
#define EYE_COLOR cv::Scalar(0,0,255)
#define FACE_COLOR cv::Scalar(255,0,0)
#define CURSOR_COLOR cv::Scalar(0,255,0)
#define GAME_ANSWER_ZONE cv::Scalar(0,0,255)
#define FONT_COLOR cv::Scalar(0,0,0)

class GAME {
   cv::Rect ANSWER;

public:
   void GAME_INIT();
   bool GAME_PLAY(cv::Point& CURSOR, int GESTURE_CODE);
};

class PLAYER {
public:
   void open_cascade();
   void open_camera();
   void lamping_time();
   int detect_gesture(cv::Mat& PLAYER_FOCUS);
};

/*------ etc ------*/
#define BORDER_LINE "------------------------------"

#endif
