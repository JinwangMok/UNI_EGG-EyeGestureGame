#ifndef __CONFIGURATION_H__
#define __CONFIGURATION_H__

/*------ for MJ ------- */
#ifdef _WIN32
#include <Windows.h>
#define sleep(x) Sleep((x*1000))
/*------ CAMERA_SETTINGS ------- */
#define DEFAULT_WIDTH 1280
#define DEFAULT_HEIGHT 720

/*------ CASCADE_DIR -------*/
#define CASCADE_FACE "haarcascade_frontalface_alt2.xml"
#define CASCADE_EYE "haarcascade_eye.xml"

#endif

/*------ for JW ------- */
#ifdef __APPLE__    
#include <unistd.h>
/*------ CAMERA_SETTINGS ------- */
#define DEFAULT_WIDTH 3840
#define DEFAULT_HEIGHT 2160

/*------ CASCADE_DIR -------*/
#define CASCADE_FACE "./cascade_xmls/haarcascade_frontalface_alt2.xml"
#define CASCADE_EYE "./cascade_xmls/haarcascade_eye.xml"

#endif

/*------ WINDOW_NAME ------ */
#define GAME_IMG	"GAME"
#define WEBCAM_IMG	"Webcam"
#define FACE_IMG	"FACE"

/*------  GAME ------*/
#define GAME_IMG_FILE "GAME_IMAGE.jpg"

/*------  GESTURE ------*/
#define GESTURE_ERROR -1
#define GESTURE_STOP 0
#define GESTURE_UP 1
#define GESTURE_DOWN 2
#define GESTURE_LEFT 3
#define GESTURE_RIGHT 4
#define GESTURE_MARGIN 5

/*------ PARAMETERS ------*/
#define LAMPING_TIME 20
#define CASCADE_FACE_MIN_NEIGHBORS 9
#define CASCADE_EYES_MIN_NEIGHBORS 15

cv::Point ANSWER_POINT1(685, 275);
cv::Point ANSWER_POINT2(715, 335);


class GAME {
	cv::Rect ANSWER;

public:
	void GAME_INIT();
	void GAME_PLAY(cv::Point& CURSUR, cv::Point& CURSUR_EX);
};

class PLAYER {
public:
	void open_cascade();
	void open_camera();
	void lamping_time();
	int detect_gesture(cv::Mat& PLAYER_FOCUS, cv::Mat& GAME_FRAME);
};

/*------ etc ------*/
#define BORDER_LINE "------------------------------"

#endif