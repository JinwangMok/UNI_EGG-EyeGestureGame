#ifndef __CONFIGURATION_H__
#define __CONFIGURATION_H__

/*------ for MJ ------- */
#ifdef _WIN32
/*------ CAMERA_SETTINGS ------- */
#define DEFAULT_WIDTH 1280
#define DEFAULT_HEIGHT 720

/*------ CASCADE_DIR -------*/
#define CASCADE_FACE "haarcascade_frontalface_alt2.xml"
#define CASCADE_EYE "haarcascade_eye.xml"

#endif

/*------ for JW ------- */
#ifdef __APPLE__    
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

/*------ INITALIZING ------*/
#define LAMPING_TIME 20
#define CASCADE_FACE_STORAGE_SIZE 30
#define CASCADE_FACE_ERROR_COUNT 30
#define CASCADE_FACE_MIN_NEIGHBORS 9
#define CASCADE_EYES_MIN_NEIGHBORS 15
#define CASCADE_LEFT_EYE_ERROR_COUNT 30
#define CASCADE_RIGHT_EYE_ERROR_COUNT 30

/*------  COLOR ------*/
#define EYE_COLOR Scalar(0,0,255)
#define CURSOR_COLOR Scalar(0, 0,255)

/*------  GAME ------*/
#define GAME_IMG_FILE "GAME_IMAGE.jpg"

cv::Point ANSWER_POINT1(685, 275);
cv::Point ANSWER_POINT2(715, 335);


class GAME {
	cv::Rect ANSWER;


public:
	void GAME_INIT();
	void GAME_PLAY(cv::Point& CURSUR, cv::Point& CURSUR_EX);
};

class PLAYER {
private:
	cv::Point focus_point;
	double perspective_weight;
public:
	void set_focus_point(cv::Point focus){ this->focus_point = focus; }
	cv::Point get_focus_point(){ return this->focus_point; }
	void set_perspective_weight(double weight){ this->perspective_weight = weight; }
	double get_perspective_weight(){ return this->perspective_weight; }
public:
	void open_cascade();
	void open_camera();
	void lamping_time();
	void detect_Eyes(cv::Mat& PLAYER_FOCUS, cv::Mat& GAME_FRAME, cv::Point* EYES_COORDINATE, std::queue<cv::Rect>& DETECTED_FACES_QUEUE, std::queue<cv::Point>& DETECTED_LEFT_EYE_QUEUE, std::queue<cv::Point>& DETECTED_RIGHT_EYE_QUEUE);
};

/*------ etc ------*/
#define BORDER_LINE "------------------------------"

#endif