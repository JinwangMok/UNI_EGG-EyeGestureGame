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

/*------ PARAMETERS ------*/
#define EYE_CONTRAST_THRESHOLD 25
#define EYE_CONTRAST_WEIGHT 2.f
#define INITIALIZATION_FRAME_NUM 30

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
	std::vector<cv::Point> INITIAL_FOCUS_POINTS; 		// (TL, TR, BR, BL) * 2
	std::vector<double> INITIAL_PERSPECTIVE_WEIGHTS; 	// (TL, TR, BR, BL) * 2
	std::vector<cv::Rect> INITIAL_FACES; 				// (TL, TR, BR, BL) * 2
	std::vector<cv::Rect> INITIAL_LEFT_EYES; 			// (TL, TR, BR, BL) * 2
	std::vector<cv::Rect> INITIAL_RIGHT_EYES; 			// (TL, TR, BR, BL) * 2
	cv::Point focus_point;
	double perspective_weight;
public:
	/* Getter & Setter functions */
	void set_focus_point(cv::Point focus){ this->focus_point = focus; }
	cv::Point get_focus_point(){ return this->focus_point; }

	void set_perspective_weight(double weight){ this->perspective_weight = weight; }
	double get_perspective_weight(){ return this->perspective_weight; }

	void push_to_initial_focus_points(cv::Point focus_point){ this->INITIAL_FOCUS_POINTS.push_back(focus_point);}
	std::vector<cv::Point> get_initial_focus_points(){ return this->INITIAL_FOCUS_POINTS; }

	void push_to_initial_perspective_weights(double perspective_weight){ this->INITIAL_PERSPECTIVE_WEIGHTS.push_back(perspective_weight);}
	std::vector<double> get_initial_perspective_weights(){ return this->INITIAL_PERSPECTIVE_WEIGHTS; }

	void push_to_initial_faces(cv::Rect face){ this->INITIAL_FACES.push_back(face); }
	std::vector<cv::Rect> get_initial_faces(){ return this->INITIAL_FACES; }
	
	void push_to_initial_left_eyes(cv::Rect left_eye){ this->INITIAL_LEFT_EYES.push_back(left_eye); }
	std::vector<cv::Rect> get_initial_left_eyes(){ return this->INITIAL_LEFT_EYES; }
	
	void push_to_initial_right_eyes(cv::Rect right_eye){ this->INITIAL_RIGHT_EYES.push_back(right_eye); }
	std::vector<cv::Rect> get_initial_right_eyes(){ return this->INITIAL_RIGHT_EYES; }

	void initialize_members(){ this->set_focus_point(cv::Point()); this->set_perspective_weight(0.);}
public:
	void open_cascade();
	void open_camera();
	void lamping_time();
	// void get_face_color(cv::Mat& PLAYER_FOCUS, cv::Rect& FACE_ROI);
	void detect_Eyes(cv::Mat& PLAYER_FOCUS, cv::Mat& GAME_FRAME, cv::Point* EYES_COORDINATE, std::queue<cv::Rect>& DETECTED_FACES_QUEUE, std::queue<cv::Rect>& DETECTED_LEFT_EYE_QUEUE, std::queue<cv::Rect>& DETECTED_RIGHT_EYE_QUEUE);
	void initial_setup(cv::Mat& PLAYER_FOCUS, cv::Mat& GAME_FRAME, cv::Point* EYES_COORDINATE, std::queue<cv::Rect>& DETECTED_FACES_QUEUE, std::queue<cv::Rect>& DETECTED_LEFT_EYE_QUEUE, std::queue<cv::Rect>& DETECTED_RIGHT_EYE_QUEUE);
};

/*------ etc ------*/
#define BORDER_LINE "------------------------------"

#endif