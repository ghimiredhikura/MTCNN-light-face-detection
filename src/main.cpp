//author: samylee
//date: 2018.4.19

#include <time.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "network.h"
#include "mtcnn.h"

#include <sys/stat.h>
#include <sys/time.h>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

enum DATASET { AFW, PASCAL, FDDB, WIDER_TEST, WIDER_VAL};
enum MODE {WEBCAM, BENCHMARK_EVALUATION, IMAGE, IMAGE_LIST};

typedef struct _tagMTCNNResult {
	cv::Rect r;
	float score;
}MTCNNResult;

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

std::vector<MTCNNResult> mtcnnDetection(mtcnn& _find, cv::Mat image, int min_img_size)
{
	std::vector<MTCNNResult> m_results;

	//detect face by min_size(min_img_size)
	std::vector<struct Bbox> detBox = _find.findFace(image, min_img_size);

	for (std::vector<struct Bbox>::iterator it = detBox.begin(); it != detBox.end(); it++) 
	{
		if ((*it).exist) 
		{
			float det_score = (*it).score;
			if(det_score > 0.7) {
				rectangle(image, Point((*it).y1, (*it).x1), Point((*it).y2, (*it).x2), Scalar(0, 0, 255), 2, 8, 0);
				for (int num = 0; num<5; num++)circle(image, Point((int)*(it->ppoint + num), (int)*(it->ppoint + num + 5)), 3, Scalar(0, 255, 255), -1);
			}

			cv::Rect r1(Point((*it).y1, (*it).x1), Point((*it).y2, (*it).x2));
			
			MTCNNResult result;
			result.r = r1;
			result.score = det_score;

			m_results.push_back(result);
		}
	}

	return m_results;
}


void run_afw_pascal_fddb(DATASET m_data)
{
	std::string str_img_file;
	if(m_data == AFW) str_img_file = "../detections/AFW/afw_img_list.txt";
	else if (m_data == PASCAL) str_img_file = "../detections/PASCAL/pascal_img_list.txt";
	else if (m_data == FDDB) str_img_file = "../detections/FDDB/fddb_img_list.txt";	
	else return ;

	std::ifstream inFile(str_img_file.c_str(), std::ifstream::in);

	std::vector<string> image_list;
	std::string imname;
	while(std::getline(inFile, imname)) 
	{
		image_list.push_back(imname);
	}

	std::string str_out_file;
	if(m_data == AFW) str_out_file = "../detections/AFW/mtcnn_afw_dets.txt";
	else if (m_data == PASCAL) str_out_file = "../detections/PASCAL/mtcnn_pascal_dets.txt";
	else if (m_data == FDDB) str_out_file = "../detections/FDDB/mtcnn_fddb_dets.txt";	

	std::ofstream outFile(str_out_file.c_str());
	// process each image one by one
	for(int i = 0; i < image_list.size(); i++)
	{
		//initial models without image's width or height
		mtcnn find;

		std::string imname = image_list[i];
		std::string tempname = imname;

		if(m_data==AFW) imname = "../../../SFD_ROOT/datasets/AFW/testimages/"+imname+".jpg";
		else if(m_data==PASCAL) imname = "../../../SFD_ROOT/datasets/PASCAL/VOCdevkit/VOC2012/JPEGImages/"+imname;
		else if(m_data==FDDB) imname = "../../../SFD_ROOT/datasets/FDDB/"+imname+".jpg";

		cout << "processing image " << i+1 << "/" << image_list.size() << " [" << imname.c_str() << "]" << endl;

		cv::Mat image = cv::imread(imname);

		std::vector<MTCNNResult> mtcnn_results = mtcnnDetection(find, image, 30);

		if(m_data!=FDDB) 
		{			
			for(int j=0;j<mtcnn_results.size();j++) {
				outFile << tempname << " " << mtcnn_results[j].score << " " << mtcnn_results[j].r.x << " "
					<< mtcnn_results[j].r.y << " " << mtcnn_results[j].r.x+mtcnn_results[j].r.width << " " 
					<< mtcnn_results[j].r.y+mtcnn_results[j].r.height << endl;		
			}			
		} else {
			
			outFile << tempname << "\n";
			outFile << mtcnn_results.size() << "\n";
			for(int j =0; j < mtcnn_results.size(); j++) {
				outFile << mtcnn_results[j].r.x << " " << mtcnn_results[j].r.y << " " 
				<< mtcnn_results[j].r.width << " " << mtcnn_results[j].r.height << " " << mtcnn_results[j].score << "\n";
			}
		}

		imshow("test", image);

		waitKey(1);
	}
	outFile.close();
}

void make_wider_dir(DATASET m_data, std::string& result_dir)
{
	std::vector<std::string> dirs;
	std::ifstream indir("../detections/WIDER/wider_dirs.txt");
	while(!indir.eof()) {
		std::string str_dir;
		indir >> str_dir;		
		if(str_dir=="") continue;
		dirs.push_back(str_dir);
	}

	for(int i =0; i < dirs.size(); i++) 
	{
		int dir_err = mkdir(result_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		std::string dir = result_dir+dirs[i];
		dir_err = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	}
}


void run_wider(DATASET m_data)
{
	std::string result_dir;
	if(m_data == WIDER_TEST) result_dir = "../detections/WIDER/mtcnn_test/";
	else result_dir = "../detections/WIDER/mtcnn_val/";

	make_wider_dir(m_data, result_dir);

	std::string str_img_file;
	if(m_data == WIDER_TEST) str_img_file = "../detections/WIDER/wider_test_list.txt";
	else if (m_data == WIDER_VAL) str_img_file = "../detections/WIDER/wider_val_list.txt";
	else return;

	std::ifstream inFile(str_img_file.c_str(), std::ifstream::in);

	std::vector<string> image_list;
	while(!inFile.eof()) 
	{
		std::string str;
		inFile >> str;
		if(str=="") continue;
		image_list.push_back(str);
	}

	std::string wider_path;
	if(m_data==WIDER_TEST) wider_path = "/home/anudee/Desktop/CAFFE_SFD/datasets/WIDER/WIDER_test/images/";
	else if(m_data==WIDER_VAL) wider_path = "/home/anudee/Desktop/CAFFE_SFD/datasets/WIDER/WIDER_val/images/"; 

	for(int i = 0; i < image_list.size(); i++)
	{
		std::string imname = wider_path+image_list[i];
		cout << imname << endl;
		cout << "processing image " << i+1 << "/" << image_list.size() << " [" << image_list[i].c_str() << "]" << endl;

		cv::Mat image = cv::imread(imname);

		//initial models without image's width or height
		mtcnn find;
		std::vector<MTCNNResult> mtcnn_results = mtcnnDetection(find, image, 30);

		std::string txt_name = image_list[i];
		txt_name.replace(txt_name.end()-4, txt_name.end(), ".txt");
		txt_name = result_dir+txt_name;
	
		std::ofstream out_txt(txt_name.c_str());
		out_txt << image_list[i] << "\n";
		out_txt << mtcnn_results.size() << "\n";
		for(int j =0; j < mtcnn_results.size(); j++) {
			out_txt << mtcnn_results[j].r.x << " " << mtcnn_results[j].r.y << " " << mtcnn_results[j].r.width << " " 
			<< mtcnn_results[j].r.height << " " << mtcnn_results[j].score << "\n";
		}
		out_txt.close();

		imshow("image", image);

		waitKey(30);
	}	
}

void run_webcam()
{
	mtcnn find;

	VideoCapture cap(0);
	if(!cap.isOpened()) {
		cout << "fail to open webcam!" << endl;
		return;
	}

	cv::Mat image;
	while(true) {
	
		cap >> image;

		if(image.empty()) break;
		
		double time_begin = what_time_is_it_now();
		//detect face by min_size(60)
		std::vector<MTCNNResult> mtcnn_results = mtcnnDetection(find, image, 60);		
		double time_now = what_time_is_it_now();
		double time_diff = time_now-time_begin;
		
		cout << "MTCNN FPS: " << 1/time_diff << endl;

		imshow("result", image);
		if (waitKey(1) >= 0) break;		
	}
}

void run_image()
{
	mtcnn find;
	cv::Mat image = cv::imread("../image/1.jpg");	
	if(image.empty()) {
		cout << "Image not exist in the specified dir!";
		return;	
	} 

	std::vector<MTCNNResult> mtcnn_results = mtcnnDetection(find, image, 60);		

	imshow("result", image);	
	waitKey(1);
	getchar();
} 

int main()
{	
	// select running mode
	MODE m_mode = WEBCAM;

	if(m_mode == WEBCAM)
	{
		run_webcam();	
	}
	else if(m_mode == IMAGE) 
	{
		run_image();
	}
	else if(m_mode == IMAGE_LIST) 
	{
		// to be implemented
	}
	else if (m_mode == BENCHMARK_EVALUATION)
	{	
		// select dataset
		DATASET m_data = WIDER_VAL;

		if(m_data == AFW || m_data == PASCAL || m_data == FDDB)
			run_afw_pascal_fddb(m_data);
		else if(m_data == WIDER_VAL || m_data == WIDER_TEST) 
			run_wider(m_data);
	}

	
	return 1;
}