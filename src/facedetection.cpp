#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <windows.h>
#include <process.h>



using namespace cv;
using namespace std;


/// Global variables

int threshold_value = 0;
int threshold_type = 3;;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;
int countF = 0;
ofstream myFile;
std::ofstream outStream;
Mat src, src_gray, dst;
char* window_name = "Threshold Demo";

char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
char* trackbar_value = "Value";
vector< vector<int> > combinations;
VideoWriter record;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
String face_cascade_name = "E:/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "E:/opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

String face_cascade_names [5] = {
		"E:/opencv/data/haarcascades/haarcascade_frontalface_alt.xml",
		"E:/opencv/data/haarcascades/haarcascade_frontalface_alt_tree.xml",
		"E:/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml",
		"E:/opencv/data/haarcascades/haarcascade_frontalface_default.xml",
		"E:/opencv/data/haarcascades/haarcascade_profileface.xml"

};

CascadeClassifier face_cascade;
CascadeClassifier face_cascades[5];
CascadeClassifier eyes_cascade;
RNG rng(12345);


class bg :  public BackgroundSubtractorMOG2
{

public:

	void setnmixtures()
	{
		this->backgroundRatio = 0.9;
		this->bShadowDetection = true;
		this->fVarInit = 50;
		this->fVarMin = 10;
		this->fVarMax = 100;
		this->fCT = 0.05;
	}

};


template <class type>

class Parallel_clipBufferValues: public cv::ParallelLoopBody
{
private:
  type *bufferToClip;
  type minValue, maxValue;

public:
  Parallel_clipBufferValues(type* bufferToProcess, const type min, const type max)
    : bufferToClip(bufferToProcess), minValue(min), maxValue(max){}

  virtual void operator()( const cv::Range &r ) const {
    register type *inputOutputBufferPTR=bufferToClip+r.start;
    for (register int jf = r.start; jf != r.end; ++jf, ++inputOutputBufferPTR)
    {
        if (*inputOutputBufferPTR>maxValue)
            *inputOutputBufferPTR=maxValue;
        else if (*inputOutputBufferPTR<minValue)
            *inputOutputBufferPTR=minValue;
    }
  }
};





class rectclass{

	Rect rect;
	int classifier;
public:
	rectclass(Rect rc, int cl)
	{
		rect = rc;
		classifier = cl;
	}

	Rect getRect() { return rect;}
	int getClassifier() { return classifier;}


};

float distance (int x1, int y1, int x2, int y2)
{
	float dist = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1);
	return sqrt(dist);
}

vector<Rect> detectAndDisplay(CascadeClassifier &face_cascade, Mat frame )
{

  std::vector<Rect> faces;
  Mat frame_gray;

  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  //-- Detect faces
  std::time_t t = std::time(0);  // t is an integer type
  //std::cout << t << " start\n";
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(20, 20) );
  //-- Detect faces
  std::time_t t2 = std::time(0);  // t is an integer type
 // std::cout << t2 << " end\n";
 // std::cout << t2-t << " diff\n";

  return faces;
 }

void comb(int N, int K)
{
    std::string bitmask(K, 1); // K leading 1's
    bitmask.resize(N, 0); // N-K trailing 0's

    // print integers and permute bitmask
    do {
		vector<int> combin;
        for (int i = 0; i < N; ++i) // [0..N-1] integers
        {
            if (bitmask[i])
			{
			//	std::cout << " " << i;
				combin.push_back(i);
			}
        }
		combinations.push_back(combin);
       // std::cout << std::endl;
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
}

vector<Rect> findaccurateFaces(vector<rectclass> allrects)
{

	//vector< vector<int> > combinations;

	combinations.clear();


	if(allrects.size() >= 3) comb(allrects.size(),3);



	vector<Rect> differentRects;

/*
	if(allrects.size() < 3  && allrects.size() > 0 )
	{
		for(int i=0; i<allrects.size() ; i++)
		{
			differentRects.push_back(allrects[i].getRect());
		}
	}
*/

	for(int i=0; i<combinations.size(); i++)
	{

		int sum = 0;

		vector<int> zeros;
		for(int i=0; i<5; i++) zeros.push_back(0);
		for(int j=0; j<combinations[i].size();j++)
		{

			int cl = allrects[combinations[i][j]].getClassifier();
			zeros[cl-1] = 1;

		}
		bool different = true;

		int count = 0;
		for(int i=0; i<5; i++)
		{
			if(zeros[i] == 1)
			{
				count++;
			}
		}

		if(count == 3) different = true;

		if(different)
		{
			int x=0,y=0,w=0,h=0;
			for(int j=0; j<combinations[i].size();j++)
			{
				//std::cout << " " << combinations[i][j];
				Rect rec = allrects[combinations[i][j]].getRect();
				x =  x + rec.x;
				y = y + rec.y;
				w = w + rec.width;
				h = h + rec.height;
			}

			Rect rect1 =  allrects[combinations[i][0]].getRect();
			Rect rect2 =  allrects[combinations[i][1]].getRect();
			Rect rect3 =  allrects[combinations[i][2]].getRect();

			float dist1 = distance(rect1.x,rect1.y, rect2.x, rect2.y);
			float dist2 = distance(rect1.x + rect1.width, rect1.y + rect1.height, rect2.x + rect2.width, rect2.y + rect2.height);

			float dist3 = distance(rect2.x,rect2.y,rect3.x, rect3.y);
			float dist4 = distance(rect2.x + rect2.width,rect2.y + rect2.height, rect3.x + rect3.width, rect3.y + rect3.height);

			if(dist1 < 40.0 && dist2 < 40.0 && dist3 < 40.0 && dist4 < 40.0)
			{
				int size = combinations[i].size();
				//ofSetColor(ofColor::yellow);
				Rect rec;
				rec.x= x/size;
				rec.y = y/size;
				rec.width = w/size;
				rec.height = h/size;
				bool newface = true;

				for(int i=-0; i< differentRects.size(); i++ )
				{
					float dist1 = distance(rec.x,rec.y, differentRects[i].x, differentRects[i].y);
					float dist2 = distance(rec.x + rec.width, rec.y + rec.height, differentRects[i].x + differentRects[i].width, differentRects[i].y + differentRects[i].height);

					if(dist1 < 40.0 && dist2 < 40.0)
					{
						newface = false;
						break;
					}
				}

				if(newface)
				{
					differentRects.push_back(rec);
				}

				//do skin detection here maybe
			}

		}
		//std::cout << std::endl;
	}

	return differentRects;

}




void detectFaces(Mat frame)
{

	vector<rectclass> allrects;
	vector<Scalar> colors;
	colors.push_back(Scalar( 255, 0, 0 ));
	colors.push_back(Scalar( 0, 0, 255 ));
	colors.push_back(Scalar( 0, 0, 0 ));
	colors.push_back(Scalar( 0, 255, 0 ));
	colors.push_back(Scalar( 255, 255, 0 ));
	Mat faceFrame = frame.clone();

	std::time_t t1 = clock();

	for(int i=0; i<5; i++)
	{
		int val = 0;
		if( !face_cascades[i].load( face_cascade_names[i] ) )
		{
			printf("--(!)Error loading\n");
		}
		else
		{
			vector<Rect> rects = detectAndDisplay(face_cascades[i],frame);
			for(int i=0; i<rects.size(); i++)
			{
				Rect rc =  rects[i];
				rectclass rcClass (rc,i+1);
				allrects.push_back(rcClass);
				Point center( rects[i].x + rects[i].width*0.5, rects[i].y + rects[i].height*0.5 );
				//ellipse( frame, center, Size( rects[i].width*0.5, rects[i].height*0.5), 0, 0, 360, Scalar( 0, 0, val ), 4, 8, 0 );
				rectangle(frame, rects[i] ,colors[i],1,8,0);
			}
			val = val + 50;
		}

	}

	std::time_t t2 = clock();
	//std::cout << t2-t1 << " diff\n";
	//cout<<"calling find accurate face"<<endl;

	vector<Rect> faces = findaccurateFaces(allrects);
	Mat faceRoi;

	int rows = frame.rows;
	int cols = frame.cols;
	int numfaces = faces.size();
	int temp = 22;
	//outStream.write((char*)(&numfaces), sizeof(int));
	outStream << numfaces;

	cout<<"numfaces="<<numfaces<<endl;
	for( int i = 0; i < faces.size(); i++ )
	{
	    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
	  //  ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );



		/*frame.at<Vec3b>(0,1)[0] = faces[i].width;
		frame.at<Vec3b>(0,1)[1] = faces[i].height;
		frame.at<Vec3b>(0,1)[2] =  0;
*/
	    frame.at<Vec3b>(0,0)[0] = faces[i].x;
		frame.at<Vec3b>(0,0)[1] = faces[i].y;
		frame.at<Vec3b>(0,0)[2] = 0;

		frame.at<Vec3b>(0,1)[0] = faces[i].width;
		frame.at<Vec3b>(0,1)[1] = faces[i].height;
		frame.at<Vec3b>(0,1)[2] = 0;



	    faceRoi = frame(faces[i]);



	    /*faceRoi.at<Vec3b>(0,0)[0] = faces[i].x;
		faceRoi.at<Vec3b>(0,0)[1] = faces[i].y;
		faceRoi.at<Vec3b>(0,0)[2] = 0;*/
	    //rectangle(frame, faces[i],Scalar( 255, 255, 255 ),1,8,0);
	    rectangle(faceFrame, faces[i],Scalar( 255, 255, 255 ),1,8,0);


		cout<<"x="<<faces[i].x<<" y="<<faces[i].y<<" width="<<faces[i].width <<" height="<<faces[i].height<<endl;


	//    imshow( "Face Roi", faceRoi );
		outStream << '\t';
		outStream << faces[i].x;
		outStream << '\t';
		outStream << faces[i].y;
		outStream << '\t';
		outStream << faces[i].width;
		outStream << '\t';
		outStream << faces[i].height;

	/*    outStream.write((char*)(&x), sizeof(int));
	    outStream.write((char*)(&y), sizeof(int));
	    outStream.write((char*)(&width), sizeof(int));
	    outStream.write((char*)(&height), sizeof(int));*/
	    outStream.flush();

	   // imwrite( "E:/opencv/facedetection/samples/faceRoi.jpg", faceRoi );
	 //   imwrite( "E:/opencv/facedetection/samples/origImage.jpg", frame );
	}
	outStream << '\n';
	//record<<frame;

	countF++;
//	cout<<"Count:-"<<countF<<endl;
	imshow( "others", frame );
	imshow( "Face Detected", faceFrame );

}


int videoProcess(string location,string video)
{
	VideoCapture cap;
	ofstream myfile;
	myfile.open ("example.bin", ios::out | ios::app | ios::binary);
	cout<<sizeof(int)<<endl;

  //  cap.open(-1); // Bu þekliyle uygun olan Web kamerasýný açar ve görüntüyü ordan alýr bu astýrý açarsanýz alttaki satýrý kapatýn


//	cap.open("C:/Users/vk2382/Dropbox/SL Identification Project/videosbatch/" + video); // RGB perfroms better than GMM

	cap.open(location + video);
	cout<<"video:"<<video;

	//cap.open("C:/workspace2/stats/src/mp4/EoLynF7FPQ8.mp4"); //almost all perform well
	if( !cap.isOpened() )
	{

		puts("***Could not initialize capturing...***\n");
		return 0;
	}
//	namedWindow( "Capture ", CV_WINDOW_AUTOSIZE);
//	namedWindow( "Foreground ", CV_WINDOW_AUTOSIZE );
	Mat frame,foreground,image, initial;

	bg back;
	//back.setnmixtures();
	BackgroundSubtractorMOG2 mog = back;

	int fps=cap.get(CV_CAP_PROP_FPS);
	cout<<fps<<endl;
	if(fps<=0)
		fps=10;
	else
		fps=1000/fps;

	cap>>frame;

	//record  = VideoWriter("RobotVideo.mpg", CV_FOURCC('P','I','M','1'), fps, frame.size(), true);

	//record  = VideoWriter("RobotVideo.avi", CV_FOURCC('D', 'I', 'B', ' ') , fps, frame.size(), true);





	int count = 0;
	for(;;)
	{

		cap>>frame;   // Bir frame alýyoruz
		image=frame.clone();

		// Arka plan çýkarma kýsmý

		count = count + 1;
		if(!frame.empty())
		{
			detectFaces( frame );
		}

		mog(image,foreground,0.01);
		imshow("mog ",foreground);
		threshold(foreground,foreground,25,255,THRESH_BINARY);
		imshow("Thresholded ",foreground);
		medianBlur(foreground,foreground,0.01);
		imshow("blurred ",foreground);
		erode(foreground,foreground,Mat());
		imshow("Eroded ",foreground);
		dilate(foreground,foreground,Mat());
		imshow("dilated ",foreground);

		//skin GMM




        if(count == 3600) break;


		if( frame.empty() )
				break;

		char c = (char)waitKey(fps);
		if( c == 27 )   // ESC tuþuna basýlana kadar çalýþ
			break;

	}


	outStream.close();
}

void Threshold_Demo( int, void* );


void threshholdDemo()
{
	/// Load an image
	  src = imread( "E:/Desert.jpg", 1 );

	  /// Convert the image to Gray
	  cvtColor( src, src_gray, CV_RGB2GRAY );

	  /// Create a window to display results
	  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

	  /// Create Trackbar to choose type of Threshold
	  createTrackbar( trackbar_type,
	                  window_name, &threshold_type,
	                  max_type, Threshold_Demo );

	  createTrackbar( trackbar_value,
	                  window_name, &threshold_value,
	                  max_value, Threshold_Demo );

	  /// Call the function to initialize
	  Threshold_Demo( 0, 0 );

	  /// Wait until user finishes program
	  while(true)
	  {
	    int c;
	    c = waitKey( 20 );
	    if( (char)c == 27 )
	      { break; }
	   }

}



/**
 * @function Threshold_Demo
 */
void Threshold_Demo( int, void* )
{
  /* 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted
   */

  threshold( src_gray, dst, threshold_value, max_BINARY_value,threshold_type );

  imshow( window_name, dst );
}

void skindetection()
{

}

int main(int argc, char * argv[])
{

	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	string filename = string(argv[1]) + string(argv[2]) + ".txt";
	outStream.open(filename.c_str(), std::ios::out | std::ios::app);

	videoProcess(argv[1], argv[2]);
	//threshholdDemo();

}
