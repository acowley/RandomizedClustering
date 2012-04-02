#include <opencv2/opencv.hpp>
#include <sys/time.h>
using namespace std;

float timeDiff(struct timeval& start, struct timeval& stop) {
  return (stop.tv_sec - start.tv_sec) + \
         0.000001f * (stop.tv_usec - start.tv_usec);
}

vector<float> makeRandomPlanes(int numPlanes, int numDimensions);
int hammingHash(const cv::Mat& imgIn, cv::Mat& imgOut,
                vector<float>& planes,
                uint32_t hammingK,
                int maxRetries = 5);
int findComponents(cv::Mat& imgIn);
void colorContours(cv::Mat& base, cv::Mat& codeImg);
int simplify(const cv::Mat& imgColor, cv::Mat& imgCode, int numCodes);

inline void equalizeChannelHistograms(cv::Mat& imgIn) {
    cv::Mat chanH(imgIn.rows, imgIn.cols, CV_8U);
    cv::Mat chanS(imgIn.rows, imgIn.cols, CV_8U);
    cv::Mat chanV(imgIn.rows, imgIn.cols, CV_8U);
    cv::Mat chanArray[] = {chanH, chanS, chanV};
    cv::split(imgIn, chanArray);
    cv::equalizeHist(chanH,chanH);
    cv::equalizeHist(chanS,chanS);
    cv::equalizeHist(chanV,chanV);
    cv::merge(chanArray, 3, imgIn);
}

void segmentImage(const char* imageFile) {
  // We're using a colorspace with 3 channels, and 8 splitting planes.
  vector<float> planes = makeRandomPlanes(8,3);

  cv::Mat imgIn = cv::imread(imageFile);
  cv::Mat imgHSV, imgCode;
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  cv::cvtColor(imgIn, imgHSV, CV_BGR2HSV);
  equalizeChannelHistograms(imgHSV);
  int numMaxima = hammingHash(imgHSV, imgCode, planes, 2, 3);//1);
  int numComponents = findComponents(imgCode);
  int numSimple = simplify(imgIn, imgCode, numComponents);
  gettimeofday(&stop, NULL);
  printf("Found %d Hamming maxima producing %d components", 
         numMaxima, numComponents);
  printf(", simplified to %d regions\n", numSimple);
  printf("Hamming hash took %.1fms\n", timeDiff(start,stop)*1000.0f);
  colorContours(imgIn, imgCode);
  imwrite("coded.png", imgIn);
  usleep(100000); // Without this, imwrite is sometimes stopped prematurely
}

void segmentCamera() {
  // We're using a colorspace with 3 channels, and 8 splitting planes.
  vector<float> planes = makeRandomPlanes(8,3);
  cv::VideoCapture cam = cv::VideoCapture(0);
  cv::Mat imgIn, imgHSV, imgCode, imgDisplay;
  struct timeval start, stop;
  const char* title = "Hamming Hasher: Esc=Exit, Space=Pause/Resume";
  cv::namedWindow(title, CV_WINDOW_AUTOSIZE);
  while(1) {
    int key = cv::waitKey(1);
    // esc to exit, space to pause/unpause
    if(key == 27) break;
    else if(key == 32)
      while(cv::waitKey(100) != 32);

    cam >> imgIn;
    gettimeofday(&start, NULL);
    cv::cvtColor(imgIn, imgHSV, CV_BGR2HSV);
    equalizeChannelHistograms(imgHSV);
    int numMaxima = hammingHash(imgHSV, imgCode, planes, 2, 1);
    if(numMaxima) {
      int numComponents = findComponents(imgCode);
      int numSimple = simplify(imgIn, imgCode, numComponents);
      gettimeofday(&stop, NULL);
      printf("%d Hamming maxima; %d components; ", numMaxima, numComponents);
      printf(" %d regions after simplification\n", numSimple);
      printf("Hamming hash took %.1fms\n", timeDiff(start,stop)*1000.0f);
      colorContours(imgIn, imgCode);
    }
    cv::imshow(title, imgIn);
  }
}

void segmentVideo(const char* videoFile) {
  // We're using a colorspace with 3 channels, and 8 splitting planes.
  vector<float> planes = makeRandomPlanes(8,3);
  cv::VideoCapture vid = cv::VideoCapture(videoFile);
  cv::Mat imgIn, imgHSV, imgCode;
  struct timeval start, stop;
  const char* title = "Hamming Hasher: Esc=Exit, Space=Pause/Resume";
  cv::namedWindow(title, CV_WINDOW_AUTOSIZE);
  while(1) {
    int key = cv::waitKey(1);
    // esc to exit, space to pause/unpause
    if(key == 27) break;
    else if(key == 32)
      while(cv::waitKey(100) != 32);

    if(!vid.read(imgIn)) {
      // Loop the video
      cout << "Looping the video..." << endl;
      vid.set(CV_CAP_PROP_POS_FRAMES, 0);
      continue;
    }
    gettimeofday(&start, NULL);
    cv::GaussianBlur(imgIn, imgIn, cv::Size(3,3), 0);
    cv::cvtColor(imgIn, imgHSV, CV_BGR2HSV);
    equalizeChannelHistograms(imgHSV);
    int numMaxima = hammingHash(imgHSV, imgCode, planes, 2, 1);
    if(numMaxima) {
      int numComponents = findComponents(imgCode);
      int numSimple = simplify(imgHSV, imgCode, numComponents);
      gettimeofday(&stop, NULL);
      printf("%d Hamming maxima; %d components; ", numMaxima, numComponents);
      printf(" %d regions after simplification\n", numSimple);
      printf("Hamming hash took %.1fms\n", timeDiff(start,stop)*1000.0f);
      colorContours(imgIn, imgCode);
    }
    cv::imshow(title, imgIn);
  }
}

bool isVideoFile(const char* fileName) {
  while(*fileName && *fileName != '.') fileName++;
  if(*fileName) fileName++;
  if(strcmp(fileName, "mp4") == 0 ||
     strcmp(fileName, "mpg") == 0 ||
     strcmp(fileName, "avi") == 0 ||
     strcmp(fileName, "x264") == 0)
    return true;
  return false;
}

int main(int argc, char** argv) {
  // Seed the PRNG with current milliseconds.
  // Comment out the srand to make results repeatable.
  struct timeval t;
  gettimeofday(&t, NULL);
  srand((t.tv_sec*1000) + (t.tv_usec / 1000));
  
  if(argc == 1) 
    segmentCamera();
  else if(argc == 2) {
    if(isVideoFile(argv[1])) segmentVideo(argv[1]); 
    else segmentImage(argv[1]);
  }
  else {
    cout << "Usage: ./segment imgFile" << endl;
    return 1;
  }
  return 0;
}
