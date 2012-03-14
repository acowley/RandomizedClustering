#include <opencv2/opencv.hpp>
using namespace std;

template<typename T>
inline T dot(int n, T* x, T* y) {
  T sum = 0;
  for(int i = 0; i < n; i++)
    sum += x[i] * y[i];
  return sum;
}

// Check to see how redundant certain planes are. If one plane
// perfectly predicts another, then the second should be abandoned.
vector<float> planeCorrelation(int numPlanes, vector<uint32_t>& maxima) {
  uint32_t mask = 1;
  int si = maxima.size();
  float* planeVectors = (float*)malloc(sizeof(float)*numPlanes*si);

  for(int i = 0; i < numPlanes; i++, mask <<= 1) {
    float* v = planeVectors + i*si;
    int len = 0;
    for(int j = 0; j < si; j++)
      if(mask & maxima[j]) {
        len++;
        v[j] = 1.0f;
      }
      else v[j] = 0.0f;
    
    // Normalize the vector
    float lenf = 1.0f / sqrt((float)len);
    for(int j = 0; j < si; j++) v[j] *= lenf;
  }

  vector<float> correlation(numPlanes, 0.0f);
  for(int i = 1; i < numPlanes; i++) {
    float highestCorrelation = 0.0f;
    float* v = planeVectors + i*si;
    for(int j = 0; j < i; j++) {
      float d = fabs(dot(si, planeVectors + j*si, v));
      if(d > highestCorrelation) highestCorrelation = d;
    }
    correlation[i] = highestCorrelation;
  }
  free(planeVectors);
  return correlation;
}

// Compute the ratio of maxima distinguished by a particular
// plane. E.g. if a plane only distinguishes one maximum from the
// rest, then it gets a value of 1 / numMaxima.
vector<float> discriminativePower(int numPlanes, vector<uint32_t>& maxima) {
  vector<float> power(numPlanes);
  uint32_t mask = 1;
  int si = maxima.size();
  float s = 1.0f / (float)si;

  for(int i = 0; i < numPlanes; i++, mask <<= 1) {
    int aboveMidpoint = 0;
    for(int j = 0; j < si; j++)
      if(mask & maxima[j]) aboveMidpoint++;

    if(aboveMidpoint > (si - aboveMidpoint))
      power[i] = (float)(si - aboveMidpoint) * s;
    else
      power[i] = (float)aboveMidpoint * s;
  }
  return power;
}
