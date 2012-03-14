#include <opencv2/opencv.hpp>
#include <vector>
#include <set>
#include "HammingNeighborhoodFilters.h"

using namespace std;

inline void randomUnitVector(int dims, float* v) {
  float* p;
  while(1) {
    float len = 0.0f;
    p = v;
    for(int i = 0; i < dims; i++, p++) {
      float r = ((rand() / (float)RAND_MAX) - 0.5) * 2.0;
      len += r*r;
      *p = r;
    }
    // Normalize the vector
    if(len < 0.00001) continue;
    len = 1.0f / sqrt(len);
    p = v;
    for(int i = 0; i < dims; i++, p++)
      *p = *p * len;
    break;
  }
}

inline void randomizeAllPlanes(int numPlanes, int dims, float* vs) {
  for(;numPlanes > 0; numPlanes--, vs += dims) {
    randomUnitVector(dims, vs);
  }
}

// Produce [numPlanes] random vectors, each of dimension [numDimensions].
vector<float> makeRandomPlanes(int numPlanes, int numDimensions) {
  vector<float> planes(numPlanes*numDimensions);
  randomizeAllPlanes(numPlanes, numDimensions, &planes[0]);
  return planes;
}

inline uint32_t hammingDistance(uint32_t x, uint32_t y)
{
    uint32_t dist = 0;
    uint32_t r = x ^ y;
    for(dist = 0; r; dist++) r &= r - 1;
    return dist;
}

// Compute per-pixel projections
inline void projectPixels(const cv::Mat& imgIn,
                          const vector<float>& planes,
                          float* projections,
                          vector<float>& minimums,
                          vector<float>& maximums) {
  int numChannels = imgIn.channels();
  int numPlanes = planes.size() / numChannels;
  float** minRows = (float**)malloc(sizeof(float*)*imgIn.rows);
  float** maxRows = (float**)malloc(sizeof(float*)*imgIn.rows);
  
  #pragma omp parallel for
  for(int y = 0; y < imgIn.rows; y ++) {
    const uchar* row = imgIn.ptr(y);
    // HACK: we almost always have 3 color channels. Allocate 4 for
    // good luck.
    float pixelBuffer[4];
    float* minRow = (float*)malloc(sizeof(float)*numPlanes);
    memset(minRow, 0, sizeof(float)*numPlanes);
    float* maxRow = (float*)malloc(sizeof(float)*numPlanes);
    memset(maxRow, 0, sizeof(float)*numPlanes);

    float* projPtr = &projections[y*numPlanes*imgIn.cols];
    for(int x = 0, p = 0; x < imgIn.cols; x++) {
      for(int d = 0; d < numChannels; d++, row++) {
        pixelBuffer[d] = (float)(*row) - 128.0f;
      }

      const float* planePtr = &planes[0];
      for(int plane = 0; plane < numPlanes; plane++) {
        float sum = 0.0f;
        for(int d = 0; d < numChannels; d++, planePtr++)
          sum += pixelBuffer[d] * (*planePtr);
        *(projPtr++) = sum;
        if(sum > maxRow[plane]) maxRow[plane] = sum;
        if(sum < minRow[plane]) minRow[plane] = sum;
      }
    }
    minRows[y] = minRow;
    maxRows[y] = maxRow;
  }
  
  memcpy(&(minimums[0]), minRows[0], sizeof(float)*numPlanes);
  memcpy(&(maximums[0]), maxRows[0], sizeof(float)*numPlanes);

  free(minRows[0]);
  free(maxRows[0]);

  for(int y = 1; y < imgIn.rows; y++) {
    float* minRow = minRows[y];
    float* maxRow = maxRows[y];
    for(int plane = 0; plane < numPlanes; plane++) {
      if(minRow[plane] < minimums[plane])
        minimums[plane] = minRow[plane];
      if(maxRow[plane] > maximums[plane])
        maximums[plane] = maxRow[plane];
    }
    free(minRow);
    free(maxRow);
  }

  free(minRows);
  free(maxRows);
}

// Compute the midpoint of the extrema of the projections for each
// plane.
inline void encodeProjections(const vector<float>& minimums,
                              const vector<float>& maximums,
                              const float* projections,
                              const cv::Mat& imgIn,
                              cv::Mat& imgOut,
                              float* binColors,
                              vector<uint32_t>& bins) {
  int numPlanes = minimums.size();
  //vector<float> midpoints(numPlanes);
  float midpoints[numPlanes];
  for(int plane = 0; plane < numPlanes; plane++)
    midpoints[plane] = (maximums[plane]+minimums[plane]) * 0.5f;

  // Encode the per-pixel projections using midpoint information.
  for(int y = 0; y < imgOut.rows; y++) {
    //int* row = imgOut.ptr(y);
    uint32_t* row = ((uint32_t*)imgOut.data) + y*imgOut.cols;
    uint8_t* color = (uint8_t*)imgIn.ptr(y);
    const float* projRow = &(projections[y * imgOut.cols * numPlanes]);
    for(int x = 0; x < imgOut.cols; x++, row++, color+=3) {
      uint32_t code = 0;
      for(uint32_t plane = 0, mask = 1; 
          plane < numPlanes; 
          plane++, mask *= 2, projRow++) {
        if(*projRow > midpoints[plane]) code |= mask;
      }
      *row = code;
      bins[code]++;
      binColors[code*3] += (float)*color;
      binColors[code*3+1] += (float)*(color+1);
      binColors[code*3+2] += (float)*(color+2);
    }
  }
}

// Compute local maxima in Hamming space
void hammingMaxima(const vector<uint32_t>& bins,
                   int numPlanes,
                   int hammingK,
                   vector<uint32_t>& hMaxima) {
  uint32_t *counts = NULL;
  uint32_t *neighborMasks = NULL;
  switch(numPlanes) {
  case 3:
    counts = filterCounts3;
    neighborMasks = filter3;
    break;
  case 8:
    counts = filterCounts8;
    neighborMasks = filter8;
    break;
  case 9:
    counts = filterCounts9;
    neighborMasks = filter9;
    break;
  case 10:
    counts = filterCounts10;
    neighborMasks = filter10;
    break;
  case 11:
    counts = filterCounts11;
    neighborMasks = filter11;
    break;
  case 12:
    counts = filterCounts12;
    neighborMasks = filter12;
    break;
  case 13:
    counts = filterCounts13;
    neighborMasks = filter13;
    break;
  case 14:
    counts = filterCounts14;
    neighborMasks = filter14;
    break;
  }
  if(counts && neighborMasks) {
    for(int i = 0; i < bins.size(); i++) {
      int myPop = bins[i];
      if(myPop == 0) continue;
      //if(myPop < 100) continue;
      bool ismax = true;
      uint32_t *neighborMask = neighborMasks;
      for(int j = 0; j < hammingK; j++) {
        for(int k = 0; k < counts[j]; k++, neighborMask++) {
          if(bins[(*neighborMask) ^ i] >= myPop) {
            ismax = false;
            j = hammingK;
            break;
          }
        }
      }
      if(ismax) hMaxima.push_back(i);
    }
  }
  else {
    // Fallback only capable of traversing the Hamming-space
    // 1-neighborhood.
    for(int i = 0; i < bins.size(); i++) {
      int myPop = bins[i];
      if(myPop == 0) continue;
      bool ismax = true;
      uint32_t neighbor = 1 << (numPlanes - 1);
      while(neighbor > 1) {
        if(bins[neighbor ^ i] >= myPop) {
          ismax = false;
          break;
        }
        neighbor = neighbor >> 1;
      }
      if(ismax) hMaxima.push_back(i);
    }
  }
}

inline float colorDiff(const float* binColors, uint32_t x, uint32_t y) {
  float sum = 0.0f;
  float d;
  const float *xCol, *yCol;
  xCol = &binColors[x*3];
  yCol = &binColors[y*3];
  for(int i = 0; i < 3; i++, xCol++, yCol++) {
    d = *xCol - *yCol;
    sum += d*d;
  }
  return sum;
}

// Compute a mapping from each present Hamming code to a local maximum.
void mapToMaxima(vector<uint32_t>& bins,
                 const vector<uint32_t>& hMaxima,
                 float* binColors,
                 uint32_t* binMapping) {
  set<uint32_t> maximaSet(hMaxima.begin(), hMaxima.end());
  
  // The maxima map to themselves (this preserves the most popular
  // Hamming codes in the resulting coded image.
  for(int i = 0; i < hMaxima.size(); i++) {
    uint32_t x = hMaxima[i];
    binMapping[x] = x;
    // Normalize the bin color of each maximum
    float s = 1.0f / (float)bins[x];
    binColors[x*3] *= s;
    binColors[x*3+1] *= s;
    binColors[x*3+2] *= s;
  }

  for(int i = 0; i < bins.size(); i++) {
    if(bins[i] == 0) continue;
    if(maximaSet.find(i) == maximaSet.end()) {
      // Find the nearest Hamming maximum
      uint32_t minDist = hammingDistance(hMaxima[0], i);
      int bestCenter = 0;
      float bestColorDiff = colorDiff(binColors, hMaxima[0], i);
      for(int b = 1; b < hMaxima.size(); b++) {
        uint32_t dist = hammingDistance(hMaxima[b], i);
        if(dist < minDist) {
          minDist = dist;
          bestCenter = b;
          bestColorDiff = colorDiff(binColors, hMaxima[b], i);
        }
        else if(dist == minDist) {
          float diff = colorDiff(binColors, hMaxima[b], i);
          if(diff < bestColorDiff) {
            bestColorDiff = diff;
            minDist = dist;
            bestCenter = b;
          }
        }
      }
      binMapping[i] = binMapping[hMaxima[bestCenter]];
      bins[hMaxima[bestCenter]] += bins[i];
    }
  }
}

// We use a couple heuristics to decide when to swap out a plane for a
// random new one.
vector<float> discriminativePower(int numPlanes, vector<uint32_t>& maxima);
vector<float> planeCorrelation(int numPlanes, vector<uint32_t>& maxima);

// Returns the number of Hamming-space k-maxima. Arguments are an
// input image, the output image, a set of splitting planes, a value
// for k (e.g. k = 1 means find the Hamming codes that are maximal
// with respect to their immediate neighbors, k = 2 considers
// neighbors one step removed), and the number of allowed
// retries. Available retries are used when heuristics suggest that
// the provided splitting planes have not produced a "good"
// partitioning of the image.
int hammingHash(const cv::Mat& imgIn, cv::Mat& imgOut,
                 vector<float>& planes,
                 uint32_t hammingK,
                 int maxRetries) {
  if(imgOut.rows != imgIn.rows ||
     imgOut.cols != imgIn.cols ||
     imgOut.type() != CV_32S)
    imgOut.create(imgIn.rows, imgIn.cols, CV_32S);

  int numChannels = imgIn.channels();
  int numPlanes = planes.size() / numChannels;
  vector<float> minimums(numPlanes);
  vector<float> maximums(numPlanes);

  static int lastFrameRows = 0;
  static int lastFrameCols = 0;
  static int lastFrameNumPlanes = 0;
  static float* projections = 0L;
  static vector<uint32_t> bins;
  static uint32_t* binMapping = 0L;
  static float* binColors = 0L;

  if(imgIn.rows != lastFrameRows ||
     imgIn.cols != lastFrameCols ||
     numPlanes != lastFrameNumPlanes) {
    lastFrameRows = imgIn.rows;
    lastFrameCols = imgIn.cols;
    lastFrameNumPlanes = numPlanes;
    
    if(projections) free(projections);
    projections = (float*)malloc(sizeof(float)*imgIn.rows*imgIn.cols*numPlanes);

    bins.clear();
    bins.resize(1 << numPlanes, 0);
    if(binMapping) free(binMapping);
    binMapping = (uint32_t*)malloc(sizeof(uint32_t)*bins.size());

    if(binColors) free(binColors);
    binColors = (float*)malloc(sizeof(float)*3*bins.size());
  }
  
  vector<uint32_t> hMaxima;
  int retryCount = 0;

  while(retryCount < maxRetries) {
    retryCount++;
    bool hasBadPlane = false;

    memset(binColors, 0, sizeof(float)*3*bins.size());
    memset(&bins[0], 0, sizeof(uint32_t)*bins.size());

    // Project pixels onto the given planes. This is effected by
    // considering the sign of the dot product between each pixel and
    // the vector associated with each plane.
    projectPixels(imgIn, planes, projections, minimums, maximums);

    // Generate a binary encoding of each projection, store the codes in imgOut.
    encodeProjections(minimums, maximums, projections, imgIn, 
                      imgOut, binColors, bins);

    // Compute local maxima in Hamming space
    hammingMaxima(bins, numPlanes, hammingK, hMaxima);
    if(hMaxima.size() < 1) {
      randomizeAllPlanes(numPlanes, numChannels, &planes[0]);
      goto KEEP_TRYING;
    }

    // If a splitting plane does not differentiate the maxima at all,
    // replace it with a new one and try again.
    {
      vector<float> power = discriminativePower(numPlanes, hMaxima);
      for(int i = 0; i < numPlanes; i++) {
        if(power[i] < 0.0001) {
          hasBadPlane = true;
          randomUnitVector(numChannels, &planes[i*numChannels]);
        }
      }
    }
    if(hasBadPlane && retryCount < maxRetries) goto KEEP_TRYING;

    // Look for redundant planes. I.e. planes whose contributions to
    // the local maxima could be predicted by other planes.
    {
      vector<float> correlations = planeCorrelation(numPlanes, hMaxima);
      for(int i = 0; i < numPlanes; i++) {
        if(correlations[i] > 0.9f) {
          hasBadPlane = true;
          randomUnitVector(numChannels, &planes[i*numChannels]);
        }
      }
    }
    if(hasBadPlane && retryCount < maxRetries) goto KEEP_TRYING;

    // Map non-maxima in Hamming space to nearest maximum.
    mapToMaxima(bins, hMaxima, binColors, binMapping);

    break;
  KEEP_TRYING:
    if(retryCount < maxRetries) {
      memset((uint32_t*)&bins[0], 0, sizeof(uint32_t)*bins.size());
      hMaxima.clear();
    } else {
      mapToMaxima(bins, hMaxima, binColors, binMapping);
    }
    continue;
  }

  if(hMaxima.size() < 1) {
    return 0;
  }

  // Apply the mapping to our coded image
  for(int y = 0; y < imgIn.rows; y++) {
    int* row = (int*)(imgOut.data) + y*imgIn.cols;
    for(int x = 0; x < imgIn.cols; x++, row++) {
      *row = binMapping[*row];
    }
  }
  return hMaxima.size();
}
