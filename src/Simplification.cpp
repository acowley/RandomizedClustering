#include <opencv2/opencv.hpp>
#include <map>
#include <queue>

using namespace std;

// Compute the union of edges in all color channels
void triChromaticEdges(const cv::Mat& imgColor, cv::Mat& edgeMask) {
  cv::Mat chan1(imgColor.rows, imgColor.cols, CV_8U);
  cv::Mat chan2(imgColor.rows, imgColor.cols, CV_8U);
  cv::Mat chan3(imgColor.rows, imgColor.cols, CV_8U);
  cv::Mat chanArray[] = {chan1, chan2, chan3};
  cv::split(imgColor, chanArray);
  double t1 = 32;
  double t2 = 128;
  cv::Mat edges2(imgColor.rows, imgColor.cols, CV_8U);
  cv::Mat edges3(imgColor.rows, imgColor.cols, CV_8U);
  cv::Canny(chan1, edgeMask, t1, t2);
  cv::Canny(chan2, edges2, t1, t2);
  cv::Canny(chan3, edges3, t1, t2);
  cv::bitwise_or(edgeMask, edges2, edgeMask);
  cv::bitwise_or(edgeMask, edges3, edgeMask);
}

// Compute edges in a grayscale version of the supplied image.
void triChromaticEdges2(const cv::Mat& imgColor, cv::Mat& edgeMask) {
  cv::Mat g;
  double t1 = 32;
  double t2 = 128;
  cv::cvtColor(imgColor, g, CV_RGB2GRAY);
  cv::Canny(g, edgeMask, t1, t2);
}

typedef struct EDGEINFO {
  uint32_t edgeLength;
  uint32_t edgeWeight;
  EDGEINFO() : edgeLength(0), edgeWeight(0) {}
} EdgeInfo;

typedef pair<uint32_t, uint32_t> Edge;
typedef pair<float, Edge> WeightedEdge;

inline uint32_t lookupAndFlatten(vector<uint32_t>& v, uint32_t i) {
  uint32_t dst = i;
  while(v[dst]) dst = v[dst];
  
  // Now flatten the mapping
  uint32_t tmp = i;
  while(v[tmp]) {
    uint32_t nxt = v[tmp];
    v[tmp] = dst;
    tmp = nxt;
  }
  return dst;
}

// Simplify a segmentation with a bias to preserving segment
// boundaries that are supported by Canny edges. Returns the number of
// distinct codes after simplification.
int simplify(const cv::Mat& imgColor, cv::Mat& imgCode, int numCodes) {
  cv::Mat edgeMask;
  triChromaticEdges2(imgColor, edgeMask);
  cv::boxFilter(edgeMask, edgeMask, CV_8U, cv::Size(3,3), cv::Point(-1,-1), false);
  //cv::GaussianBlur(edgeMask, edgeMask, cv::Size(3,3), 0);

  vector<map<uint32_t, EdgeInfo> > adj(numCodes);
  for(int y = 1; y < imgCode.rows - 1; y++) {
    uint32_t *code_ptr = (uint32_t*)imgCode.ptr(y);
    uint8_t *edge_ptr = (uint8_t*)edgeMask.ptr(y);
    uint32_t code, left, right, above, below;
    left = *code_ptr;
    code_ptr++;
    edge_ptr++;
    code = *code_ptr;
    for(int x = 1; x < imgCode.cols - 1; x++, code_ptr++, edge_ptr++) {
      right = *(code_ptr+1);
      above = *(code_ptr-imgCode.cols);
      below = *(code_ptr+imgCode.cols);
      map<uint32_t, EdgeInfo>& ns = adj[code];
      if(code != left) {
        EdgeInfo& e = ns[left];
        e.edgeLength++;
        e.edgeWeight+=(uint32_t)*edge_ptr;
      }
      if(code != above) {
        EdgeInfo& e = ns[above];
        e.edgeLength++;
        e.edgeWeight+=(uint32_t)*edge_ptr;
      }
      if(code != right) {
        EdgeInfo& e = ns[right];
        e.edgeLength++;
        e.edgeWeight+=(uint32_t)*edge_ptr;
      }
      if(code != below) {
        EdgeInfo& e = ns[below];
        e.edgeLength++;
        e.edgeWeight+=(uint32_t)*edge_ptr;
      }
      left = code;
      code = right;
    }
  }

  // Now we can build a heap structure of all adjacencies prioritized
  // by normalized edge weight.
  priority_queue<WeightedEdge> q;
  for(int i = 0; i < adj.size(); i++) {
    int totalBoundaryWeight = 0;
    //int totalBoundaryLength = 0;
    map<uint32_t, EdgeInfo>& ns = adj[i];
    if(ns.size() == 0) continue;
    for(map<uint32_t, EdgeInfo>::const_iterator it = ns.begin();
        it != ns.end();
        it++) {
      totalBoundaryWeight += (*it).second.edgeWeight;
      //totalBoundaryLength += (*it).second.edgeLength;
    }
    // If a component has a long boundary with moderate edge support,
    // we don't try to merge that component. Alternately, if a
    // component has a short boundary, we push a single edge to be
    // collapsed. Finally, if an edge is of moderate length we push
    // weighted edges onto the heap.
    if(totalBoundaryWeight > 600*128) continue;
    if(totalBoundaryWeight < 200*128) {// && ns.size() == 1) {
      q.push(pair<float,Edge>(1000000.0f, pair<uint32_t,uint32_t>(i,(*ns.begin()).first)));
      continue;
    }

    for(map<uint32_t, EdgeInfo>::const_iterator it = ns.begin();
        it != ns.end();
        it++) {
      // We have a max-heap, so the desire to remove an edge is
      // proportional to the inverse of its average weight.
      float w = (float)((*it).second.edgeLength) / 
                (float)((*it).second.edgeWeight);

      q.push(pair<float, Edge>(w, pair<uint32_t, uint32_t>(i, (*it).first)));
    }
  }

  // Iterate until all edge segments are supported by Canny or their
  // own length.
  vector<uint32_t> renamer(numCodes, 0);
  int remainingCodes = numCodes;
  while(q.size() > 0) {
    const WeightedEdge& we = q.top();
    if(remainingCodes <= 20 && we.first < 1000000) break;
    if(we.first < 1) break;

    // Resolve the endpoints of the edge
    uint32_t src = we.second.first;
    uint32_t dst = we.second.second;
    src = lookupAndFlatten(renamer, src);
    dst = lookupAndFlatten(renamer, dst);

    // merge the codes that share this edge
    if(src != dst) { remainingCodes--; renamer[dst] = src; }

    q.pop();
  }
 
  // Final flattening pass
  for(int i = 0; i < renamer.size(); i++) lookupAndFlatten(renamer, i);

  // Now apply the renamer map to imgCode
  {
    uint32_t *row = (uint32_t*)imgCode.ptr(0);
    for(int i = 0; i < imgCode.rows*imgCode.cols; i++, row++) {
      *row = renamer[*row] ? renamer[*row] : *row;
    }
  }
  return remainingCodes;
}
