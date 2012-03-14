#include <opencv2/opencv.hpp>

using namespace std;

// Distinguish connected components from other regions with the same
// code.
int findComponents(cv::Mat& imgIn) {
  uint32_t prevRow[imgIn.cols];
  uint32_t componentCount = 1;
  uint32_t prevCol;
  uint32_t curr;
  uint32_t prevComponent;
  uint32_t prevRowComponents[imgIn.cols];
  uint32_t *row = (uint32_t*)imgIn.ptr(0);
  vector<uint32_t> renamer;

  memcpy(prevRow, row, sizeof(uint32_t)*imgIn.cols);

  renamer.reserve(1024);

  // We start naming components at 1, so fill in the first entries in
  // the population tracker and renamer. Note that a zero in the
  // renamer represents the nop that terminates a renaming chain.
  renamer.push_back(0);

  // Handle first pixel
  prevRowComponents[0] = 1;
  prevComponent = 1;
  prevCol = *row;
  *row = 1;
  renamer.push_back(0);
  componentCount = 2;
  row++;

  // Handle the rest of the first row
  for(int x = 1; x < imgIn.cols; x++, row++) {
    curr = *row;
    if(curr != prevCol) {
      *row = componentCount;
      prevComponent = componentCount++;
      renamer.push_back(0);
      prevCol = curr;
    }
    else {
      *row = prevComponent;
    }
    prevRowComponents[x] = prevComponent;
  }

  // Handle the rest of the rows
  for(int y = 1; y < imgIn.rows; y++) {

    // Handle the first column
    curr = *row;
    if(prevRow[0] == curr) {
      uint32_t oldName = prevRowComponents[0];
      prevComponent = oldName;
      if(renamer[prevComponent]) {
        while(renamer[prevComponent]) prevComponent = renamer[prevComponent];
        while(renamer[oldName]) {
          uint32_t temp = renamer[oldName];
          renamer[oldName] = prevComponent;
          oldName = temp;
        }
      }

      *row = prevComponent;
    }
    else {
      *row = componentCount;
      prevRow[0] = curr;
      prevComponent = componentCount;
      renamer.push_back(0);
      componentCount++;
    }
    prevRowComponents[0] = prevComponent;
    prevCol = curr;
    row++;
    
    // Handle the rest of the columns
    for(int x = 1; x < imgIn.cols; x++, row++) {
      curr = *row;
      if(prevCol == curr) {
        // Continue component from the left
        if(prevRow[x] == curr) {
          // Continue from above, too
          uint32_t oldName = prevRowComponents[x];
          uint32_t above = oldName;
          while(renamer[above]) above = renamer[above];
          while(renamer[oldName]) {
            uint32_t temp = renamer[oldName];
            renamer[oldName] = above;
            oldName = temp;
          }

          if(above != prevComponent) {
            // Merge components
            renamer[prevComponent] = above;
            prevComponent = above;
          }
        }
        else {
          // Not connected above
          prevRow[x] = curr;
          prevRowComponents[x] = prevComponent;
        }
      }
      else {
        // Not connected on the left
        prevCol = curr;
        if(prevRow[x] == curr) {
          // Continue component from above
          uint32_t oldName = prevRowComponents[x];
          prevComponent = oldName;
          while(renamer[prevComponent]) 
            prevComponent = renamer[prevComponent];
          while(renamer[oldName]) {
            uint32_t temp = renamer[oldName];
            renamer[oldName] = prevComponent;
            oldName = temp;
          }

          prevRowComponents[x] = prevComponent;
        }
        else {
          // New component
          prevComponent = componentCount++;
          prevRow[x] = curr;
          prevRowComponents[x] = prevComponent;
          renamer.push_back(0);
        }
      }
      *row = prevComponent;
    }
  }

  // Compress the range of component identifiers
  vector<uint32_t> shrinker(componentCount, 0);
  uint32_t keeperCount = 1;
  for(int i = 0; i < componentCount; i++) {
    uint32_t finalCode = i;
    while(renamer[finalCode]) finalCode = renamer[finalCode];
    uint32_t oldName = i;
    while(renamer[oldName]) {
      uint32_t temp = renamer[oldName];
      renamer[oldName] = finalCode;
      oldName = temp;
    }

    uint32_t col = shrinker[finalCode];
    if(col) shrinker[i] = col;
    else {
      shrinker[finalCode] = keeperCount;
      shrinker[i] = keeperCount;
      keeperCount++;
    }
  }

  row = (uint32_t*)imgIn.ptr(0);
  for(int i = 0; i < imgIn.rows * imgIn.cols; i++, row++)
    *row = shrinker[*row];

  return keeperCount;
}
