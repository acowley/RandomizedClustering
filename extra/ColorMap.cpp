#include <opencv2/opencv.hpp>

int palette_length = 64;
unsigned char* palette = 0L;

#define SATURATE_BYTE(x) (x < 0 ? 0 : (x > 255 ? 255 : x))

void initGnomeGolorMap()
{
    unsigned char tmp[] = {234, 232, 227,
                           186, 181, 171,
                           128, 125, 116,
                           86, 82, 72,
                           197, 210, 200,
                           131, 166, 127,
                           93, 117, 85,
                           68, 86, 50,
                           224, 182, 175,
                           193, 102, 90,
                           136, 70, 49,
                           102, 56, 34,
                           173, 167, 200,
                           136, 127, 163,
                           98, 91, 129,
                           73, 64, 102,
                           157, 184, 210,
                           117, 144, 174,
                           75, 104, 131,
                           49, 78, 108,
                           239, 224, 205,
                           224, 195, 158,
                           179, 145, 105,
                           130, 102, 71,
                           223, 66, 30,
                           153, 0, 0,
                           238, 214, 128,
                           209, 148, 12,
                           70, 160, 70,
                           38, 199, 38,
                           255, 255, 255,
                           0, 0, 0};
    
    if(palette != 0L) free(palette);
    palette_length = 32;
    palette = (unsigned char*)malloc(32 * 3);
    memcpy(palette, tmp, 32*3);
}

// This is an approximation of Octave's jet colormap
void initJetColorMap()
{
    int x;
    unsigned char* p;
    float c;
    
    if(palette != 0L) free(palette);
    
    palette_length = 64;
    palette = (unsigned char*)malloc(palette_length * 3);
    p = palette;
    for(x = 0; x < palette_length; x++)
    {
        // red
        c = (x >= 96 && x < 160) * (4 * x - 2) + 
            (x >= 160 && x < 224) + (x >= 224) * (-4 * x + 5);
        c += 32;
        *p = SATURATE_BYTE(c);
        p++;
        
        // green
        c = (x >= 32 && x < 96) * (4 * x - 1) + 
            (x >= 96 && x < 160) + (x >= 160 && x < 224) * (-4 * x + 4);
        c += 32;
        *p = SATURATE_BYTE(c);
        p++;
            
        // blue
        c = (x < 32) * (4 * x + 1) + (x >= 32 && x < 96) + 
            (x >= 96 && x < 160) * (-4 * x + 3);
        c += 32;
        *p = SATURATE_BYTE(c);
        p++;
    }
}

void initHueSpectrum()
{
  unsigned char *p;
  if(palette != 0L) free(palette);
  palette_length = 90;
  palette = (unsigned char*)malloc(palette_length * 3);
  uint8_t* b = palette;
  uint8_t* g = palette + 1;
  uint8_t* r = palette + 2;
  for(int theta = 0; theta < 360; theta += 4, r+=3, g+=3, b+=3) {
    float h = (float)theta / 60.0f;
    int i = floor(h);
    float f = h - i;
    float q = 1.0f - f;
    uint8_t fByte = (uint8_t)(f*255.0f);
    uint8_t qByte = (uint8_t)(q*255.0f);
    //p = 0, t = f
    switch(i) {
    case 0:
      *r = 255; *g = fByte; *b = 0;
      break;
    case 1:
      *r = qByte; *g = 255; *b = 0;
      break;
    case 2:
      *r = 0; *g = 255; *b = fByte;
      break;
    case 3:
      *r = 0; *g = qByte; *b = 255;
      break;
    case 4:
      *r = fByte; *g = 0; *b = 255;
      break;
    default:
      *r = 255; *g = 0; *b = qByte;
      break;
    }
  }
}

void initColorMap(uint8_t whichPalette)
{
    switch(whichPalette)
    {
        case 1:
        initGnomeGolorMap();
        break;

        case 2:
          initHueSpectrum();
          break;
        
        default:
        initJetColorMap();
    }
}

void colorize(cv::Mat& imgIn, cv::Mat& imgOut) {
  imgOut.create(imgIn.rows, imgIn.cols, CV_8UC3);
  uint8_t* optr = (uint8_t*)imgOut.ptr(0);
  uint32_t* iptr = (uint32_t*)imgIn.ptr(0);
  for(int i = 0; i < imgIn.cols * imgIn.rows; i++, optr+=3, iptr++)
    memcpy(optr, palette + (*iptr % palette_length) * 3, 3);
}

void colorContours(cv::Mat& base, cv::Mat& codeImg) {
  for(int y = 0; y < base.rows - 1; y++) {
    uint8_t* base_ptr = (uint8_t*)base.ptr(y);
    uint32_t* code_ptr = (uint32_t*)codeImg.ptr(y);
    for(int x = 0; x < base.cols - 1; x++, base_ptr+=3, code_ptr++) {
      uint32_t code = *code_ptr;
      if(code != *(code_ptr+1) || 
         code != *(code_ptr+base.cols-1) ||
         code != *(code_ptr+base.cols) ||
         code != *(code_ptr+base.cols+1)) {
        *base_ptr = 255;
        *(base_ptr+1) = 0;
        *(base_ptr+2) = 0;
      } 
    }
  }
}
