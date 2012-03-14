A library for randomized clustering of multidimensional data, with an application to image segmentation.

Prerequisite: [OpenCV](http://opencv.willowgarage.com/wiki/)   
_Tested against version 2.3.1, but should be fairly easygoing compatibility-wise._

The demo program produces an executable that may be used three ways:

* Running the executable with no arguments (e.g. `./segment`) will attempt to open an attached webcam and segment the video stream.
* Providing a video file as the argument (e.g. `./segment MyMovie.mp4`) will play the movie in a window with the segmentation drawn over the video.
* Providing an image file as the argument (e.g. `./segment MyImage.jpg`) will produce (or overwrite) an image file `coded.png` in the current working directory.

Note that the segmentation is based entirely on color, and performance
is impacted by image size. Since the underlying process is driven by a
pseudorandom number generator, segmentations produced on consecutive
runs of the program are very likely to be different, sometimes subtly
so, sometimes not so subtly. These different segmentations will likely
be of different subjective quality; repeated trials may be called for
when using this as a frontend to further image analysis.

An example segmentation from the [Berkeley dataset](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) (computed in 41.3ms on a dual-core Core i5 laptop):

![100099.jpg segmentation](https://github.com/acowley/RandomizedClustering/raw/master/demo/sample/bear-water.jpg)


