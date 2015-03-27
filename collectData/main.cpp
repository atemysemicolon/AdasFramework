#include <iostream>

#include <boost/python.hpp>
#include <opencv2/opencv.hpp>
#include "np_opencv_converter.hpp"
#inc
using namespace std;

namespace py = boost::python;

cv::Mat process_mat(const cv::Mat& in) {
   // process matrix, or just plain-simple cloning!
   cv::Mat out = in.clone();
   return out;
}
boost::python::def("process_mat", &process_mat);



