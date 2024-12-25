all: annotate.out stitch.out undistort.out

annotate.out: annotate.cpp utility.hpp
	g++ $< -o $@ -std=c++2a \
	-I argparse/include/ -I json/include/ -I /usr/local/include/opencv4/ \
	-l opencv_core -l opencv_imgproc -l opencv_videoio

stitch.out: stitch.cpp utility.hpp
	g++ $< -o $@ -std=c++2a \
	-I argparse/include/ -I json/include/ -I /usr/local/include/opencv4/ \
	-l opencv_core -l opencv_cudawarping -l opencv_imgcodecs -l opencv_videoio

undistort.out: undistort.cpp utility.hpp
	g++ $< -o $@ -std=c++2a -Wno-trigraphs \
	-I argparse/include/ -I dscam/include/ -I json/include/ -I /usr/local/include/opencv4/ \
	-l opencv_core -l opencv_cudawarping -l opencv_imgproc -l opencv_videoio

clean:
	rm -f *.out
