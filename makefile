all: annotate.out stitch.out

annotate.out: annotate.cpp
	g++ $< -o $@ -std=c++2a \
	-I argparse/include/ -I json/include/ -I /usr/local/include/opencv4/ \
	-l opencv_core -l opencv_imgproc -l opencv_videoio

stitch.out: stitch.cpp
	g++ $< -o $@ -std=c++2a \
	-I argparse/include/ -I json/include/ -I /usr/local/include/opencv4/ \
	-l opencv_core -l opencv_cudawarping -l opencv_imgcodecs -l opencv_videoio

clean:
	rm *.out
