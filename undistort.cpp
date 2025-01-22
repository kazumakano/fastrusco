#include <ds.hpp>
#include "utility.hpp"
#include <opencv2/opencv.hpp>
#include <argparse/argparse.hpp>
#include <glob.h>
#include <omp.h>
#include <opencv2/cudawarping.hpp>

/**
 * @brief Maximum number of parallel tasks. Dynamic if undefined.
 */
#define MAX_TASK_NUM 8

namespace cuda = cv::cuda;
using DSCam = DoubleSphereCamera;
namespace fs = std::filesystem;

/**
 * @brief Compute projection map based on double sphere camera model.
 * @param cam Camera model.
 * @param f_len Focal length.
 * @param img_size Image size.
 * @return Map of every pixel.
 */
cv::Mat2f compute_map(const DSCam cam, const double f_len, const cv::Size2i img_size) {
  cv::Mat2f map(img_size, CV_32FC2);
  const auto z = f_len * MIN(img_size.height, img_size.width);

  for (auto y = 0; y < img_size.height; y++) {
    for (auto x = 0; x < img_size.width; x++) {
      DSCam::Vec2 pj_pt;
      cam.project(DSCam::Vec4(x - img_size.width / 2, y - img_size.height / 2, z, 0), pj_pt);
      map.at<cv::Vec2f>(y, x) = {(float) pj_pt[0], (float) pj_pt[1]};
    }
  }

  return map;
}

int main(int argc, char **argv) {
  #ifdef MAX_TASK_NUM
  omp_set_num_threads(MAX_TASK_NUM);
  #endif

  // parse arguments
  argparse::ArgumentParser parser;
  parser.add_argument("-c", "--cam_dir").required().help("specify camera calibration directory").metavar("PATH_TO_CAM_DIR");
  parser.add_argument("-s", "--src_dir").required().help("specify source video directory").metavar("PATH_TO_SRC_DIR");
  parser.add_argument("-t", "--tgt_dir").required().help("specify target video directory").metavar("PATH_TO_TGT_DIR");
  parser.parse_args(argc, argv);

  glob_t cam_files;
  glob((fs::path(parser.get("--cam_dir")) / "camera*.json").c_str(), 0, NULL, &cam_files);
  for (auto i = 0; i < cam_files.gl_pathc; i++) {
    const auto cam_name = fs::path(cam_files.gl_pathv[i]).filename().stem().string().substr(6);

    glob_t src_files;
    glob((fs::path(parser.get("--src_dir")) / ("camera" + cam_name) / "video_??-??-??_*.mp4").c_str(), 0, NULL, &src_files);
    if (src_files.gl_pathc == 0) continue;

    std::cout << "undistorting for camera " << cam_name << std::endl;

    // load projection map
    const auto param_dict = read_json(cam_files.gl_pathv[i])["value0"];
    DSCam::VecN param_vec;
    param_vec << param_dict["intrinsics"][0]["intrinsics"]["fx"], param_dict["intrinsics"][0]["intrinsics"]["fy"], param_dict["intrinsics"][0]["intrinsics"]["cx"], param_dict["intrinsics"][0]["intrinsics"]["cy"], param_dict["intrinsics"][0]["intrinsics"]["xi"], param_dict["intrinsics"][0]["intrinsics"]["alpha"];
    std::vector<cv::Mat1f> map;
    cv::split(compute_map(DSCam(param_vec), param_dict.find("f") == param_dict.end() ? 0.5 : (double) param_dict["f"], cv::Size2i(param_dict["resolution"][0][0], param_dict["resolution"][0][1])), map); 
    cuda::GpuMat x_map_on_gpu, y_map_on_gpu;
    x_map_on_gpu.upload(map[0]);
    y_map_on_gpu.upload(map[1]);

    const auto tgt_dir = fs::path(parser.get("--tgt_dir")) / ("camera" + cam_name);
    fs::create_directories(tgt_dir);

    // undistort
    #pragma omp parallel for
    for (auto j = 0; j < src_files.gl_pathc; j++) {
      cv::VideoCapture cap(src_files.gl_pathv[j]);
      cv::VideoWriter rec(tgt_dir / fs::path(src_files.gl_pathv[j]).filename(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), cap.get(cv::CAP_PROP_FPS), cv::Size2i(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
      while (true) {
        cv::Mat3b frm, mapped_frm;
        cap >> frm;
        if (frm.empty()) break;

        cuda::GpuMat frm_on_gpu, mapped_frm_on_gpu;
        frm_on_gpu.upload(frm);
        cuda::remap(frm_on_gpu, mapped_frm_on_gpu, x_map_on_gpu, y_map_on_gpu, cv::INTER_LINEAR);
        mapped_frm_on_gpu.download(mapped_frm);

        rec << mapped_frm;
      }
      cap.release();
      rec.release();
    }

    globfree(&src_files);
  }
  globfree(&cam_files);

  return 0;
}
