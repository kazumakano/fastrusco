#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <argparse/argparse.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <glob.h>

/*
FRM_SIZE : int, int
  Width and height of target video.
MASK_REG_EXP : (cam_name: string) -> string
  Regular expression of mask image file names.
VID_REG_EXP : (cam_name: string) -> string
  Regular expression of source video file names.
*/

#define FRM_SIZE 8000, 4000
#define MASK_REG_EXP(cam_name) cam_name + ".png"
#define VID_REG_EXP(cam_name)  cam_name + ".mp4"

namespace cuda = cv::cuda;
namespace fs = std::filesystem;

cuda::GpuMat stitch(const std::vector<cuda::GpuMat> imgs, const std::vector<cv::Mat> pjs, const std::vector<cuda::GpuMat> warped_masks) {
  cuda::GpuMat stitched_img(warped_masks[0].size(), CV_8UC3, cv::Scalar(0));

  for (int i = 0; i < imgs.size(); i++) {
    cuda::GpuMat warped_img;
    cuda::warpPerspective(imgs[i], warped_img, pjs[i], warped_masks[i].size());
    warped_img.copyTo(stitched_img, warped_masks[i]);
  }

  return stitched_img;
}

int main(int argc, char **argv) {
  // parse arguments
  argparse::ArgumentParser parser;
  parser.add_argument("-m", "--mask_dir").required().help("specify mask image directory").metavar("PATH_TO_MASK_DIR");
  parser.add_argument("-p", "--pj_file").required().help("specify projection matrix file").metavar("PATH_TO_PJ_FILE");
  parser.add_argument("-s", "--src_dir").required().help("specify source video directory").metavar("PATH_TO_SRC_DIR");
  parser.add_argument("-t", "--tgt_file").required().help("specify target video file").metavar("PATH_TO_TGT_FILE");
  parser.parse_args(argc, argv);

  // read projection matrix file
  std::ifstream pj_file(parser.get("--pj_file"));
  if (!pj_file.is_open()) {
    std::cout << "failed to open " << parser.get("--pj_file") << std::endl;
    exit(EXIT_FAILURE);
  }
  const auto pj_dict = nlohmann::json::parse(pj_file);
  pj_file.close();

  // setup
  std::vector<cv::VideoCapture> caps;
  std::vector<cv::Mat> pjs;
  std::vector<cuda::GpuMat> warped_masks;
  for (const auto [n, p] : pj_dict.items()) {
    glob_t mask_files, vid_files;
    glob(fs::path(parser.get("--mask_dir")).append(MASK_REG_EXP(n)).c_str(), 0, NULL, &mask_files);
    glob(fs::path(parser.get("--src_dir")).append(VID_REG_EXP(n)).c_str(), 0, NULL, &vid_files);
    if (mask_files.gl_pathc == vid_files.gl_pathc == 1) {
      caps.emplace_back(vid_files.gl_pathv[0]);
      pjs.push_back((cv::Mat_<double>(3, 3) <<
        p["projective_matrix"][0][0], p["projective_matrix"][0][1], p["projective_matrix"][0][2],
        p["projective_matrix"][1][0], p["projective_matrix"][1][1], p["projective_matrix"][1][2],
        p["projective_matrix"][2][0], p["projective_matrix"][2][1], p["projective_matrix"][2][2]
      ));
      const auto mask = cv::imread(mask_files.gl_pathv[0], cv::IMREAD_GRAYSCALE);
      cuda::GpuMat mask_on_gpu, warped_mask_on_gpu;
      mask_on_gpu.upload(mask);
      cuda::warpPerspective(mask_on_gpu, warped_mask_on_gpu, pjs.back(), cv::Size2i(FRM_SIZE));
      warped_masks.push_back(warped_mask_on_gpu);
    }
    globfree(&mask_files);
    globfree(&vid_files);
  }
  if (caps.size() == 0) {
    std::cout << "no mask image file or source video file was found" << std::endl;
    exit(EXIT_FAILURE);
  }

  cv::VideoWriter rec(parser.get("--tgt_file"), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 5, warped_masks[0].size());

  // stitch
  while (true) {
    std::vector<cuda::GpuMat> frms;
    auto is_eof = false;
    for (auto c : caps) {
      cv::Mat frm;
      c >> frm;
      if (frm.empty()) {
        is_eof = true;
        break;
      }
      cuda::GpuMat frm_on_gpu;
      frm_on_gpu.upload(frm);
      frms.push_back(frm_on_gpu);
    }
    if (is_eof) break;

    const auto stitched_frm_on_gpu = stitch(frms, pjs, warped_masks);
    cv::Mat stitched_frm;
    stitched_frm_on_gpu.download(stitched_frm);

    rec << stitched_frm;

    if ((int) caps[0].get(cv::CAP_PROP_POS_FRAMES) % 100 == 0) {
      std::cout << (int) caps[0].get(cv::CAP_PROP_POS_FRAMES) << " frames have been completed" << std::endl;
    }
  }

  for (auto c : caps) c.release();
  rec.release();

  return 0;
}
