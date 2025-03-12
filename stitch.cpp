#include <opencv2/opencv.hpp>
#include "utility.hpp"
#include <opencv2/cudawarping.hpp>
#include <argparse/argparse.hpp>
#include <glob.h>

/**
 * @brief Regular expression of mask image file names.
 * @param n Camera name.
 */
#define MASK_REG_EXP(n) n + ".png"

/**
 * @brief Regular expression of source video file names.
 * @param n Camera name.
 */
#define VID_REG_EXP(n) n + ".mp4"

namespace cuda = cv::cuda;
namespace fs = std::filesystem;

/**
 * @brief Add margins or remove paddings to fit stitched images.
 * @param pjs Projection matrices. This value will be updated.
 * @return Stitched image size.
 */
cv::Size2i crop(std::vector<cv::Mat1d> pjs) {
  std::vector stitched_ltrb = {cv::Point2d(INFINITY, INFINITY), cv::Point2d(-INFINITY, -INFINITY)};
  for (const auto p : pjs) {
    std::vector<cv::Point2d> tf_corners;
    cv::perspectiveTransform((std::vector<cv::Point2d>) {cv::Point2i(0, 0), cv::Point2i(1920, 0), cv::Point2i(0, 1080), cv::Point2i(1920, 1080)}, tf_corners, p);
    stitched_ltrb[0].x = std::min({stitched_ltrb[0].x, tf_corners[0].x, tf_corners[2].x});
    stitched_ltrb[0].y = std::min({stitched_ltrb[0].y, tf_corners[0].y, tf_corners[1].y});
    stitched_ltrb[1].x = std::max({stitched_ltrb[1].x, tf_corners[1].x, tf_corners[3].x});
    stitched_ltrb[1].y = std::max({stitched_ltrb[1].y, tf_corners[2].y, tf_corners[3].y});
  }
  for (auto p : pjs) {
    p = (cv::Mat_<double>(3, 3) <<
      1, 0, -stitched_ltrb[0].x,
      0, 1, -stitched_ltrb[0].y,
      0, 0, 1
    ) * p;
  }

  return cv::Size2i(stitched_ltrb[1].x - stitched_ltrb[0].x, stitched_ltrb[1].y - stitched_ltrb[0].y);
}

cuda::GpuMat stitch(const std::vector<cuda::GpuMat> imgs, const std::vector<cv::Mat1d> pjs, const std::vector<cuda::GpuMat> warped_masks) {
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
  parser.add_argument("-s", "--src_dir").required().help("specify source undistorted video directory").metavar("PATH_TO_SRC_DIR");
  parser.add_argument("-t", "--tgt_file").required().help("specify target video file").metavar("PATH_TO_TGT_FILE");
  parser.parse_args(argc, argv);

  // read projection matrix file
  const auto pj_dict = read_json(parser.get("--pj_file"));

  // load constants
  std::vector<cv::VideoCapture> caps;
  std::vector<cuda::GpuMat> masks;
  std::vector<cv::Mat1d> pjs;
  for (const auto [n, p] : pj_dict.items()) {
    glob_t mask_files, vid_files;
    glob((fs::path(parser.get("--mask_dir")) / (MASK_REG_EXP(n))).c_str(), 0, NULL, &mask_files);
    glob((fs::path(parser.get("--src_dir")) / (VID_REG_EXP(n))).c_str(), 0, NULL, &vid_files);
    if (mask_files.gl_pathc == vid_files.gl_pathc == 1) {
      caps.emplace_back(vid_files.gl_pathv[0]);
      cuda::GpuMat mask;
      mask.upload(cv::imread(mask_files.gl_pathv[0], cv::IMREAD_GRAYSCALE));
      masks.push_back(mask);
      pjs.push_back((cv::Mat_<double>(3, 3) <<
        p["projective_matrix"][0][0], p["projective_matrix"][0][1], p["projective_matrix"][0][2],
        p["projective_matrix"][1][0], p["projective_matrix"][1][1], p["projective_matrix"][1][2],
        p["projective_matrix"][2][0], p["projective_matrix"][2][1], p["projective_matrix"][2][2]
      ));
    }
    globfree(&mask_files);
    globfree(&vid_files);
  }
  if (caps.size() == 0) {
    std::cout << "no mask image file or source video file was found" << std::endl;
    exit(EXIT_FAILURE);
  }

  // prepare constants
  const auto frm_size = crop(pjs);

  fs::create_directories(fs::absolute(parser.get("--tgt_file")).parent_path());
  cv::VideoWriter rec(parser.get("--tgt_file"), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), caps[0].get(cv::CAP_PROP_FPS), frm_size);

  std::vector<cuda::GpuMat> warped_masks;
  for (int i = 0; i < caps.size(); i++) {
      cuda::GpuMat warped_mask;
      cuda::warpPerspective(masks[i], warped_mask, pjs[i], frm_size);
      warped_masks.push_back(warped_mask);
  }

  // stitch
  while (true) {
    std::vector<cuda::GpuMat> frms;
    auto is_eof = false;
    for (auto c : caps) {
      cv::Mat3b frm;
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
    cv::Mat3b stitched_frm;
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
