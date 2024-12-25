#include <opencv2/opencv.hpp>
#include <argparse/argparse.hpp>
#include "utility.hpp"


/**
 * @brief Compute coordinate offset of bounding boxes in result.
 * @param pjs Projection matrices.
 * @return Coordinate offset.
 */
cv::Size2d compute_offset(std::vector<cv::Mat1d> pjs) {
  cv::Size2d offset(INFINITY, INFINITY);
  for (const auto p : pjs) {
    std::vector<cv::Point2d> tf_corners;
    cv::perspectiveTransform((std::vector<cv::Point2d>) {cv::Point2i(0, 0), cv::Point2i(1920, 0), cv::Point2i(0, 1080)}, tf_corners, p);
    offset.width = std::min({offset.width, tf_corners[0].x, tf_corners[2].x});
    offset.height = std::min({offset.height, tf_corners[0].y, tf_corners[1].y});
  }

  return offset;
}

void draw_bbox(cv::Mat3b img, cv::Rect2d bbox, cv::Scalar color, std::string label) {
  cv::rectangle(img, bbox, color, 6);
  const auto txt_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1.5, 4, NULL);
  cv::rectangle(img, cv::Rect2d(bbox.x, bbox.y - txt_size.height - 4, txt_size.width, txt_size.height + 4), color, -1);
  cv::putText(img, label, cv::Point2d(bbox.x, bbox.y - 4), cv::FONT_HERSHEY_SIMPLEX, 1.5, color[0] + color[1] + color[2] > 382.5 ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255), 4);
}

int main(int argc, char **argv) {
  argparse::ArgumentParser parser;
  parser.add_argument("-p", "--pj_file").required().help("specify projection matrix file").metavar("PATH_TO_PJ_FILE");
  parser.add_argument("-t", "--track_file").required().help("specify track file").metavar("PATH_TO_TRACK_FILE");    // json format for multicam_tracking
  parser.add_argument("-sv", "--src_vid_file").required().help("specify source video file").metavar("PATH_TO_SRC_VID_FILE");
  parser.add_argument("-tv", "--tgt_vid_file").required().help("specify target video file").metavar("PATH_TO_TGT_VID_FILE");
  parser.parse_args(argc, argv);

  // read projection matrix file
  const auto pj_dict = read_json(parser.get("--pj_file"));

  // compute coordinate offset
  std::vector<cv::Mat1d> pjs;
  for (const auto [_, p] : pj_dict.items()) {
    pjs.push_back((cv::Mat_<double>(3, 3) <<
      p["projective_matrix"][0][0], p["projective_matrix"][0][1], p["projective_matrix"][0][2],
      p["projective_matrix"][1][0], p["projective_matrix"][1][1], p["projective_matrix"][1][2],
      p["projective_matrix"][2][0], p["projective_matrix"][2][1], p["projective_matrix"][2][2]
    ));
  }
  const auto offset = compute_offset(pjs);

  // read track file
  auto track_dict = read_json(parser.get("--track_file"));

  // draw bboxes
  cv::VideoCapture cap(parser.get("--src_vid_file"));
  cv::VideoWriter rec(parser.get("--tgt_vid_file"), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), cap.get(cv::CAP_PROP_FPS), cv::Size2i(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
  std::map<int, cv::Scalar> colors;
  auto result_idx = 0;
  while (true) {
    cv::Mat3b frm;
    cap >> frm;
    if (frm.empty()) break;

    while (result_idx < track_dict.size() && track_dict[result_idx]["frame_id"] < cap.get(cv::CAP_PROP_POS_FRAMES) - 1) result_idx++;
    if (result_idx < track_dict.size() && track_dict[result_idx]["frame_id"] == cap.get(cv::CAP_PROP_POS_FRAMES) - 1) {
      for (const auto t : track_dict[result_idx]["tracks"]) {
        if (colors.find(t["track_id"]) == colors.end()) {
          colors[t["track_id"]] = cv::Scalar(random() % 255, random() % 255, random() % 255);
        }
        draw_bbox(frm, cv::Rect2d((double) t["bbox"][0] - offset.width, (double) t["bbox"][1] - offset.height, t["bbox"][2], t["bbox"][3]), colors[t["track_id"]], std::to_string((int) t["track_id"]));
      }
    }

    rec << frm;

    if ((int) cap.get(cv::CAP_PROP_POS_FRAMES) % 100 == 0) {
      std::cout << (int) cap.get(cv::CAP_PROP_POS_FRAMES) << " frames have been completed" << std::endl;
    }
  }

  cap.release();
  rec.release();

  return 0;
}
