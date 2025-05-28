import json
import os.path as path
import pickle
from datetime import datetime
from glob import glob
from os import makedirs
import cv2
import numpy as np
from tqdm import tqdm

"""
BEGIN : str
    Begin time to stitch video.
END : str
    End time to stitch video.
MASK_REG_EXP : (n: str) -> str
    Regular expression of mask image file name.
"""

BEGIN = "00:00:00"
END = "23:59:59"
MASK_REG_EXP = lambda n: n + ".png"

def _crop(pjs: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
    """
    Add margins or remove paddings to fit stitched images.

    Parameters
    ----------
    pjs : dict[str, ndarray[float64]]
        Dictionary of camera names and original projection matrices.

    Returns
    -------
    pjs : dict[str, ndarray[float64]]
        Dictionary of camera names and cropped projection matrices.
    img_size : tuple[int, int]
        Stitched image size.
    """

    stitched_ltrb = [np.inf, np.inf, -np.inf, -np.inf]
    for p in pjs.values():
        tf_corners = cv2.perspectiveTransform(np.array(((0, 0), (1920, 0), (0, 1080), (1920, 1080)), dtype=np.float32)[np.newaxis], p).squeeze(axis=0)
        stitched_ltrb[0] = min(stitched_ltrb[0], tf_corners[0, 0], tf_corners[2, 0])
        stitched_ltrb[1] = min(stitched_ltrb[1], tf_corners[0, 1], tf_corners[1, 1])
        stitched_ltrb[2] = max(stitched_ltrb[2], tf_corners[1, 0], tf_corners[3, 0])
        stitched_ltrb[3] = max(stitched_ltrb[3], tf_corners[2, 1], tf_corners[3, 1])
    pjs = pjs.copy()
    for n, p in pjs.items():
        pjs[n] = np.dot(np.array((
            (1, 0, -stitched_ltrb[0]),
            (0, 1, -stitched_ltrb[1]),
            (0, 0, 1)
        ), dtype=np.float64), p)

    return pjs, (int(stitched_ltrb[2] - stitched_ltrb[0]), int(stitched_ltrb[3] - stitched_ltrb[1]))

def stitch(mask_dir: str, pj_file: str, src_dir: str, tgt_file: str, ts_cache_file: str) -> None:
    # load constants
    begin_in_sec = int((datetime.strptime(BEGIN, "%H:%M:%S") - datetime(1900, 1, 1)).total_seconds())
    end_in_sec = int((datetime.strptime(END, "%H:%M:%S") - datetime(1900, 1, 1)).total_seconds())
    with open(pj_file) as f:
        pj_dict: dict[str, dict[str, int | list[list[float]]]] = json.load(f)
    with open(ts_cache_file, mode="rb") as f:
        ts_cache: tuple[dict[str, list[tuple[int, int]]], dict[str, int]] = pickle.load(f)

    # prepare constants
    cam_names = pj_dict.keys() & ts_cache[0].keys()
    pjs, frm_size = _crop({n: np.array(pj_dict[n]["projective_matrix"], dtype=np.float64) for n in cam_names})
    makedirs(path.dirname(tgt_file), exist_ok=True)
    rec = cv2.VideoWriter(tgt_file, cv2.VideoWriter_fourcc(*"mp4v"), 5, frm_size)
    warped_masks = {n: cv2.warpPerspective(cv2.imread(path.join(mask_dir, MASK_REG_EXP(n)), flags=cv2.IMREAD_GRAYSCALE), pjs[n], frm_size) for n in cam_names}

    # stitch
    status: dict[str, dict[str, int | np.ndarray | cv2.VideoCapture]] = {}
    for cur_in_sec in tqdm(np.arange(begin_in_sec, end_in_sec, step=0.2), desc="stitching"):
        stitched_frm = np.zeros((frm_size[1], frm_size[0], 3), dtype=np.uint8)
        for n in cam_names:
            vid_idx, frm_idx = ts_cache[0][n][int(5 * (cur_in_sec - ts_cache[1][n]))]

            if n in status.keys() and status[n]["vid_idx"] != vid_idx:
                status[n]["cap"].release()
            if n not in status.keys() or status[n]["vid_idx"] != vid_idx:
                status[n] = {"cap": cv2.VideoCapture(filename=glob(path.join(src_dir, f"camera{n}/video_??-??-??_{vid_idx:02d}.mp4"))[0]), "vid_idx": vid_idx}

            while status[n]["cap"].get(cv2.CAP_PROP_POS_FRAMES) <= frm_idx:
                status[n]["frm"] = status[n]["cap"].read()[1]

            cv2.copyTo(cv2.warpPerspective(status[n]["frm"], pjs[n], frm_size), warped_masks[n], dst=stitched_frm)
        rec.write(stitched_frm)

    for s in status.values():
        s["cap"].release()
    rec.release()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mask_dir", required=True, help="specify mask image directory", metavar="PATH_TO_MASK_DIR")
    parser.add_argument("-p", "--pj_file", required=True, help="specify projection matrix file", metavar="PATH_TO_PJ_FILE")
    parser.add_argument("-s", "--src_dir", required=True, help="specify source undistorted video directory", metavar="PATH_TO_SRC_DIR")
    parser.add_argument("-t", "--tgt_file", required=True, help="specify target video file", metavar="PATH_TO_TGT_FILE")
    parser.add_argument("-tc", "--ts_cache_file", required=True, help="specify timestamp cache file", metavar="PATH_TO_TS_CACHE_FILE")
    args = parser.parse_args()

    stitch(args.mask_dir, args.pj_file, args.src_dir, args.tgt_file, args.ts_cache_file)
