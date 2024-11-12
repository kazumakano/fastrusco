import csv
import logging
import os
import os.path as path
import pickle
import threading
from datetime import datetime
from glob import glob, iglob
from os import makedirs
from typing import Optional
import cv2
import ray
from ray.util import queue as ray_queue
from torch import cuda
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results as YoloResults

"""
BEGIN : str
    Begin time to detect workers.
END : str
    End time to detect workers.
GPU_PER_TASK : float
    Number of gpus per one task.
"""

BEGIN = "00:00:00"
END = "23:59:59"
GPU_PER_TASK = 0.2

@ray.remote(num_gpus=GPU_PER_TASK)
def _detect_by_cam(model_file: str, progress_queue: ray_queue.Queue, result_file: str, ts_cache: list[tuple[int, int]], vid_dir: str) -> None:
    logging.disable()

    result_dir = path.dirname(result_file)
    if not path.exists(result_dir):
        makedirs(result_dir)

    cam_name = path.basename(result_dir)
    model = YOLO(model=model_file)

    with open(result_file, mode="w") as f:
        writer = csv.writer(f)
        writer.writerow(("Camera_ID", "Frame_Number", "Tracker_ID", "Class_Name", "Coordinates"))

        cap, vid_idx = None, -1
        detect_id = 0
        for i, (vi, fi) in enumerate(ts_cache):
            if cap is not None and vid_idx != vi:
                cap.release()
            if vid_idx != vi:
                cap, vid_idx = cv2.VideoCapture(filename=glob(path.join(vid_dir, f"video_??-??-??_{vi:02d}.mp4"))[0]), vi
            while cap.get(cv2.CAP_PROP_POS_FRAMES) <= fi:
                frm = cap.read()[1]

            results: YoloResults = model(frm)[0]

            for b in results.boxes:
                writer.writerow((cam_name, i, detect_id, "worker", f"[{b.xywh[0, 0].item()} {b.xywh[0, 1].item()} {b.xywh[0, 2].item()} {b.xywh[0, 3].item()}]"))
                detect_id += 1

            progress_queue.put(cam_name)

def _show_progress_bars(frm_num: int, progress_queue: ray_queue.Queue) -> None:
    bars: dict[str, tqdm] = {}
    while True:
        try:
            cam_name = progress_queue.get()
        except EOFError:
            break

        if cam_name not in bars.keys():
            bars[cam_name] = tqdm(desc=f"detecting for camera {cam_name}", total=frm_num, position=len(bars))
        else:
            bars[cam_name].update()

def detect(model_file: str, result_dir: str, ts_cache_file: str, vid_dir: str, gpu_ids: Optional[list[int]] = None) -> None:
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpu_ids])
    ray.init()

    begin_in_sec = int((datetime.strptime(BEGIN, "%H:%M:%S") - datetime(1900, 1, 1)).total_seconds())
    end_in_sec = int((datetime.strptime(END, "%H:%M:%S") - datetime(1900, 1, 1)).total_seconds())
    date_str = path.basename(path.normpath(vid_dir))
    with open(ts_cache_file, mode="rb") as f:
        ts_cache: tuple[dict[str, list[tuple[int, int]]], dict[str, int]] = pickle.load(f)

    progress_queue = ray_queue.Queue()
    threading.Thread(target=_show_progress_bars, args=(5 * (end_in_sec - begin_in_sec), progress_queue)).start()

    pid_queue = []
    for d in sorted(iglob(path.join(vid_dir, "camera*"))):
        cam_name = path.basename(d)[6:]
        if cam_name in ts_cache[0].keys():
            if len(pid_queue) >= cuda.device_count() // GPU_PER_TASK:
                pid_queue.remove(ray.wait(pid_queue, num_returns=1)[0][0])
            pid_queue.append(_detect_by_cam.remote(
                path.abspath(model_file),
                progress_queue,
                path.join(path.abspath(result_dir), cam_name, f"{cam_name}_{date_str}_{begin_in_sec}_{end_in_sec}.csv"),
                ts_cache[0][cam_name][5 * (begin_in_sec - ts_cache[1][cam_name]):5 * (end_in_sec - ts_cache[1][cam_name])],
                path.abspath(d)
            ))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_file", required=True, help="specify model file", metavar="PATH_TO_MODEL_FILE")
    parser.add_argument("-r", "--result_dir", required=True, help="specify result directory", metavar="PATH_TO_RESULT_DIR")
    parser.add_argument("-tc", "--ts_cache_file", required=True, help="specify timestamp cache file", metavar="PATH_TO_TS_CACHE_FILE")
    parser.add_argument("-v", "--vid_dir", required=True, help="specify video directory", metavar="PATH_TO_VID_DIR")
    parser.add_argument("-g", "--gpu_ids", nargs="+", type=int, help="specify list of GPU device IDs", metavar="GPU_ID")
    args = parser.parse_args()

    detect(args.model_file, args.result_dir, args.ts_cache_file, args.vid_dir, args.gpu_ids)
