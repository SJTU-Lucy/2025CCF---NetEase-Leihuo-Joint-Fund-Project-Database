#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import shutil
from moviepy import AudioFileClip
import cv2
from tqdm import tqdm


class AudioError(RuntimeError):
    def __init__(self, arg):
        self.args = arg


class FPSError(RuntimeError):
    def __init__(self, arg):
        self.args = arg


class MergeError(RuntimeError):
    def __init__(self, arg):
        self.args = arg


# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')


def get_video_info(src_video_path):
    cap = cv2.VideoCapture(src_video_path)

    if not cap.isOpened():
        raise IOError("Video loads error with information!")
    if int(major_ver) < 3:
        total_frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
        height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    cap.release()
    return total_frames, fps, height, width


def generate_output_interface(num_windows, width, height, output_video_path, fps):
    window_size = (width * num_windows, height)

    video_output = cv2.VideoWriter(output_video_path,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, window_size)
    return window_size, video_output


def get_image_data(capture):
    ret, frame = capture.read()
    return frame


def merge_videos(tar_video_path, video_paths, video_labels, tar_fps):
    total_frames_list = []
    width_list = []
    height_list = []

    for video_path in video_paths:
        v_total_frames, _, v_height, v_width = get_video_info(video_path)
        width_list.append(v_width)
        height_list.append(v_height)
        total_frames_list.append(v_total_frames)

    min_frames = min(total_frames_list)
    window_height = max(height_list)
    window_width = 0
    for i in range(len(width_list)):
        ori_width = width_list[i]
        ori_height = height_list[i]
        temp_width = int(window_height * ori_width / ori_height)
        window_width += temp_width
        width_list[i] = temp_width
        height_list[i] = window_height

    merged_window_size = (window_width, window_height)

    video_writer = cv2.VideoWriter(tar_video_path, cv2.VideoWriter_fourcc(*'mp4v'), tar_fps, merged_window_size)

    caps_interface = tuple(cv2.VideoCapture(path) for path in video_paths)
    for i in tqdm(range(min_frames)):
        images_source = [get_image_data(cap) for cap in caps_interface]

        resized_images = []
        for j, image in enumerate(images_source):
            image = cv2.resize(image, (width_list[j], height_list[j]), interpolation=cv2.INTER_LINEAR)
            resized_images.append(image)

        merged_frame = cv2.hconcat(resized_images)

        video_writer.write(merged_frame)

    [cap.release() for cap in caps_interface]
    video_writer.release()


def modify_video_fps(sor_video_path, tar_fps, video_fps, synced_video_path):
    rate = video_fps / tar_fps
    cmd_sync = f'ffmpeg -i {sor_video_path} -filter:v "setpts={rate}*PTS" -r {tar_fps} {synced_video_path}'
    # cmd_sync = f"ffmpeg -i {sor_video_path} -r {tar_fps} -vsync passthrough -an {synced_video_path}"
    print(cmd_sync)
    os.system(cmd_sync)


def synchronic_video(folder, v_path, tar_fps, video_fps):
    print(v_path, tar_fps)
    default_temp_path_name = "~syncedfpsvideo"
    index = 1
    while True:
        temp_synced_path = os.path.join(folder,
                                        (default_temp_path_name + str(index) + ".mp4"))
        if os.path.exists(temp_synced_path):
            index += 1
        else:
            break

    modify_video_fps(v_path, tar_fps, video_fps, temp_synced_path)
    return temp_synced_path


def get_video_labels(video_infos):
    video_labels = []
    for video_info in video_infos:
        label = video_info[1]
        video_labels.append(label)
    return video_labels


def synchronic_fps(root_dir, video_infos, target_fps):
    synced_video_paths = []
    for video_info in video_infos:
        video_path = video_info[0]
        _, video_fps, _, _ = get_video_info(video_path)
        if video_fps != target_fps:
            video_path = synchronic_video(root_dir, video_path, target_fps, video_fps)
        synced_video_paths.append(video_path)

    return synced_video_paths


def concat_videos(video_infos, aligned_audio_path, tar_fps=60, output="output.mp4"):
    """concatenating source videos and output one merged video

    Args:
        video_infos: 2 dimensional array. Maximum support 6 videos.
         For instance:
            video_list = [[video1_path(str), label1(str)],
                          [video2_path(str), label2(str)],
                          [video3_path(str), label3(str)],
                          [video4_path(str), label4(str)],
                          ...]
        aligned_audio_path: aligned audio path.
        tar_fps: the target video fps, default set 60.
        output: the output video filepath, default is 'output.mp4'

    Returns:
        None

    Raises:
        MergeError: An error occurred combining audio and animated video.
    """
    # Create temp folder to store temp concat video
    time_stamp = int(time.time())
    default_temp_dir = f"temp{time_stamp}"
    print(default_temp_dir)
    if not os.path.exists(default_temp_dir):
        os.mkdir(default_temp_dir)
    else:
        shutil.rmtree(default_temp_dir)
        os.mkdir(default_temp_dir)

    temp_concat_video = os.path.join(default_temp_dir, "temp_concat_video.mp4")

    synced_videos = synchronic_fps(default_temp_dir, video_infos, tar_fps)
    labels = get_video_labels(video_infos)

    merge_videos(temp_concat_video, synced_videos, labels, tar_fps)

    if aligned_audio_path is None:
        cmd_merge = f"ffmpeg -y -i {temp_concat_video} -shortest {output}"
    else:
        my_audio_clip = AudioFileClip(aligned_audio_path)
        aligned_audio_path = os.path.join(default_temp_dir, "temp_audio.wav")
        my_audio_clip.write_audiofile(aligned_audio_path)
        cmd_merge = f"ffmpeg -y -i {temp_concat_video} -i {aligned_audio_path} -shortest {output}"

    os.system(cmd_merge)
    shutil.rmtree(default_temp_dir)


def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    print(path, fps, frame_count, duration)
    return fps, frame_count, duration


if __name__ == "__main__":
    gt_dir = "data/video_test"
    pred_dir = "data/render_final"
    output_dir = "data/concat_final"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for file in os.listdir(pred_dir):
        pred_file = os.path.join(pred_dir, file)
        gt_file = os.path.join(gt_dir, file.replace(".mov", ".mp4"))
        out_file = os.path.join(output_dir, file.replace(".mov", ".mp4"))
        if os.path.exists(out_file):
            continue
        video_list = [[gt_file, "GT"],
                      [pred_file, "PRED"]]

        gt_fps, _, _ = video_info(gt_file)

        target_fps = min(gt_fps, 30)

        concat_videos(video_list, aligned_audio_path=gt_file, tar_fps=target_fps, output=out_file)

