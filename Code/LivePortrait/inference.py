# coding: utf-8

import os
import os.path as osp
import shutil
import tyro
import cv2
import subprocess
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline

def extract_frames_from_videos(video_dir, output_dir, image_format='jpg'):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(video_dir):
        if filename.endswith('.mp4'):
            video_path = os.path.join(video_dir, filename)
            base_name = os.path.splitext(filename)[0]
            save_path = os.path.join(output_dir, base_name)
            os.makedirs(save_path, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_filename = os.path.join(save_path, f"{frame_idx:05d}.{image_format}")
                cv2.imwrite(frame_filename, frame)
                frame_idx += 1
            cap.release()
            print(f"Extracted {frame_idx} frames from {filename} into {save_path}")



def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

def main():
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")

    ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
    if osp.exists(ffmpeg_dir):
        os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

    if not fast_check_ffmpeg():
        raise ImportError(
            "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
        )

    # specify configs for inference
    args = tyro.cli(ArgumentConfig)
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    live_portrait_pipeline = LivePortraitPipeline(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg
    )

    # run for every individual frame
    root_dir = os.path.abspath("../data")
    source_file = os.path.join(root_dir, "neutral.jpg")
    video_dir = os.path.join(root_dir, "video_test")
    driving_dir = os.path.join(root_dir, "frames_test")
    output_dir = os.path.join(root_dir, "animations")
    save_dir = os.path.join(root_dir, "retarget_test")
    args.source = source_file
    # video -> images
    # extract_frames_from_videos(video_dir, driving_dir)

    for name in sorted(os.listdir(driving_dir)):
        in_path = os.path.join(driving_dir, name)
        out_path = os.path.join(output_dir, name)
        save_path = os.path.join(save_dir, name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        args.output_dir = out_path
        for file in os.listdir(in_path):
            if file.endswith(".jpg"):
                in_file = os.path.join(in_path, file)
                out_file = os.path.join(out_path, "neutral--" + file)
                print(in_file, out_file)
                if os.path.exists(out_file):
                    continue
                args.driving = in_file
                # run
                live_portrait_pipeline.execute(args)
        # post process
        # Get all frame numbers
        image_files = [f for f in os.listdir(out_path) if not "_concat" in f]
        frame_numbers = sorted([int(f.split('.')[0].replace("neutral--", "")) for f in image_files])

        # Fill missing frames
        max_frame = frame_numbers[-1]
        prev_frame = max_frame
        frame_set = set(frame_numbers)

        for i in range(max_frame, -1, -1):
            if i in frame_set:
                prev_frame = i
                source = os.path.join(out_path, f"neutral--{i:05d}.jpg")
                target = os.path.join(save_path, f"{i:05d}.jpg")
            else:
                source = os.path.join(out_path, f"neutral--{prev_frame:05d}.jpg")
                target = os.path.join(save_path, f"{i:05d}.jpg")
            shutil.copy(source, target)


if __name__ == "__main__":
    main()


