import os
import cv2


def extract_frames_from_videos(video_dir, output_dir, image_format='jpg'):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(video_dir):
        if filename.endswith('.mp4'):
            video_path = os.path.join(video_dir, filename)
            base_name = os.path.splitext(filename)[0].replace("neutral--", "")
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


# Example usage:
video_dir = "/data2/liuchang/Retarget_data/Retarget_train"    # your video folder
output_dir = "/data2/liuchang/Retarget_data/frames_train"     # folder to store extracted images
extract_frames_from_videos(video_dir, output_dir)
