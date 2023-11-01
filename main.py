import cv2
import os
from mtcnn.mtcnn import MTCNN
import multiprocessing
from functools import partial
import gc
import argparse


# Initialize the MTCNN face detector only once to improve efficiency.
detector = MTCNN()


def bounding_box_from_masked_frame(image):
    """
    Extract the bounding box of the main white region from a masked frame.
    """
    # Convert the image to grayscale for thresholding.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to the grayscale image.
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find the contours in the binary image.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve the bounding rectangle of the largest contour (assuming it's the white mask).
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return x, y, w, h


def video_to_frames(video_path, masked_video_path, output_folder, video_name, frame_interval=10):
    """
    Extracts frames from the video and saves the detected faces from the frames.
    """
    cap = cv2.VideoCapture(video_path)
    cap_masked = cv2.VideoCapture(masked_video_path)

    frame_count = 0
    saved_frames = []

    # Folder names corresponding to different methods of creating deepfakes.
    folder_names = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "original"]

    max_frames = 300
    while True:
        # Read frames from the original and masked videos.
        ret, frame = cap.read()
        ret_masked, masked_frame = cap_masked.read()

        # Check if frames were read correctly.
        if not ret or not ret_masked:
            break
        if masked_frame is None or frame is None:
            continue
        if frame_count == max_frames:
            break

        # Process every Nth frame where N = frame_interval.
        if frame_count % frame_interval == 0:
            # Get the bounding box from the masked frame.
            X, Y, W, H = bounding_box_from_masked_frame(masked_frame)

            # Crop the region from the original frame.
            frame = frame[Y - 22: Y + H + 33, X - 27: X + W + 28]

            # Convert frame color space for MTCNN detection.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces using MTCNN.
            faces = detector.detect_faces(frame)

            max_area = 0
            best_face = None
            for i, face in enumerate(faces):
                x, y, width, height = face['box']
                area = width * height  # area of the current face
                confidence = face['confidence']

                # Update max_area and best_face if current face has larger area and high confidence.
                if area > max_area and confidence > 0.99:
                    max_area = area
                    best_face = frame[y:y + height, x:x + width]

            # If a best face is detected, save it.
            if best_face is not None:
                cropped_face = cv2.cvtColor(best_face, cv2.COLOR_BGR2RGB)

                # Determine the correct folder name from video_path.
                folder_name = next((name for name in folder_names if name in video_path), None)

                # Construct the filename to save.
                frame_filename = f"{output_folder}/{folder_name if folder_name != 'original' else 'originalVideo'}_{video_name}_frame_{frame_count}_{height}x{width}.jpg"

                # Save the cropped face.
                cv2.imwrite(frame_filename, cropped_face)
                saved_frames.append(frame_filename)

        frame_count += 1

    cap.release()
    return saved_frames


def process_video(inputs, output_folder_base):
    """
    Process a video by extracting frames and saving detected faces.
    """
    video_path, masked_video_path = inputs
    output_folder_for_video = os.path.join(output_folder_base)

    if not os.path.exists(output_folder_for_video):
        os.makedirs(output_folder_for_video)

    video_name, _ = os.path.splitext(os.path.basename(video_path))
    return video_to_frames(video_path, masked_video_path, output_folder_for_video, video_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos by extracting certain frames and saving detected faces.")
    parser.add_argument('--input_folder', required=True, help='Path to the input videos folder.')
    parser.add_argument('--masked_input_folder', required=True, help='Path to the masked input videos folder.')
    parser.add_argument('--output_folder_base', required=True, help='Base path for saving the extracted frames.')

    args = parser.parse_args()

    input_folder = args.input_folder
    masked_input_folder = args.masked_input_folder
    output_folder_base = args.output_folder_base

    cpu_count = multiprocessing.cpu_count()
    recommended_cpu_usage = cpu_count - 2

    # Lists to store paths.
    video_paths = []
    masked_video_paths = []

    # Populate the video paths list from the directory structure.
    for subdir, _, files in os.walk(input_folder):
        video_files = [f for f in files if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        for video_file in video_files:
            video_paths.append(os.path.join(subdir, video_file))
            masked_video_paths.append(os.path.join(masked_input_folder, video_file))

        # Set batch size for processing.
        batch_size = 12

        # Process videos in batches.
        for i in range(0, len(video_paths), batch_size):
            batch_video_paths = video_paths[i:i + batch_size]
            batch_masked_video_paths = masked_video_paths[i:i + batch_size]

            input_tuples = list(zip(batch_video_paths, batch_masked_video_paths))

            # Process videos in parallel using multiprocessing.
            with multiprocessing.Pool(processes= recommended_cpu_usage) as pool:
                func = partial(process_video, output_folder_base=output_folder_base)
                results = pool.map(func, input_tuples)

                # Clear memory.
                del results
                gc.collect()
