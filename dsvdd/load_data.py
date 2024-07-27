import os
import cv2
import shutil
from tqdm import tqdm

from datasets.mnist import get_mnist
from datasets.cifar10 import get_cifar10
from datasets.car import get_car

def load_data(args):

    implemented_datasets = ('mnist', 'cifar10', 'car_v1', 'car_v2', 'car_v3')
    assert args.dataset in implemented_datasets

    data = None

    if args.dataset == 'mnist':
        data = get_mnist(args)
    if args.dataset == 'cifar10':
        data = get_cifar10(args)
    if args.dataset == 'car_v1' or args.dataset == 'car_v2' or args.dataset == 'car_v3':
        data = get_car(args)

    return data

## prepare test data
## mp4 to frames

import os
import cv2
import shutil
from tqdm import tqdm

# Set the video path and the save directory
video_path = "mydata/YOLO_car.mp4"
save_dir = "mydata/test_hj/frames"

# Create the save directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def get_frame_from_video(video_path, save_dir, frames_per_second=10):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise Exception("Video load error")

    # Get the total frame count and FPS of the video
    len_video = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Set the folder to save images
    images_save_folder = save_dir
    if os.path.exists(images_save_folder):
        shutil.rmtree(images_save_folder)
    os.makedirs(images_save_folder)
    
    # Calculate the frame interval based on the desired frames per second
    frame_interval = fps // frames_per_second
    
    # Save video frames
    count = 0
    success = True
    frame_count = 0
    with tqdm(total=len_video) as pbar:
        while success:
            success, image = video.read()
            if not success:
                break
            
            # Check if the current frame is at the specified interval
            if frame_count % frame_interval == 0:
                save_idx = str(count + 1).zfill(5)
                save_image_path = os.path.join(images_save_folder, f"frame_{save_idx}.jpg")
                cv2.imwrite(save_image_path, image)
                count += 1
            
            frame_count += 1
            pbar.update(1)
    
    video.release()
    print("Success!")

# Call the function
get_frame_from_video(video_path, save_dir)
