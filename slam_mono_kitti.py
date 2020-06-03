#!/usr/bin/env python3
import sys
import os.path
# sys.path.append("/home/yoyee/Documents/deep_keyframe/ORB_SLAM2-PythonBindings/lib")
# import orbslam2
import time
import cv2
from tqdm import tqdm
import numpy as np
# from slam import SLAM
from slam_vo import SLAM
print(f"+++++ using slam_vo! +++++")
from pathlib import Path

class foo(object):
    def __init__(self):
        pass

config = foo()
config.detector = 'sift'
# def main(vocab_path, settings_path, sequence_path):
def main(sequence_path, save_path):

    image_filenames, timestamps = load_images(sequence_path)
    num_images = len(image_filenames)
    # load intrinsics
    image_0 = cv2.imread(image_filenames[0], cv2.IMREAD_UNCHANGED)
    W, H = image_0.shape[1], image_0.shape[0]
    K, Kinv = load_intrinsics(F=719, W=W, H=H)
    print(f"K = {K}")
    # slam = orbslam2.System(vocab_path, settings_path, orbslam2.Sensor.MONOCULAR)
    # slam.set_use_viewer(True)
    # slam.initialize()
    # twitchslam
    slam = SLAM(W, H, K)
    est_poses = []


    times_track = [0 for _ in range(num_images)]
    print('-----')
    print('Start processing sequence ...')
    print('Images in the sequence: {0}'.format(num_images))

    for idx in tqdm(range(num_images)):
        image = cv2.imread(image_filenames[idx], cv2.IMREAD_UNCHANGED)
        tframe = timestamps[idx]

        if image is None:
            print("failed to load image at {0}".format(image_filenames[idx]))
            return 1

        t1 = time.time()
        
        frame = cv2.resize(image, (W, H))
        # p = slam.process_frame(frame, None if gt_pose is None else np.linalg.inv(gt_pose[i]))
        print(f"detector: {config.detector}")
        p = slam.process_frame(frame, None, ba_optimize=False, detector=config.detector)
        # print(f"pose: {p}")
        est_poses.append(p)
        # slam.process_image_mono(image, tframe)

        t2 = time.time()

        # ttrack = t2 - t1
        # times_track[idx] = ttrack

        # t = 0
        # if idx < num_images - 1:
        #     t = timestamps[idx + 1] - tframe
        # elif idx > 0:
        #     t = tframe - timestamps[idx - 1]

        # if ttrack < t:
        #     time.sleep(t - ttrack)
        # if idx > 10:
        #     break

    # save poses
    from helpers import save_poses
    est_poses = np.array(est_poses)
    print(f"est_poses: {est_poses.shape}")
    save_poses(est_poses, f"{save_path}/est_poses.txt")

    # save_trajectory(slam.get_trajectory_points(), 'trajectory.txt')

    # slam.shutdown()

    times_track = sorted(times_track)
    total_time = sum(times_track)
    print('-----')
    print('median tracking time: {0}'.format(times_track[num_images // 2]))
    print('mean tracking time: {0}'.format(total_time / num_images))

    return 0


def load_intrinsics(F, W, H):
    if W > 1024:
        downscale = 1024.0/W
        F *= downscale
        H = int(H * downscale)
        W = 1024
    print("using camera %dx%d with F %f" % (W,H,F))

    # camera intrinsics
    K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
    Kinv = np.linalg.inv(K)
    return K, Kinv

def load_images(path_to_sequence):
    timestamps = []
    with open(os.path.join(path_to_sequence, 'times.txt')) as times_file:
        for line in times_file:
            if len(line) > 0:
                timestamps.append(float(line))

    return [
        # os.path.join(path_to_sequence, 'image_0', "{0:06}.png".format(idx)) # original orbslam
        os.path.join(path_to_sequence, 'image_2', "{0:06}.png".format(idx)) # scsfm uses image_2
        for idx in range(len(timestamps))
    ], timestamps

def load_image_files(dataset):
    # dataset
    if args.dataset == "kitti":
        image_dir = Path(args.dataset_dir + args.sequence + "/image_2/")
    elif args.dataset == "euroc":
        # subfolders = '/datasets/euroc/V1_01_easy/mav0/cam0/data/'
        subfolders = "/mav0/"
        image_dir = Path(args.dataset_dir + args.sequence + subfolders)
    # elif args.dataset == "euroc_undist":
    #     image_dir = Path(args.dataset_dir + args.sequence + "_0")
    #     print(f"image_dir: {image_dir}")

    if args.dataset == "kitti":
        test_files = sum(
            [image_dir.files("*.{}".format(ext)) for ext in args.img_exts], []
        )
    elif args.dataset == "euroc":
        test_files = read_images_files_from_folder(image_dir, folder="cam0", ext=args.img_exts)
        pass
    
    test_files.sort()
    return test_files

def save_trajectory(trajectory, filename):
    with open(filename, 'w') as traj_file:
        traj_file.writelines('{time} {r00} {r01} {r02} {t0} {r10} {r11} {r12} {t1} {r20} {r21} {r22} {t2}\n'.format(
            time=repr(t),
            r00=repr(r00),
            r01=repr(r01),
            r02=repr(r02),
            t0=repr(t0),
            r10=repr(r10),
            r11=repr(r11),
            r12=repr(r12),
            t1=repr(t1),
            r20=repr(r20),
            r21=repr(r21),
            r22=repr(r22),
            t2=repr(t2)
        ) for t, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 in trajectory)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: ./orbslam_mono_kitti  path_to_sequence save_path')
        save_path = './'
    else:
        save_path = sys.argv[2]
    Path(save_path).mkdir(exist_ok=True, parents=True)
    main(sys.argv[1], save_path)
