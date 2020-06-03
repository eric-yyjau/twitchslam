import os
import cv2
import numpy as np
from scipy.spatial import cKDTree
from constants import RANSAC_RESIDUAL_THRES, RANSAC_MAX_TRIALS
np.set_printoptions(suppress=True)

from skimage.measure import ransac
from helpers import add_ones, poseRt, fundamentalToRt, normalize, EssentialMatrixTransform, myjet
from utils.utils_misc import crop_or_pad_choice

def extractFeatures(img, detector='orb', num_points=3000):
  if detector == 'orb':
    orb = cv2.ORB_create()
    # detection
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)
    # extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
    kps, des = orb.compute(img, kps)
    kps = np.array([(kp.pt[0], kp.pt[1]) for kp in kps])
  elif detector == 'sift':
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    # kps = sift.detect(gray,None)
    # kps, des = sift.compute(gray, kps)

    # kps, des = sift.detectAndCompute(gray, None)
    kps, des = sift.detectAndCompute(img, None)
    x_all = np.array([p.pt for p in kps])
    choice = crop_or_pad_choice(x_all.shape[0], num_points, shuffle=True)
    kps = x_all[choice]
    des = des[choice]
  else:
    raise f"detector ({detector}) not defined"

  # return pts and des
  return kps, des

def match_frames_v2(f1, f2, K, detector='orb', if_ratio_test=True):
  from Pose_estimation import Pose_estimation
  # get variables
  des1, kp1 = f1.des, f1.kpus
  des2, kp2 = f2.des, f2.kpus
  # print(f"kp1: {kp1}, kp2: {kp2}")
  x1_all = kp1
  x2_all = kp2
  # print(f"des1: {des1}, des2: {des2}")

  # matching
  matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
  matches = matcher.knnMatch(des1, des2, k=2)

  good = []
  all_m = []
  
  for m,n in matches:
    all_m.append(m)
    if if_ratio_test:
      if m.distance < 0.8*n.distance:
        good.append(m)
  if not if_ratio_test:
    good = all_m
  # print(f"good: {good[0].queryIdx}")
  idx1 = [mat.queryIdx for mat in good]
  x1 = x1_all[idx1, :]
  idx2 = [mat.trainIdx for mat in good]
  x2 = x2_all[idx2, :]
  match_quality_good = np.hstack((x1, x2))
  # return {'matches': match_quality_good, 'x1': x1, 'x2': x2}
  data = {'matches': match_quality_good, 'x1': x1, 'x2': x2}
  # solve for pose
  # print(f"matches: {match_quality_good[:5]}")
  # x1 = data['x1']
  # x2 = data['x2']
  # K = np.identity(3)
  pose_est = Pose_estimation()
  results = pose_est.recover_camera(K, x1, x2, threshold=3)
  # results = pose_est.recover_camera(K, x1, x2, five_point=True, threshold=3)
  inliers = results['inliers']
  pose = results['pose']
  pose = np.concatenate((pose, np.array([[0,0,0,1]])), axis=0)
  print("Matches:  %d -> %d -> %d -> %d" % (len(f1.des), len(matches), len(inliers), sum(inliers)))
  
  # print(f"idx1: {len(idx1)}, inliers: {inliers}")
  idx1 = np.array(idx1)
  idx2 = np.array(idx2)
  data.update(results)
  # print(f"idx1[inliers]: {idx1[inliers.flatten() > 0]}")
  return idx1[inliers.flatten() > 0], idx2[inliers.flatten() > 0], np.linalg.inv(pose), data

def match_frames(f1, f2, detector='orb'):
  if detector == 'orb':
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.des, f2.des, k=2)
  elif detector == 'sift':
    # bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    # matches = bf.match(f1.des, f2.des)
    matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
    matches = matcher.knnMatch(f1.des, f2.des, k=2)
    print(f"len: {len(matches)}")

  # Lowe's ratio test
  ret = []
  idx1, idx2 = [], []
  idx1s, idx2s = set(), set()

  for m,n in matches:
    if m.distance < 0.75*n.distance:
      p1 = f1.kps[m.queryIdx]
      p2 = f2.kps[m.trainIdx]

      # be within orb distance 32
      if m.distance < 32:
        # keep around indices
        # TODO: refactor this to not be O(N^2)
        if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
          idx1.append(m.queryIdx)
          idx2.append(m.trainIdx)
          idx1s.add(m.queryIdx)
          idx2s.add(m.trainIdx)
          ret.append((p1, p2))

  # no duplicates
  assert(len(set(idx1)) == len(idx1))
  assert(len(set(idx2)) == len(idx2))

  assert len(ret) >= 8
  ret = np.array(ret)
  idx1 = np.array(idx1)
  idx2 = np.array(idx2)

  # fit matrix
  model, inliers = ransac((ret[:, 0], ret[:, 1]),
                          EssentialMatrixTransform,
                          min_samples=8,
                          residual_threshold=RANSAC_RESIDUAL_THRES,
                          max_trials=RANSAC_MAX_TRIALS)
  
  # print("Matches:  %d -> %d -> %d -> %d" % (len(f1.des), len(matches), len(inliers), sum(inliers)))
  # return idx1[inliers], idx2[inliers], fundamentalToRt(model.params)

  print("Matches:  %d -> %d -> %d -> %d" % (len(f1.des), len(matches), len(inliers), sum(inliers)))
  return idx1[inliers], idx2[inliers], fundamentalToRt(model.params)

class Frame(object):
  def __init__(self, mapp, img, K, pose=np.eye(4), tid=None, verts=None, detector='orb'):
    self.K = np.array(K)
    self.pose = np.array(pose)

    if img is not None:
      self.h, self.w = img.shape[0:2]
      if verts is None:
        self.kpus, self.des = extractFeatures(img, detector=detector)
      else:
        assert len(verts) < 256
        self.kpus, self.des = verts, np.array(list(range(len(verts)))*32, np.uint8).reshape(32, len(verts)).T
      self.pts = [None]*len(self.kpus)
    else:
      # fill in later
      self.h, self.w = 0, 0
      self.kpus, self.des, self.pts = None, None, None

    self.id = tid if tid is not None else mapp.add_frame(self)

  def annotate(self, img):
    # paint annotations on the image
    for i1 in range(len(self.kpus)):
      u1, v1 = int(round(self.kpus[i1][0])), int(round(self.kpus[i1][1]))
      if self.pts[i1] is not None:
        if len(self.pts[i1].frames) >= 5:
          cv2.circle(img, (u1, v1), color=(0,255,0), radius=3)
        else:
          cv2.circle(img, (u1, v1), color=(0,128,0), radius=3)
        # draw the trail
        pts = []
        lfid = None
        for f, idx in zip(self.pts[i1].frames[-9:][::-1], self.pts[i1].idxs[-9:][::-1]):
          if lfid is not None and lfid-1 != f.id:
            break
          pts.append(tuple(map(lambda x: int(round(x)), f.kpus[idx])))
          lfid = f.id
        if len(pts) >= 2:
          cv2.polylines(img, np.array([pts], dtype=np.int32), False, myjet[len(pts)]*255, thickness=1, lineType=16)
      else:
        cv2.circle(img, (u1, v1), color=(0,0,0), radius=3)
    return img


  # inverse of intrinsics matrix
  @property
  def Kinv(self):
    if not hasattr(self, '_Kinv'):
      self._Kinv = np.linalg.inv(self.K)
    return self._Kinv

  # normalized keypoints
  @property
  def kps(self):
    if not hasattr(self, '_kps'):
      self._kps = normalize(self.Kinv, self.kpus)
    return self._kps

  # KD tree of unnormalized keypoints
  @property
  def kd(self):
    if not hasattr(self, '_kd'):
      self._kd = cKDTree(self.kpus)
    return self._kd

