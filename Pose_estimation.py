import cv2
import numpy as np

class Pose_estimation(object): 
    def __init__(self):
        pass

    def recover_camera(self, K, x1, x2, five_point=False, threshold=0.1, RANSAC=True):
        if RANSAC:
            sample_method = cv2.RANSAC
        else:
            sample_method = None

        # find essential matrix
        if five_point:
            E_5point, mask1 = cv2.findEssentialMat(x1, x2, focal=K[0, 0], 
                pp=(K[0, 2], K[1, 2]), method=sample_method, threshold=threshold) # based on the five-point algorithm solver in [Nister03]((1, 2) Nistér, D. An efficient solution to the five-point relative pose problem, CVPR 2003.). [SteweniusCFS](Stewénius, H., Calibrated Fivepoint solver. http://www.vis.uky.edu/~stewe/FIVEPOINT/) is also a related. 
            E_recover = E_5point
        else:
            F_8point, mask1 = cv2.findFundamentalMat(x1, x2, cv2.RANSAC, threshold) # based on the five-point algorithm solver in [Nister03]((1, 2) Nistér, D. An efficient solution to the five-point relative pose problem, CVPR 2003.). [SteweniusCFS](Stewénius, H., Calibrated Fivepoint solver. http://www.vis.uky.edu/~stewe/FIVEPOINT/) is also a related. 
            E_8point = K.T @ F_8point @ K
            U,S,V = np.linalg.svd(E_8point)
            E_8point = U @ np.diag([1., 1., 0.]) @ V
            # mask1 = np.ones((x1.shape[0], 1), dtype=np.uint8)
            print('8 pppppoint!')
            E_recover = E_8point

        # recover pose
        points, R, t, mask2 = cv2.recoverPose(E_recover, x1, x2, focal=K[0, 0], 
            pp=(K[0, 2], K[1, 2]), mask=mask1.copy())
        pose = np.concatenate((R,t), axis=1)
        return {'pose': pose, 'inliers': mask2}


    def recover_camera_opencv(K, x1, x2, delta_Rtij_inv, five_point=False, threshold=0.1, show_result=True, c=False, \
        if_normalized=False, method_app='', E_given=None, RANSAC=True):
        # Computes scene motion from x1 to x2
        # Compare with OpenCV with refs from:
        ## https://github.com/vcg-uvic/learned-correspondence-release/blob/16bef8a0293c042c0bd42f067d7597b8e84ef51a/tests.py#L232
        ## https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
        ## http://answers.opencv.org/question/90070/findessentialmat-or-decomposeessentialmat-do-not-work-correctly/
        method_name = '5 point'+method_app if five_point else '8 point'+method_app
        if RANSAC:
            sample_method = cv2.RANSAC
        else:
            sample_method = None

        if show_result:
            print('>>>>>>>>>>>>>>>> Running OpenCV camera pose estimation... [%s] ---------------'%method_name)

        # Mostly following: # https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
        if E_given is None:
            if five_point:
                if if_normalized:
                    E_5point, mask1 = cv2.findEssentialMat(x1, x2, method=sample_method, threshold=threshold) # based on the five-point algorithm solver in [Nister03]((1, 2) Nistér, D. An efficient solution to the five-point relative pose problem, CVPR 2003.). [SteweniusCFS](Stewénius, H., Calibrated Fivepoint solver. http://www.vis.uky.edu/~stewe/FIVEPOINT/) is also a related. 
                else:
                    E_5point, mask1 = cv2.findEssentialMat(x1, x2, focal=K[0, 0], pp=(K[0, 2], K[1, 2]), method=sample_method, threshold=threshold) # based on the five-point algorithm solver in [Nister03]((1, 2) Nistér, D. An efficient solution to the five-point relative pose problem, CVPR 2003.). [SteweniusCFS](Stewénius, H., Calibrated Fivepoint solver. http://www.vis.uky.edu/~stewe/FIVEPOINT/) is also a related. 
                # x1_norm = cv2.undistortPoints(np.expand_dims(x1, axis=1), cameraMatrix=K, distCoeffs=None) 
                # x2_norm = cv2.undistortPoints(np.expand_dims(x2, axis=1), cameraMatrix=K, distCoeffs=None)
                # E_5point, mask = cv2.findEssentialMat(x1_norm, x2_norm, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=threshold) # based on the five-point algorithm solver in [Nister03]((1, 2) Nistér, D. An efficient solution to the five-point relative pose problem, CVPR 2003.). [SteweniusCFS](Stewénius, H., Calibrated Fivepoint solver. http://www.vis.uky.edu/~stewe/FIVEPOINT/) is also a related. 
            else:
                # F_8point, mask1 = cv2.findFundamentalMat(x1, x2, method=cv2.RANSAC) # based on the five-point algorithm solver in [Nister03]((1, 2) Nistér, D. An efficient solution to the five-point relative pose problem, CVPR 2003.). [SteweniusCFS](Stewénius, H., Calibrated Fivepoint solver. http://www.vis.uky.edu/~stewe/FIVEPOINT/) is also a related. 
                F_8point, mask1 = cv2.findFundamentalMat(x1, x2, cv2.RANSAC, 0.1) # based on the five-point algorithm solver in [Nister03]((1, 2) Nistér, D. An efficient solution to the five-point relative pose problem, CVPR 2003.). [SteweniusCFS](Stewénius, H., Calibrated Fivepoint solver. http://www.vis.uky.edu/~stewe/FIVEPOINT/) is also a related. 
                E_8point = K.T @ F_8point @ K
                U,S,V = np.linalg.svd(E_8point)
                E_8point = U @ np.diag([1., 1., 0.]) @ V
                # mask1 = np.ones((x1.shape[0], 1), dtype=np.uint8)
                print('8 pppppoint!')

            E_recover = E_5point if five_point else E_8point
        else:
            E_recover = E_given
            print('Use given E @recover_camera_opencv')
            mask1 = np.ones((x1.shape[0], 1), dtype=np.uint8)

        if if_normalized:
            if E_given is None:
                points, R, t, mask2 = cv2.recoverPose(E_recover, x1, x2, mask=mask1.copy()) # returns the inliers (subset of corres that pass the Cheirality check)
            else:
                points, R, t, mask2 = cv2.recoverPose(E_recover.astype(np.float64), x1, x2) # returns the inliers (subset of corres that pass the Cheirality check)
        else:
            if E_given is None:
                points, R, t, mask2 = cv2.recoverPose(E_recover, x1, x2, focal=K[0, 0], pp=(K[0, 2], K[1, 2]), mask=mask1.copy())
            else:
                points, R, t, mask2 = cv2.recoverPose(E_recover.astype(np.float64), x1, x2, focal=K[0, 0], pp=(K[0, 2], K[1, 2]))

        # print(R, t)
        # else:
            # points, R, t, mask = cv2.recoverPose(E_recover, x1, x2)
        if show_result:
            print('# (%d, %d)/%d inliers from OpenCV.'%(np.sum(mask1!=0), np.sum(mask2!=0), mask2.shape[0]))

        R_cam, t_cam = utils_geo.invert_Rt(R, t)

        error_R = utils_geo.rot12_to_angle_error(R_cam, delta_Rtij_inv[:3, :3])
        error_t = utils_geo.vector_angle(t_cam, delta_Rtij_inv[:3, 3:4])
        if show_result:
            print('Recovered by OpenCV %s (camera): The rotation error (degree) %.4f, and translation error (degree) %.4f'%(method_name, error_R, error_t))
            print(np.hstack((R, t)))

        # M_r = np.hstack((R, t))
        # M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
        # P_l = np.dot(K,  M_l)
        # P_r = np.dot(K,  M_r)
        # point_4d_hom = cv2.triangulatePoints(P_l, P_r, np.expand_dims(x1, axis=1), np.expand_dims(x2, axis=1))
        # point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
        # point_3d = point_4d[:3, :].T
        # scipy.io.savemat('test.mat', {'X': point_3d})

        if show_result:
            print('<<<<<<<<<<<<<<<< DONE Running OpenCV camera pose estimation. ---------------')

        E_return  = E_recover if five_point else (E_recover, F_8point)
        return np.hstack((R, t)), (error_R, error_t), mask2.flatten()>0, E_return