#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
# from real.camera import Camera
from robot import Robot
from scipy import optimize  
from mpl_toolkits.mplot3d import Axes3D  
from CPS import CPSClient
import math
import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import signal
import os


# Estimate rigid transform with SVD (from Nghia Ho)
def get_rigid_transform(A, B):
    assert len(A) == len(B)
    N = A.shape[0]; # Total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - np.tile(centroid_A, (N, 1)) # Centre the points
    BB = B - np.tile(centroid_B, (N, 1))
    H = np.dot(np.transpose(AA), BB) # Dot is matrix multiplication for array
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0: # Special reflection case
       Vt[2,:] *= -1
       R = np.dot(Vt.T, U.T)
    t = np.dot(-R, centroid_A.T) + centroid_B.T
    return R, t

def get_rigid_transform_error(z_scale):
    global measured_pts, observed_pts, observed_pix, world2camera, camera

    # Apply z offset and compute new observed points using camera intrinsics
    observed_z = observed_pts[:,2:] * z_scale
    observed_x = np.multiply(observed_pix[:,[0]]-cam_intrinsics[0][2],observed_z/cam_intrinsics[0][0])
    observed_y = np.multiply(observed_pix[:,[1]]-cam_intrinsics[1][2],observed_z/cam_intrinsics[1][1])
    new_observed_pts = np.concatenate((observed_x, observed_y, observed_z), axis=1)

    # Estimate rigid transform between measured points and new observed points
    R, t = get_rigid_transform(np.asarray(measured_pts), np.asarray(new_observed_pts))
    t.shape = (3,1)
    world2camera = np.concatenate((np.concatenate((R, t), axis=1),np.array([[0, 0, 0, 1]])), axis=0)

    # Compute rigid transform error
    registered_pts = np.dot(R,np.transpose(measured_pts)) + np.tile(t,(1,measured_pts.shape[0]))
    error = np.transpose(registered_pts) - new_observed_pts
    error = np.sum(np.multiply(error,error))
    rmse = np.sqrt(error/measured_pts.shape[0])
    return rmse
def realsense_stream(color,depth):
    bridge = CvBridge()
    im_height = 480
    # im_height = 360
    im_width = 848
    # im_width = 640
    global colordata
    global depthdata
    global intrinsics
    intrinsics = np.array([[435.1066589355469,0.0,431.18426513671875],[0.0,433.898681640625,242.7597198486328],[0.0,0.0,1.0]])
    # intrinsics = np.array([[262.7649841308594,0.0,324.61651611328125],[0.0,262.7649841308594,173.672607421875], [0.0,0.0,1.0]])

    
    tmp_color_data = bridge.imgmsg_to_cv2(color, 'bgr8')
    tmp_depth_data = bridge.imgmsg_to_cv2(depth, '16UC1')
    # tmp_depth_data = bridge.imgmsg_to_cv2(depth)

    # tmp_color_data = np.asarray(bytearray(color_image))
    tmp_color_data_num = np.array(tmp_color_data)
    tmp_color_data_num.shape = (im_height,im_width,3)
    # tmp_depth_data = np.asarray(depth_image)
    tmp_depth_data_num = np.array(tmp_depth_data)
    tmp_depth_data_num.shape = (im_height,im_width)
    tmp_depth_data_num = tmp_depth_data.astype(float)/1000
    colordata = tmp_color_data_num
    depthdata = tmp_depth_data_num
   



if __name__ == '__main__':
    try:

        rospy.init_node('listener',anonymous=True)
        
        colordata= None
        depthdata=None
        intrinsics=None
        tcp_host_ip = '192.168.0.10'
        tcp_port = 10003
        vel = 250
        acc = 300
        rad = 50
        TCP = 'TCP_vpg'
        UCS = 'Base'
        cps = CPSClient()
        print('Connecting to robot...')
        cps.HRIF_Connect(0,tcp_host_ip,tcp_port)
        cps.HRIF_IsConnected(0) #check is connected
        cps.HRIF_GrpEnable(0,0) #pow on
        result =[]
        cps.HRIF_ReadActPos(0,0,result) #read current pose
        dx = float(result[6])
        dy = float(result[7])
        dz = float(result[8])
        dRx = float(result[9])
        dRy = float(result[10])
        dRz = float(result[11])
        # [x,y,z] = [float(x) for x in result[6:9]]
        # #打印xyz坐标
        print("Robot start at : X = %f, Y = %f, Z = %f, RX = %f, RY = %f, RZ = %f"%(dx,dy,dz,dRx,dRy,dRz))
        # print("Robot start at : "+"X =%f"+"Y =%f"+"Z =%f"+"RX =%f"+"RY =%f"+"RZ =%f",result[6],result[7],result[8],result[9],result[10],result[11])

        # workspace_limits = np.asarray([[0.0,0.15], [0.6,0.7], [0.0,0.15]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
        workspace_limits = np.asarray([[-0.15,0.15], [0.6,0.65], [-0.45,-0.35]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
        
        calib_grid_step = 0.05
        checkerboard_offset_from_tool = [0,0.0075,0]
        # tool_orientation = [-180,50,-90] 
        tool_orientation = [-180,-15,90] 

        # arudion.close_gripper()
        # time.sleep(1.5)

        # Construct 3D calibration grid across workspace
        gridspace_x = np.linspace(workspace_limits[0][0], workspace_limits[0][1], int(1+(workspace_limits[0][1] - workspace_limits[0][0])/calib_grid_step))
        # gridspace_y = np.linspace(workspace_limits[1][0], workspace_limits[1][1], math.ceil((workspace_limits[1][1] - workspace_limits[1][0])/calib_grid_step))
        gridspace_y = np.linspace(workspace_limits[1][0], workspace_limits[1][1], int(1+(workspace_limits[1][1] - workspace_limits[1][0])/calib_grid_step))
        
        gridspace_z = np.linspace(workspace_limits[2][0], workspace_limits[2][1], int(1+(workspace_limits[2][1] - workspace_limits[2][0])/calib_grid_step))
        calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z)
        num_calib_grid_pts = calib_grid_x.shape[0]*calib_grid_x.shape[1]*calib_grid_x.shape[2]
        print("number:",num_calib_grid_pts)
        calib_grid_x.shape = (num_calib_grid_pts,1)
        calib_grid_y.shape = (num_calib_grid_pts,1)
        calib_grid_z.shape = (num_calib_grid_pts,1)
        calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1)

        measured_pts = []
        observed_pts = []
        observed_pix = []


        Jonit_angle = [90,0,90,0,90,0]
        # PCS = [0,600,0,-180,50,-90]
        PCS = [0,550,-300,-180,-10,90]

        # Make robot gripper point upwards
        cps.HRIF_WayPoint(0,0,1,PCS,Jonit_angle,TCP,UCS,vel,acc,rad,0,0,0,0,0)
        time.sleep(3)


        color = message_filters.Subscriber("/camera/color/image_raw",Image)
        # color = message_filters.Subscriber("/zed2i/zed_node/left/image_rect_color",Image)

        depth = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw",Image)
        # depth = message_filters.Subscriber("/zed2i/zed_node/depth/depth_registered",Image)

        color_depth = message_filters.ApproximateTimeSynchronizer([color,depth],10,1,allow_headerless=True)
        color_depth.registerCallback(realsense_stream)

        np.set_printoptions(threshold=np.inf)

        # Move robot to each calibration point in workspace
        print('Collecting data...')
        for calib_pt_idx in range(num_calib_grid_pts):
            tool_position = calib_grid_pts[calib_pt_idx,:]
            tool_position_2 = tool_position * 1000
            tool_position_1 = tool_position_2.astype(int).tolist()
            # tool_position_1 = tool_position_1 * 1000
            tool_pose = tool_position_1 + tool_orientation
            print(tool_pose)
            cps.HRIF_WayPoint(0,0,1,tool_pose,Jonit_angle,TCP,UCS,vel,acc,rad,0,0,0,0,0)
            time.sleep(3)
            
            # Find checkerboard center
            checkerboard_size = (3,3)
            refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            camera_color_img=colordata
            # print(colordata)SS
            camera_depth_img=depthdata
            # print(camera_depth_img)
            # camera_depth_img[np.isnan(camera_depth_img)]=0
            cam_intrinsics =intrinsics
            # bgr_color_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite('zed.png',camera_color_img)
            # print(camera_depth_img)

            gray_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2GRAY)
            checkerboard_found, corners = cv2.findChessboardCorners(gray_data, checkerboard_size, None, cv2.CALIB_CB_ADAPTIVE_THRESH)
            if checkerboard_found:
                corners_refined = cv2.cornerSubPix(gray_data, corners, (3,3), (-1,-1), refine_criteria)

                # Get observed checkerboard center 3D point in camera space
                checkerboard_pix = np.round(corners_refined[4,0,:]).astype(int)
                checkerboard_z = camera_depth_img[checkerboard_pix[1]][checkerboard_pix[0]]
                checkerboard_x = np.multiply(checkerboard_pix[0]-cam_intrinsics[0][2],checkerboard_z/cam_intrinsics[0][0])
                checkerboard_y = np.multiply(checkerboard_pix[1]-cam_intrinsics[1][2],checkerboard_z/cam_intrinsics[1][1])
                if checkerboard_z == 0:
                    continue

                # Save calibration point and observed checkerboard center
                observed_pts.append([checkerboard_x,checkerboard_y,checkerboard_z])
                # tool_position[2] += checkerboard_offset_from_tool
                tool_position = tool_position + checkerboard_offset_from_tool

                measured_pts.append(tool_position)
                observed_pix.append(checkerboard_pix)

                # Draw and display the corners
                # vis = cv2.drawChessboardCorners(robot.camera.color_data, checkerboard_size, corners_refined, checkerboard_found)
                vis = cv2.drawChessboardCorners(camera_color_img, (1,1), corners_refined[4,:,:], checkerboard_found)
                # 指定保存目录的路径
                save_directory = '/home/wangtong/音乐/VPG-master/images/calibration/'

                # 使用 os.path.join 将目录路径和文件名组合成完整的文件路径
                file_path = os.path.join(save_directory, '%06d.png' % len(measured_pts))

                # 保存图像到指定路径
                cv2.imwrite(file_path, vis)

                # cv2.imshow('Calibration',vis)
                # cv2.waitKey(10)
        # robot.go_home()

        measured_pts = np.asarray(measured_pts)
        observed_pts = np.asarray(observed_pts)
        observed_pix = np.asarray(observed_pix)
        world2camera = np.eye(4)#单位矩阵

        # Optimize z scale w.r.t. rigid transform error
        print('Calibrating...')
        z_scale_init = 1
        optim_result = optimize.minimize(get_rigid_transform_error, np.asarray(z_scale_init), method='Nelder-Mead')
        camera_depth_offset = optim_result.x

        # Save camera optimized offset and camera pose
        print('Saving...')
        np.savetxt('real/camera_depth_scale.txt', camera_depth_offset, delimiter=' ')
        get_rigid_transform_error(camera_depth_offset)
        camera_pose = np.linalg.inv(world2camera)
        np.savetxt('real/camera_pose.txt', camera_pose, delimiter=' ')
        print('Done.')
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
   