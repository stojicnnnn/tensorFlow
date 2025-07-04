from industry_picking.robots.xarm import Xarm
from industry_picking.cameras.camera import RealSense
import industry_picking.utils.helper_functions as help

#Calibrating the robot camera using the hand-eye calibration method.
#This script will move the robot to a set of predefined poses, capture images from the camera

if __name__ == "__main__": 
    arm = Xarm("192.168.1.184")
    arm.connect()
    cam = RealSense(1224,1024)
    robot_poses_to_visit = help.loadPosesFile("poses20")
    robot_poses = []
    target_poses = []
    for i, curr_pose in enumerate(robot_poses_to_visit):
        cam.connect()
        camera_matrix,dist_coeff = cam.getIntrinsics()
        print(f"\n--- Processing Pose {i+1}/{len(robot_poses_to_visit)} ---")
        arm.move(pose=curr_pose)
        pose_rob = arm.getPose()
        robot_poses.append(pose_rob)
        frames = cam.captureImage()
        pose_cam = cam.capturePose(image=frames)
        target_poses.append(pose_cam)
        
    print(f"Transformation matrix after 20th iteration.")
    help.calibrate_hand_eye(target_poses=target_poses,robot_poses=robot_poses_to_visit)
