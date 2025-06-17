from industry_picking.robots.xarm import Xarm
from industry_picking.cameras.realsense import Camera
import industry_picking.utils.helper_functions as help

if __name__ == "__main__":
    #arm = Xarm("192.168.1.184")
    cam = Camera(1224,1024)
    robot_poses_to_visit = help.loadPosesFile("test2")
    robot_poses = []
    target_poses = []
    #arm.connect()
    #for i, curr_pose in enumerate(robot_poses_to_visit):
    for i, curr_pose in enumerate(robot_poses_to_visit):
        cam.connect()
        camera_matrix,dist_coeffs = cam.getIntrinsics()
        print(f"\n--- Processing Pose {i+1}/{len(robot_poses_to_visit)} ---")
        #arm.move(pose=curr_pose)
        #pose_rob = arm.getPose()
        #robot_poses.append(pose_rob)
        frames = cam.captureImage()
        pose_cam = cam.capturePose(image=frames)
        target_poses.append(pose_cam)
        if(i>3):
            print(f"Transformation matrix after {i}th iteration.")
            help.calibrateHandEye(target_poses=target_poses,robot_poses=robot_poses_to_visit)

if __name__ == "__main1__":
    target_poses = help.loadPosesFile("Set1Camera")
    robot_poses = help.loadPosesFile("Set1Robot")
    help.calibrateHandEye(target_poses=target_poses,robot_poses=robot_poses)


        







