from industry_picking.robots.xarm import Xarm
from industry_picking.cameras.realsense import Camera
import industry_picking.utils.helper_functions as help
import time 

if __name__ == "__main__1":
    target_poses=[]
    cam = Camera(1224,1024)
    cam.connect()
    camera_matrix,dist_coeffs = cam.getIntrinsics()
    frames = cam.captureImage()
    pose_cam = cam.capturePose(image=frames)
    target_poses.append(pose_cam)
    print(target_poses)




if __name__ == "__main__":
    arm = Xarm("192.168.1.184")
    cam = Camera(1224,1024)
    robot_poses_to_visit = help.loadPosesFile("test5")
    robot_poses = []
    target_poses = []
    arm.connect()
    for i, curr_pose in enumerate(robot_poses_to_visit):
        cam.connect()
        camera_matrix,dist_coeffs = cam.getIntrinsics()
        print(f"\n--- Processing Pose {i+1}/{len(robot_poses_to_visit)} ---")
        arm.move(pose=curr_pose)
        pose_rob = arm.getPose()
        robot_poses.append(pose_rob)
        frames = cam.captureImage()
        pose_cam = cam.capturePose(image=frames)
        target_poses.append(pose_cam)
        if(i>5):
            print(f"Transformation matrix after {i}th iteration.")
            help.calibrateHandEye(target_poses=target_poses,robot_poses=robot_poses)
            
    help.savePosesFile(poses=target_poses,filename="targetset5")
    help.savePosesFile(poses=robot_poses,filename="robotset5")

            
if __name__ == "__main__":
    arm = Xarm("192.168.1.184")
    robot_poses_to_visit = help.loadPosesFile("test3")
    robot_poses = []
    arm.connect()
    for i, curr_pose in enumerate(robot_poses_to_visit):
        #cam.connect()
        #camera_matrix,dist_coeffs = cam.getIntrinsics()
        print(f"\n--- Processing Pose {i+1}/{len(robot_poses_to_visit)} ---")
        arm.move(pose=curr_pose)
        time.sleep(2)
if __name__ == "__main__":
    arm = Xarm("192.168.1.184")
    
    recorded_poses = arm.record_manual_poses(10)
    help.savePosesFile(poses=recorded_poses,filename="test4")

        


        







