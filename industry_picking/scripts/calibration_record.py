from industry_picking.robots.xarm import Xarm
from industry_picking.cameras.realsense import Camera
import industry_picking.utils.helper_functions as help

#Manually record x number of poses for robot camera calibration
#

if __name__ == "__main__":
    arm = Xarm("192.168.1.184")
    
    recorded_poses = arm.record_manual_poses(20)
    help.savePosesFile(poses=recorded_poses,filename="poses20")

        


        







