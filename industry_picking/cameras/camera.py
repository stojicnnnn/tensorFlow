import zivid
from industry_picking.cameras.zivid import connect_to_zivid_camera 
from industry_picking.cameras.zivid import get_camera_intrinsics_from_capture 

    
if __name__ == "__main__":
    # Use a 'with' statement to create the ONE AND ONLY Application instance.
    # This ensures it is properly initialized and shut down.
    camera = connect_to_zivid_camera()
    get_camera_intrinsics_from_capture(camera=camera)
    
    