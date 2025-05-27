
Project integrating Zivid M60 2+ camera with a robot arm with a task of pick and place operation, adjusted for different position and orientation of the object. 
Technologies I plan on using: Zivid SDK, Open3D, PCL, Python, Tensorflow/PyTorch, Staubli Robotics Suite and potentially ROS2.
At the early stages of the project i will be using Python3 for faster and easier development but when i achieve certain features, i will try to do it also in C++.
Reason being that C++ is used widely in robotics for critical solutions, where performance and security is important.

Introduction:

  With the boom of industrial automotion, robotics and AI, the need for vision systems is at an all time high. 
  This includes autonomous vehicles, robots in both household and industrial applications, augmented reality and more.
  So far, computer vision algorithms have come so far, that there are almost standards for specific needs.
  YOLO,SSD,CNN and others each have their own advantage and disadvantage, depending on the use.
  However, as the complexity of the tasks grows, so does the need for more intelligent approach.
  At the beggining of the project, i experimented with mentioned 2D computer vision technologies but i quickly learned that it would not be enough.
  For out problem, robot needs to know the position of an object in all 3 axis, which can not be determined by simply looking at 2D picture.

  Here is where Zivid comes to play.
  Zivid 2+ M60 is an 3D depth camera, that can capture point cloud data among others.
  Explained in simple terms, pont cloud is a collection of points that define an object, and each one of them have 3 coordinates, xyz.
  This allows us to know exact placement of the object, which is crucial to our project.
  While the camera itself is a big help, using it by itself is not enough.
  We need to use a set of algorithm specifially designed to process and analyze point cloud data.
  In our case, this will be Open3D, a Paython library or possibly PCL later on, which is more used alongside C++.

Progress Logs: 
  26.05. 
    Developed basic functions for handling image capture using camera and loading files from file path on PC.
    Loading files function will be used a lot, as a way to calibrate camera or check software on point cloud data that we know for sure is good.
    Zivid website has a range of different point cloud examples, which will be useful for offline testing when camera is not present.
    Added a visualisation block that will output captured or loaded point clouds, similary to matplotlib when using OpenCV for example.
    The code will be supported by comments so i will not be going over each line here.
    So far i have been relying on Zivid software support while using Zivid SDK, youtube for some basic introduction to point clouds and Goodle Gemini for code structure and refactoring.
    
    Next steps:
      Capture point clouds with camera and visualise them like we did with loaded files,
      Explore more about different approaches regarding object recognition in 3D machine vision (PointNet,Pose estimation,CNNs etc.)
      Pick one approach and try it out.
    
  
