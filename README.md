# Recurrent Tracker (RT)

This is our repository of our <a href="https://ieeexplore.ieee.org/abstract/document/8756630">paper</a> (Fully  end-to-end composite recurrent convolution network for deformable facial tracking in the wild). 

Requirements : 
1. MTCNN requirements : https://github.com/DuinoDu/mtcnn
2. Tensorflow GPU : https://www.tensorflow.org/install/install_linux
3. Other package, can be installed by cloning my environment file on src : env.yml

<b> Landmark Tracker </b>

This main module used to do the tracking for both 2D and 3DA-2D facial landmark. To use : 
  
  python testRealtime.py
  
Also set the is3D in that file to True to have 3D points, False otherwise. Other configuration including filename of video can be set on the configuration.py file on src. Especially to use webcam or the video input. 

Example video : 

![2D Facial Landmark Tracking](trumpShort.gif)
![3DA-2D Facial landmark Tracking](obamaShort.gif)

Real time example : 

[![IMAGE ALT TEXT](http://img.youtube.com/vi/iUwJQelqYV4/0.jpg)](http://www.youtube.com/watch?v=iUwJQelqYV4 "Demo video of facial tracking")
  
<b> Facial Localisation </b>

This module can be used independently to localise facial points from still image. To use : 

  python facial_localiser.py 
  
Set the is3D in that to be True/False and the filename on the configuration.py.

Some examples : 

2D Facial landmark : 

![Localisation example of 2D landmark](2d.png)

3DA-2D Facial landmark : 

python train.py

![Localisation example of 3DA-2D landmark](3d.png)

Citation : 

D.  Aspandi,   O.  Martinez,   F.  Sukno,   X.  Binefa,   Fully  end-to-endcomposite  recurrent  convolution  network  for  deformable  facial  track-ing  in  the  wild,   in:    2019  14th  IEEE  International  Conference  onAutomatic   Face   Gesture   Recognition   (FG   2019),   2019,   pp.   1–8.doi:10.1109/FG.2019.8756630.
