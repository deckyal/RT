# RT

This is our repository of our paper (Fully  end-to-end composite recurrent convolution network for deformable facial tracking in the wild). 

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

citation : 

D.  Latif,  O.  Pujol,  F.  Sukno,  and  X.  Binefa,  “Fully  end-to-end composite recurrent convolution network for deformable facial tracking in the wild,” in 2019 14th IEEE InternationalConference on Automatic Face Gesture Recognition (FG2019), In Press.
