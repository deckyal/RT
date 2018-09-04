'''
Created on Apr 26, 2018

@author: deckyal
'''

import tensorflow as tf
import inception_resnet_v1
import tensorflow.contrib.slim as slim

class face_localiser():
    
    def __init__(self,image_size = 128,phase_train = True,channels = 3):
         
        self.image_size = image_size
        self.phase_train = phase_train
        self.channels = channels
    
    def build(self):
        keep_probability = .8

        x = tf.placeholder(tf.float32,[None,self.image_size,self.image_size,self.channels],name="input")
        y = tf.placeholder(tf.float32,[None,136],name="GT")
        
        network = inception_resnet_v1
        if self.channels > 3 :
            pred= network.inference(x,keep_probability,self.phase_train,128,.0,None,True) 
        else : 
            pred= network.inference(x,keep_probability,phase_train=self.phase_train)
        pred = slim.fully_connected(pred, 136, activation_fn=None, scope='Bottleneck_out', reuse=False)
            
        
        return x,y,pred