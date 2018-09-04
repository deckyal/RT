'''
Created on Apr 26, 2018

@author: deckyal
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np 

class face_classifier_simple():
    
    def __init__(self,image_size = 24,batch_size = 8):
         
        self.image_size = image_size
        self.batch_size = batch_size
    
    def build(self):

        x = tf.placeholder(tf.float32,[None,68,self.image_size,self.image_size,3],name="input")
        y = tf.placeholder(tf.float32,[None,1],name="GT")
        
        #outputCNNB = np.zeros([self.batch_size,16*3*68])
        outputCNNB = []
        
        for i in range(self.batch_size) :
            
            #outputCNN = []
            
            part1 = slim.conv2d(x[i],8,3,activation_fn=None,scope="CNN1",reuse=tf.AUTO_REUSE)
            part1 = slim.dropout(part1,0.8,scope="DO1")
            part2 = slim.conv2d(part1,2,3,activation_fn=None,scope="CNN2",reuse=tf.AUTO_REUSE)
            '''print tf.reshape(part2, [-1])
            print outputCNN[k,:].shape
            print outputCNN[k].shape
            print tf.reshape(part2, [-1])
            outputCNN[k,:] = tf.reshape(part2, [-1])
            #outputCNN.append(tf.reshape(part2, [-1]))'''
            
            outputCNNB.append(tf.reshape(part2,[-1]))
        
        outputCNNB = tf.stack(outputCNNB)
        
        #print(outputCNNB)
        
        pred = slim.fully_connected(outputCNNB, 1, activation_fn=None, scope='Bottleneck_out', reuse=False)
            
        
        return x,y,pred