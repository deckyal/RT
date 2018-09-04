'''
Created on Jan 29, 2018

@author: deckyal
'''

from operator import truediv
import tensorflow as tf
import random 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re
import glob,os
import utils
from datetime import datetime
import inception_resnet_v1
from config import *
from random import randint

splitting = True

@tf.RegisterGradient("ZeroGrad")
def _zero_grad(unused_op, grad):
    return tf.zeros_like(grad)

def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1, 1)

def selu(z,
         scale=1.0507009873554804934193349852946,
         alpha=1.6732632423543772848170429916717):
    return scale * tf.where(z >= 0.0, z, alpha * tf.nn.elu(z))


def f1(x): return x
def f2(x): return x

class frameTracking():
    
    def __init__(self, batch_sequence, n_sequence, crop_size,channels,n_neurons = 100,n_outputs = 4,learning_rate = .01,test=False,model_name="model",dataType=1,n_adder=2,CNNTrainable = True,realTime = False):
        self.batch_size = batch_sequence
        self.seq_length = n_sequence 
        self.channels = channels
        self.crop_size = crop_size 
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        self.adder = n_adder;
        self.learning_rate = learning_rate
        self.test = test
        self.model_name = model_name
        self.dataType = dataType
        self.realTime = realTime
        
        with tf.variable_scope("LSTM1") as scope : 
            self.lstm = tf.contrib.rnn.BasicLSTMCell(self.n_neurons)
            self.state = self.lstm.zero_state(self.batch_size, tf.float32)
        
        if useDoubleLSTM: 
            with tf.variable_scope("LSTM2") as scope :
                self.lstm2 = tf.contrib.rnn.BasicLSTMCell(self.n_neurons)
                self.state2 = self.lstm.zero_state(self.batch_size, tf.float32)
        
        with tf.device("/cpu:0") : 
            with tf.variable_scope("Variable_Regress_Regularization") : 
                self.w_fc_o_r = tf.Variable(tf.random_normal([1792,self.n_outputs]),name="weights_regress_reg",dtype=tf.float32)
                self.b_fc_o_r = tf.Variable(tf.random_normal([self.n_outputs]),name="biases_regress_reg",dtype=tf.float32)
                
            '''with tf.variable_scope("Variables") as scope :      
                self.w_fc1 = tf.Variable(tf.random_normal([(self.crop_size/(2*2*2*2))*(self.crop_size/(2*2*2*2))*256,512], stddev=1e-4),name="weights_regress1",dtype=tf.float32,trainable=CNNTrainable)
                self.b_fc1 = tf.Variable(tf.random_normal([512], stddev=1e-4),name="biases_regress1",dtype=tf.float32)
            '''    
            with tf.variable_scope("Variable_LSTM") as scope :#+1792
                self.w_fc_o = tf.Variable(tf.random_normal([self.n_neurons,self.n_outputs]),name="weights_regress",dtype=tf.float32)
                self.b_fc_o = tf.Variable(tf.random_normal([self.n_outputs]),name="biases_regress",dtype=tf.float32)
                
                if splitting : 
                    
                    self.w_fc_o_eyes = tf.Variable(tf.random_normal([self.n_neurons,24]),name="weights_regress_eyes",dtype=tf.float32)
                    self.b_fc_o_eyes = tf.Variable(tf.random_normal([24]),name="biases_regress_eyes",dtype=tf.float32)
                    
                    self.w_fc_o_ebrows = tf.Variable(tf.random_normal([self.n_neurons,20]),name="weights_regress_ebrows",dtype=tf.float32)
                    self.b_fc_o_ebrows = tf.Variable(tf.random_normal([20]),name="biases_regress_ebrows",dtype=tf.float32)
                    
                    self.w_fc_o_nose = tf.Variable(tf.random_normal([self.n_neurons,18]),name="weights_regress_nose",dtype=tf.float32)
                    self.b_fc_o_nose = tf.Variable(tf.random_normal([18]),name="biases_regress_nose",dtype=tf.float32)
                    
                    self.w_fc_o_mouths = tf.Variable(tf.random_normal([self.n_neurons,40]),name="weights_regress_mouth",dtype=tf.float32)
                    self.b_fc_o_mouths = tf.Variable(tf.random_normal([40]),name="biases_regress_mouth",dtype=tf.float32)
                    
                    self.w_fc_o_edge = tf.Variable(tf.random_normal([self.n_neurons,34]),name="weights_regress_edge",dtype=tf.float32)
                    self.b_fc_o_edge = tf.Variable(tf.random_normal([34]),name="biases_regress_edge",dtype=tf.float32)
                
                '''#lst02self.w_fc_o = tf.Variable(tf.random_normal([self.n_neurons,self.n_neurons/2]),name="weights_regress",dtype=tf.float32)
                self.b_fc_o = tf.Variable(tf.random_normal([self.n_neurons/2]),name="biases_regress",dtype=tf.float32)
                
                self.w_fc_o_2 = tf.Variable(tf.random_normal([self.n_neurons/2,self.n_outputs]),name="weights_regress",dtype=tf.float32)
                self.b_fc_o_2 = tf.Variable(tf.random_normal([self.n_outputs]),name="biases_regress",dtype=tf.float32)'''
            
            with tf.variable_scope("Variable_Embed") as scope :
                #self.w_fce = tf.Variable(tf.random_normal([(self.crop_size/(2*2*2*2))*(self.crop_size/(2*2*2*2))*256*2,512], stddev=1e-4),name="weights_embedd",dtype=tf.float32,trainable=CNNTrainable)
                self.w_fce = tf.Variable(tf.random_normal([1792*2,self.n_neurons*2], stddev=1e-4),name="weights_embedd",dtype=tf.float32,trainable=CNNTrainable)
                self.b_fce = tf.Variable(tf.random_normal([self.n_neurons*2], stddev=1e-4),name="biases_embed",dtype=tf.float32)
                
            with tf.variable_scope("additional") as scope : 
                
                self.num_nans_grads = tf.Variable(1.0, name='num_nans_grads',trainable=False)
                
    def buildModel(self) :
        dataType = self.dataType#3
        n_o = self.n_outputs
        r_image_size = 128
        
        with tf.name_scope("inputs_train") : 
            x = tf.placeholder(tf.float32, shape = [None,self.seq_length,imHeight,imWidth,self.channels], name = "X")
            y = tf.placeholder(tf.float32,[None,self.seq_length,self.n_outputs+self.adder])
            z = tf.placeholder(tf.bool)  # Can be any computed boolean expression.#tf.placeholder(tf.int32,shape = [1],name="UseGT")
            initial_BB = tf.placeholder(tf.float32, shape = [None,4], name = "Initial_BB")
            c_state = tf.placeholder(tf.float32,shape = [None,self.n_neurons])
            h_state = tf.placeholder(tf.float32,shape = [None,self.n_neurons])
            c_state2 = tf.placeholder(tf.float32,shape = [None,self.n_neurons])
            h_state2 = tf.placeholder(tf.float32,shape = [None,self.n_neurons])
            z2 = tf.placeholder(tf.bool)
        
        keep_probability = .75

        network = inception_resnet_v1
        
        #LSTMState = tf.nn.rnn_cell.LSTMStateTuple(c_state, h_state)
        if self.test == False : 
            LSTMState = self.lstm.zero_state(self.batch_size, tf.float32)
            if useFullModel or useDoubleLSTM: 
                LSTMState2 = self.lstm2.zero_state(self.batch_size, tf.float32)
        else : 
            LSTMState = tf.nn.rnn_cell.LSTMStateTuple(c_state, h_state)
            if useFullModel or useDoubleLSTM: 
                LSTMState2 = tf.nn.rnn_cell.LSTMStateTuple(c_state2, h_state2)
        #lst02 LSTMState2 = self.lstm2.zero_state(self.batch_size, tf.float32)
        
        rnn_outputs_temp, rnn_o_p= [],[]
        
        #l_gates = np.zeros([self.seq_length,self.batch_size*4,self.n_neurons])
        l_gates, final_y, xy,sample_image = [],[],[],[]
        
        est_bb_list,est_bb_list_g = [],[]
        
        for i in range(self.seq_length) :
            
            if i == 0: 
                if self.test == False : 
                    LSTMState = self.lstm.zero_state(self.batch_size, tf.float32)
                    if useFullModel or useDoubleLSTM: 
                        LSTMState2 = self.lstm2.zero_state(self.batch_size, tf.float32)
                    t_lb = tf.maximum(y[:,i],0)
                    #lst02 LSTMState2 = self.lstm2.zero_state(self.batch_size, tf.float32)
                else : 
                    #if this is testing, get the last predicted result as gt.
                    t_lb = tf.maximum(y[:,self.seq_length-1],0)
            else : 
                
                #rnn_o_p[i-1] = tf.cond(z, lambda :f1(y[:,i-1]), lambda :f2(rnn_o_p[i-1]))
                
                def if_true():
                    return tf.maximum(y[:,i-1],0)
                
                def if_false():
                    return tf.maximum(tf.stack(rnn_o_p[i-1]),0)
                
                t_lb = tf.cond(z, if_true, if_false)
                #t_lb = tf.maximum(tf.stack(rnn_o_p[i-1]),0)
                     
            the_images = []
            
            #Now subsetting manually. The last frame and current frame
            y_c_b, r_c_w, r_c_h,si_a = [],[],[],[]
            x_min_a, y_min_a = [],[]
                    
            for j in range(self.batch_size):
                #subset_images = tf.zeros([self.batch_size,crop_size,crop_size,self.channels])
                subset_images = []
                
                if i == 0: 
                    indexBefore = i;
                else :
                    indexBefore = i-1;
                    
                if dataType == 0:
                    lImage = [x[indexBefore][j],x[i][j]]
                elif dataType in [1,2,3,4]:
                    lImage = [x[j][indexBefore],x[j][i]]
                    
                 
                x_min = 0; y_min = 0;x_max = 0; y_max = 0;
                
                #get the ratio of image to be resized and alter the ground truth to work in that resized image 
                
                ratioCropWidth = 0
                ratioCropHeight = 0
                
                if dataType in [0]: 
                    t = t_lb[j]
                elif dataType in [2,4]: 
                    if self.realTime and i == 0 : 
                        t0,t1,t2,t3 = initial_BB[j,0],initial_BB[j,2],initial_BB[j,1],initial_BB[j,3]
                    else : 
                        t0,t1,t2,t3 = utils.get_bb_tf(t_lb[j,0:int(n_o/2)], t_lb[j,int(n_o/2):],68,random.uniform(-.25,.25),random.uniform(-.25,.25),random.uniform(-.25,.25),random.uniform(-.25,.25),random.uniform(-.25,.25))
                elif dataType in [1,3]: 
                    #input is  be x1,x2,y1,y2
                    #change to be x1,y1,x2,y2
                    t0,t1,t2,t3 = t_lb[j,0],t_lb[j,2],t_lb[j,1],t_lb[j,3]
                    
                for tImage in lImage :
                    
                    #print tImage.shape
                    # we assume that t0 is minx, t1 is miny
                    # also, t2 is maxx, t3 is maxy
                    
                    #make the black picture of the enlarged 
                    #croppedImage = np.zeros([(t2-t0)*2, (t3-t1)*2]) 
                     
                    l_x = (t2-t0)/2; l_y = (t3-t1)/2
                    
                    x_min = tf.maximum(t0 - l_x,0); y_min = tf.maximum(t1 - l_y,0);
                    x1,y1 = tf.cast(x_min,tf.int32), tf.cast(y_min,tf.int32)
                    
                    x2,y2 = tf.cast(tf.minimum(t2 + l_x,imWidth),tf.int32), tf.cast(tf.minimum(t3 + l_y,imHeight),tf.int32)
                    
                    #x2 =  tf.cond(tf.less_equal(x2, tf.constant(0)),lambda :f(tf.constant(crop_size)),lambda : f(x2))
                    #y2 =  tf.cond(tf.less_equal(y2, tf.constant(0)),lambda :f(tf.constant(crop_size)),lambda : f(y2))
                    
                    croppedImage = tImage[y1:y2,x1:x2];
                    #xy.append(t_lb)

                    ratioCropWidth = truediv(crop_size,abs(x1-x2))
                    ratioCropHeight = truediv(crop_size,abs(y1-y2))
                    
                    subset_images.append(tf.image.resize_images(croppedImage, (crop_size,crop_size)))
                    si_a.append(tf.image.resize_images(croppedImage, (crop_size,crop_size)))
                    
                the_images.append(tf.stack(subset_images))
                
                #input is  be x1,x2,y1,y2
                #Correct the ground truth to work in cropped image
                #input is [x1,y1,x2,y2]
                #          0 ,1 ,2 ,3
                #we want it to be [x1,x2,y1,y2]
                #UPDATED, the input of ground truth now is x1,x2,y1,y2. Has been changed on the face_data_track.py
                
                x_1 = tf.subtract(tf.cast(y[j,i,0:int(n_o/2)],tf.float32),x_min)
                y_1 = tf.subtract(tf.cast(y[j,i,int(n_o/2):],tf.float32),y_min)
                
                x_min_a.append(x_min)
                y_min_a.append(y_min)
                
                #Correct the resiszed image to resized cropped 
                x_2 = tf.multiply(x_1,tf.cast(ratioCropWidth,tf.float32))
                y_2 = tf.multiply(y_1,tf.cast(ratioCropHeight,tf.float32))
                
                r_c_h.append(ratioCropHeight)
                r_c_w.append(ratioCropWidth)
                
                y_c_b.append(tf.concat([x_2,y_2],0))
                #si_a.append(tf.image.resize_images(croppedImage, (crop_size,crop_size)))
                #print y_c.shape
                
            sample_image.append(tf.stack(si_a))
            final_y.append(tf.stack(y_c_b))
            t_im = tf.stack(the_images)
            
            print(("Seqlength_train : {}".format(i)))
            t_out = []
            for j in range(0,2) :
                #if (i>0) : 
                    #scope.reuse_variables()

                pred= network.inference(t_im[:,j], keep_probability,phase_train=z2,bottleneck_layer_size=136,reuse=tf.AUTO_REUSE)
                
                t_out.append(pred)    
                #print conv4_flatten.shape
            #Now merging the two 
            
            difference_cnn = tf.stack(t_out[0] - t_out[1])
            
            merged_cnn = tf.concat([t_out[0],t_out[1]],1)
            
            #embedding 
            merged = selu(tf.matmul(merged_cnn,self.w_fce + self.b_fce))
            
            with tf.variable_scope("LSTM") as scope:
                
                with tf.variable_scope("LSTM1") as scope:
                    output, LSTMState= self.lstm(merged, LSTMState)
                     
                if useDoubleLSTM:       
                    with tf.variable_scope("LSTM2") as scope:
                        output, LSTMState2 = self.lstm2(output, LSTMState2)
                
                comb_lstm_cnn = tf.concat([merged,output],1)
                
                if not splitting : 
                    #o = tf.matmul(comb_lstm_cnn,self.w_fc_o) + self.b_fc_o
                    o = tf.matmul(output,self.w_fc_o) + self.b_fc_o
                else : 
                    o1 = tf.matmul(output,self.w_fc_o_edge) + self.b_fc_o_edge
                    o2 = tf.matmul(output,self.w_fc_o_ebrows) + self.b_fc_o_ebrows
                    o3 = tf.matmul(output,self.w_fc_o_nose) + self.b_fc_o_nose
                    o4 = tf.matmul(output,self.w_fc_o_eyes) + self.b_fc_o_eyes
                    o5 = tf.matmul(output,self.w_fc_o_mouths) + self.b_fc_o_mouths
                    
                    print(o1,o2)
                    o = tf.concat([o1[:,:12],o2[:,:10],o3[:,:9],o4[:,:20],o5[:,:17], o1[:,12:],o2[:,10:],o3[:,9:],o4[:,20:],o5[:,17:]],2)
                    print(o)
                    
                    
                    
                    self.w_fc_o_eyes = tf.Variable(tf.random_normal([self.n_neurons,24]),name="weights_regress_eyes",dtype=tf.float32)
                    self.b_fc_o_eyes = tf.Variable(tf.random_normal([24]),name="biases_regress_eyes",dtype=tf.float32)
                    
                    self.w_fc_o_ebrows = tf.Variable(tf.random_normal([self.n_neurons,20]),name="weights_regress_ebrows",dtype=tf.float32)
                    self.b_fc_o_ebrows = tf.Variable(tf.random_normal([20]),name="biases_regress_ebrows",dtype=tf.float32)
                    
                    self.w_fc_o_nose = tf.Variable(tf.random_normal([self.n_neurons,18]),name="weights_regress_nose",dtype=tf.float32)
                    self.b_fc_o_nose = tf.Variable(tf.random_normal([18]),name="biases_regress_nose",dtype=tf.float32)
                    
                    self.w_fc_o_mouths = tf.Variable(tf.random_normal([self.n_neurons,40]),name="weights_regress_mouth",dtype=tf.float32)
                    self.b_fc_o_mouths = tf.Variable(tf.random_normal([40]),name="biases_regress_mouth",dtype=tf.float32)
                    
                    self.w_fc_o_edge = tf.Variable(tf.random_normal([self.n_neurons,34]),name="weights_regress_edge",dtype=tf.float32)
                    self.b_fc_o_edge = tf.Variable(tf.random_normal([34]),name="biases_regress_edge",dtype=tf.float32)
                
                #o = tf.matmul(o,self.w_fc_o_2) + self.b_fc_o_2
                
                rnn_outputs_temp.append(o)
                
                o_t = []
                for j in range(self.batch_size):
                    #Correct the resiszed image to resized cropped 
                    x_1 = tf.multiply(o[j,0:int(n_o/2)],tf.cast(tf.divide(1.0,r_c_w[j]),tf.float32))
                    y_1 = tf.multiply(o[j,int(n_o/2):],tf.cast(tf.divide(1.0,r_c_h[j]),tf.float32))
                    
                    x_2 = tf.add(x_1, x_min_a[j])
                    y_2  = tf.add(y_1,y_min_a[j])
                    
                    o_t.append(tf.concat([x_2,y_2],0))
                    
                rnn_o_p.append(tf.stack(o_t)) #in seq, batch 
        
        states = tf.transpose(rnn_outputs_temp, [1,0,2], name = "result_unwrapped_stats")
        states_to_ori = tf.transpose(rnn_o_p, [1,0,2], name = "result_unwrapped_states_to_ori")
        
        
        final_y = tf.stack(final_y)
        final_y = tf.transpose(final_y,[1,0,2])

        
        sample_image = tf.stack(sample_image)
        sample_image = tf.transpose(sample_image,[1,0,2,3,4])
        
        print("Making loss and optimizer also t_op")
         
        #loss_bb = tf.reduce_mean(tf.sqrt(tf.square(tf.stack(est_bb_list_g) - tf.stack(est_bb_list))))
        loss_bb = 1
        
        #loss = tf.reduce_mean(tf.sqrt(tf.square(states[:,1:,:] - final_y[:,1:,self.adder:self.adder+self.n_outputs]))) #+ loss_bb
        
        #loss = tf.reduce_mean(tf.abs(states[:,1:,:] - final_y[:,1:,self.adder:self.adder+self.n_outputs]))#+loss_bb
        loss = tf.reduce_mean(tf.abs(states[:,:,:] - final_y[:,:,self.adder:self.adder+self.n_outputs]))#+0.1*loss_bb
        #loss = utils.calcLandmarkErrorListTF(states, final_y[:,:,self.adder:self.adder+self.n_outputs])
        
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.00001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
        10000, 0.1, staircase=True)
        
        '''starter_learning_rate = 0.001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
        10000, 0.96, staircase=True)'''
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            training_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)#tf.train.MomentumOptimizer(0.02, momentum=0.01)
            
        if useDoubleLSTM: 
            return x,c_state,c_state2,h_state,h_state2,y,z,loss, training_op,o,states,states_to_ori,l_gates,xy,final_y,sample_image,LSTMState,z2,loss_bb,initial_BB,LSTMState2
        else : 
            return x,c_state,c_state2,h_state,h_state2,y,z,loss, training_op,o,states,states_to_ori,l_gates,xy,final_y,sample_image,LSTMState,z2,loss_bb,initial_BB
        #return x,y,loss, training_op,o,states,l_gates
        
    def check_numerics_with_exception(self,grad, var):
        try:
            tf.check_numerics(grad, message='Gradient %s check failed, possible NaNs' % var.name)
        except:
            return tf.constant(False, shape=())
        else:
            return tf.constant(True, shape=()) 
        
                
    def buildTrainModel(self) :
        dataType = self.dataType#3
        n_o = self.n_outputs
        
        with tf.name_scope("inputs_train") : 
            x = tf.placeholder(tf.float32, shape = [None,self.seq_length,imHeight,imWidth,self.channels], name = "X")
            y = tf.placeholder(tf.float32,[None,self.seq_length,self.n_outputs+self.adder])
            z = tf.placeholder(tf.bool)  # Can be any computed boolean expression.#tf.placeholder(tf.int32,shape = [1],name="UseGT")
            c_state = tf.placeholder(tf.float32,shape = [None,self.n_neurons])
            h_state = tf.placeholder(tf.float32,shape = [None,self.n_neurons])
            c_state2 = tf.placeholder(tf.float32,shape = [None,self.n_neurons])
            h_state2 = tf.placeholder(tf.float32,shape = [None,self.n_neurons])
            phase_train = tf.placeholder(tf.bool)
        
        keep_probability = .75
    
        if(self.seq_length > 33) : 
            checkNan = False
        else : 
            checkNan = True
            
        network = inception_resnet_v1
         
        LSTMState = self.lstm.zero_state(self.batch_size, tf.float32)
        if useFullModel or useDoubleLSTM: 
            LSTMState2 = self.lstm2.zero_state(self.batch_size, tf.float32)
        
        rnn_outputs_temp, rnn_o_p= [],[]
        
        #l_gates = np.zeros([self.seq_length,self.batch_size*4,self.n_neurons])
        l_gates, final_y, xy= [],[],[]
        #sample_image  = []
        est_bb_list,est_bb_list_g = [],[]
                        
        if True : 
            l_cd,rv = utils.get_list_heatmap(0,None,crop_size/2,crop_size/2,.1)
            #print(l_cd)
            scaler = 255/np.max(rv)
        
        for i in range(self.seq_length-1) :
            print("seq : ",i)
            
            if i == 0:  
                LSTMState = self.lstm.zero_state(self.batch_size, tf.float32)
                if useFullModel or useDoubleLSTM: 
                    LSTMState2 = self.lstm2.zero_state(self.batch_size, tf.float32)
                t_lb = tf.maximum(y[:,i],0)
            else :
                t_lb = tf.cond(z, lambda :f1(tf.maximum(y[:,i-1],0)), lambda :f2(tf.maximum(tf.stack(rnn_o_p[i-1]),0)))
                 
            the_images = []
            
            #Now subsetting manually. The last frame and current frame
            y_c_b, r_c_w, r_c_h,si_a = [],[],[],[]
            x_min_a, y_min_a = [],[]
                    
            for j in range(self.batch_size):
                print("Batch : ",j)
                #subset_images = tf.zeros([self.batch_size,crop_size,crop_size,self.channels])
                subset_images = []
                
                '''if i == 0: 
                    indexAfter = i;
                else :
                    indexAfter = i+1;'''
                indexAfter = i+1;
                    
                if dataType == 0:
                    lImage = [x[i][j],x[indexAfter][j]]
                elif dataType in [1,2,3,4]:
                    lImage = [x[j][i],x[j][indexAfter]]
                 
                x_min = 0; y_min = 0;x_max = 0; y_max = 0;
                #get the ratio of image to be resized and alter the ground truth to work in that resized image 
                
                ratioCropWidth = 0
                ratioCropHeight = 0
                
                t0,t1,t2,t3 = utils.get_bb_tf(t_lb[j,0:int(n_o/2)], t_lb[j,int(n_o/2):],68,random.uniform(-.25,.25),random.uniform(-.25,.25),random.uniform(-.25,.25),random.uniform(-.25,.25),random.uniform(-.25,.25))
                
                for tImage in lImage :
                    
                    l_x = (t2-t0)/2; l_y = (t3-t1)/2
                    
                    x_min = tf.maximum(t0 - l_x,0); y_min = tf.maximum(t1 - l_y,0);
                    x1,y1 = tf.cast(x_min,tf.int32), tf.cast(y_min,tf.int32)
                    x2,y2 = tf.cast(tf.minimum(t2 + l_x,imWidth),tf.int32), tf.cast(tf.minimum(t3 + l_y,imHeight),tf.int32)

                    ratioCropWidth = truediv(crop_size,abs(x1-x2))
                    ratioCropHeight = truediv(crop_size,abs(y1-y2))
                    
                    x_1i = tf.subtract(tf.cast(t_lb[j,0:int(n_o/2)],tf.float32),x_min)
                    y_1i = tf.subtract(tf.cast(t_lb[j,int(n_o/2):],tf.float32),y_min)
                    
                    #Correct the resiszed image to resized cropped 
                    x_2i = tf.multiply(x_1i,tf.cast(ratioCropWidth,tf.float32))
                    y_2i = tf.multiply(y_1i,tf.cast(ratioCropHeight,tf.float32))
                    
                    y2_y1 = tf.cond(y2-y1 <=0,lambda:f1(tf.constant(True)), lambda:f2(tf.constant(False)))
                    x2_x1 = tf.cond(x2-x1 <=0,lambda:f1(tf.constant(True)), lambda:f2(tf.constant(False)))
                    
                    t_image = tf.cond(tf.logical_or(y2_y1,x2_x1), lambda :f1(tf.random_uniform((crop_size,crop_size,self.channels))), lambda :f2(tf.image.resize_images(tImage[y1:y2,x1:x2], (crop_size,crop_size))))
                    
                    if addChannel and False: 
                        #newChannel = tf.constant(0,shape=t_image[:,:,0].shape,dtype=tf.float32)# tf.Variable(tf.zeros(b_channel.shape),trainable=False,name="additional_channel")
                        
                        listIndices = []#np.zeros([68*len(l_cd),2])
                        listValues = np.zeros([68*len(l_cd)])
                        shape = [crop_size,crop_size]
                        
                        #print(shape)
                        
                        indexer = 0
                        for iter in range(68) :
                            #print(t_lb[j,iter],iter)
                            #ix,iy = tf.cast(x_2i[iter],tf.int32)+randint(0,2),tf.cast(y_2i[iter],tf.int32)+randint(0,2)
                            ix,iy = tf.cast(x_2i[iter],tf.int32),tf.cast(y_2i[iter],tf.int32)
                            #Now drawing given the center
                            for iter2 in range(len(l_cd)) : 
                                t_iy = tf.maximum(tf.minimum(iy+l_cd[iter2][0],crop_size-1),0)
                                t_ix = tf.maximum(tf.minimum(ix+l_cd[iter2][1],crop_size-1),0)
                                
                                #tf.assign(newChannel[50, 50],255)
                                #listIndices[indexer,0]=t_iy;listIndices[indexer,1]=t_ix;
                                listIndices.append([t_iy,t_ix])
                                listValues[indexer] =  rv[iter2]*scaler
                                #tf.assign(newChannel[t_iy, t_ix],rv[iter2]*scaler)#int(heatmapValue/2 + rv[iter2] * heatmapValue)
                                indexer+=1
                        
                        #print(tf.convert_to_tensor(listIndices))
                        delta = tf.SparseTensor(tf.convert_to_tensor(listIndices),listValues, shape)
                        t =  tf.sparse_tensor_to_dense(delta,0,False)
                        
                        #print("t : ",t)
                        newChannel = t_image[:,:,3] + tf.cast(t,tf.float32)
                        
                        #print(newChannel[0,0])
                        
                        #toShow = cv2.resize(cv2.merge((b_channel, newChannel,newChannel, newChannel)),(imWidth,imHeight)) 
                        #cv2.imshow('test',toShow)
                        #cv2.waitKey(0)  
                        
                        t_image = tf.stack([t_image[:,:,0],t_image[:,:,1],t_image[:,:,2], newChannel],axis = 2)
                        
                        #print("4d : ",t_image.shape)
                        
                    subset_images.append(t_image)
                    #si_a.append(tf.image.resize_images(t_image, (crop_size,crop_size)))
                    
                the_images.append(tf.stack(subset_images))
                
                x_1 = tf.subtract(tf.cast(y[j,indexAfter,0:int(n_o/2)],tf.float32),x_min)
                y_1 = tf.subtract(tf.cast(y[j,indexAfter,int(n_o/2):],tf.float32),y_min)
                
                x_min_a.append(x_min)
                y_min_a.append(y_min)
                
                #Correct the resiszed image to resized cropped 
                x_2 = tf.multiply(x_1,tf.cast(ratioCropWidth,tf.float32))
                y_2 = tf.multiply(y_1,tf.cast(ratioCropHeight,tf.float32))
                
                r_c_h.append(ratioCropHeight)
                r_c_w.append(ratioCropWidth)
                
                y_c_b.append(tf.concat([x_2,y_2],0))
                #si_a.append(tf.image.resize_images(croppedImage, (crop_size,crop_size)))
                #print y_c.shape
                
            #sample_image.append(tf.stack(si_a))
            final_y.append(tf.stack(y_c_b))
            t_im = tf.stack(the_images)
            
            print(("Seqlength_train : {}".format(i)))
            t_out = []
            
            for j in range(0,2) :
                #print(j)
                #print(t_im[:,j])
                if self.channels > 3 :
                    pred= network.inference(t_im[:,j], keep_probability,phase_train,136,.0,tf.AUTO_REUSE,True)
                else : 
                    pred= network.inference(t_im[:,j], keep_probability,phase_train,136,.0,tf.AUTO_REUSE)
                t_out.append(pred)
                    
            #Now merging the two
            merged_cnn = tf.concat([t_out[0],t_out[1]],1)
            
            #embedding 
            merged = selu(tf.matmul(merged_cnn,self.w_fce + self.b_fce))
            
            if experimental : 
                #concatenating with the previous output
                merged = tf.concat([merged,t_lb],1)
            
            #print merged.shaped
            with tf.variable_scope("LSTM") as scope:
                #if(i>0) : 
                    #scope.reuse_variables()
                with tf.variable_scope("LSTM1") as scope:
                    #output1, LSTMState= self.lstm(merged, LSTMState)
                    #output, LSTMState,i_g,j_g,f_g,o_g= self.lstm(merged, LSTMState)
                    output, LSTMState= self.lstm(merged, LSTMState)
                #,i_g,j_g,f_g,o_g 
                if useFullModel or useDoubleLSTM :       
                    
                    with tf.variable_scope("LSTM2") as scope:
                        output, LSTMState2 = self.lstm2(output, LSTMState2)
                 
                #Now regressing to the output space      #Output is on the cropped coordinate
                
                if not splitting : 
                    #o = tf.matmul(comb_lstm_cnn,self.w_fc_o) + self.b_fc_o
                    o = tf.matmul(output,self.w_fc_o) + self.b_fc_o
                else : 
                    o1 = tf.matmul(output,self.w_fc_o_edge) + self.b_fc_o_edge
                    o2 = tf.matmul(output,self.w_fc_o_ebrows) + self.b_fc_o_ebrows
                    o3 = tf.matmul(output,self.w_fc_o_nose) + self.b_fc_o_nose
                    o4 = tf.matmul(output,self.w_fc_o_eyes) + self.b_fc_o_eyes
                    o5 = tf.matmul(output,self.w_fc_o_mouths) + self.b_fc_o_mouths
                    
                    print(o1,o2)
                    #o = tf.concat([o1,o2,o3,o4,o5],1)
                    
                    o = tf.concat([o1[:,:12],o2[:,:10],o3[:,:9],o4[:,:20],o5[:,:17], o1[:,12:],o2[:,10:],o3[:,9:],o4[:,20:],o5[:,17:]],1)
                    print(o)
                
                rnn_outputs_temp.append(o)
                
                o_t = []
                for j in range(self.batch_size):
                    #Correct the resiszed image to resized cropped 
                    x_1 = tf.multiply(o[j,0:int(n_o/2)],tf.cast(tf.divide(1.0,r_c_w[j]),tf.float32))
                    y_1 = tf.multiply(o[j,int(n_o/2):],tf.cast(tf.divide(1.0,r_c_h[j]),tf.float32))
                    
                    x_2 = tf.add(x_1, x_min_a[j])
                    y_2  = tf.add(y_1,y_min_a[j])
                    
                    o_t.append(tf.concat([x_2,y_2],0))
                    
                rnn_o_p.append(tf.stack(o_t)) #in seq, batch 
        
        states = tf.transpose(rnn_outputs_temp, [1,0,2], name = "result_unwrapped_stats")
        states_to_ori = tf.transpose(rnn_o_p, [1,0,2], name = "result_unwrapped_states_to_ori")
        
        final_y = tf.stack(final_y)
        final_y = tf.transpose(final_y,[1,0,2])

        print("Making loss and optimizer also t_op")
         
        loss = tf.reduce_mean(tf.abs(states[:,:,:] - final_y[:,:,self.adder:self.adder+self.n_outputs]))#+0.1*loss_bb
        #loss = utils.calcLandmarkErrorListTF(states, final_y[:,:,self.adder:self.adder+self.n_outputs])
        
        global_step = tf.Variable(0, trainable=False,name="global_step")
        starter_learning_rate = 0.0001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
        10000, 0.1, staircase=True)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            opt = tf.train.AdamOptimizer(learning_rate = learning_rate)
            
            #If does not use GT, check if any Nan. if not useGT
            grads = opt.compute_gradients(loss)
            
            def fn_true_apply_grad(grads, global_step):
                apply_gradients_true = opt.apply_gradients(grads, global_step=global_step)
                return apply_gradients_true
        
            def fn_false_ignore_grad(grads, global_step):
                #print('batch update ignored due to nans, fake update is applied')
                g = tf.get_default_graph()
                with g.gradient_override_map({"Identity": "ZeroGrad"}):
                    clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in grads]
                    apply_gradients_false = opt.apply_gradients(clipped_gradients, global_step=global_step)
                    
                    '''for (grad, var) in grads:
                        tf.assign(var, tf.identity(var, name="Identity"))
                        apply_gradients_false = opt.apply_gradients(grads, global_step=global_step)'''
                return apply_gradients_false
            
            
            '''#Gradient clipping    
            grads_and_vars = optimizer.compute_gradients(loss)
            clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in grads_and_vars]
            #capped_gvs = [(tf.clip_by_value(grad,-thresold,thresold),var) for grad, var in grads_and_vars]
            #training_op = optimizer.minimize(loss)
            training_op = optimizer.apply_gradients(clipped_gradients)
            '''
            
            training_op = tf.cond(tf.is_inf(loss), lambda : fn_false_ignore_grad(grads, global_step), lambda :  fn_true_apply_grad(grads, global_step))
            
            if checkNan:# and not z: 
                ##############################
                check_all_numeric_op = tf.reduce_sum(tf.cast(tf.stack([tf.logical_not(self.check_numerics_with_exception(grad, var)) for grad, var in grads]), dtype=tf.float32))
                
                with tf.control_dependencies([tf.assign(self.num_nans_grads, check_all_numeric_op)]):
                    # Apply the gradients to adjust the shared variables.
                    
                    training_op = tf.cond(tf.equal(self.num_nans_grads, 0.), lambda : fn_true_apply_grad(grads, global_step), lambda :  fn_false_ignore_grad(grads, global_step))
                #################################
                
                
        if useFullModel or useDoubleLSTM : 
            return x,c_state,c_state2,h_state,h_state2,y,z,loss, training_op,o,states,states_to_ori,l_gates,xy,final_y,LSTMState,LSTMState2,phase_train#,sample_image
        else : 
            return x,c_state,c_state2,h_state,h_state2,y,z,loss, training_op,o,states,states_to_ori,l_gates,xy,final_y,LSTMState,phase_train#,sample_image
        #return x,y,loss, training_op,o,states,l_gates
        
        
        
    def buildTracker(self) :
        dataType = self.dataType#3
        n_o = self.n_outputs
        
        with tf.name_scope("inputs_train") : 
            x = tf.placeholder(tf.float32, shape = [None,self.seq_length,imHeight,imWidth,self.channels], name = "X")
            y = tf.placeholder(tf.float32,[None,self.seq_length,self.n_outputs+self.adder])
            initial_BB = tf.placeholder(tf.float32, shape = [None,4], name = "Initial_BB")
            c_state = tf.placeholder(tf.float32,shape = [None,self.n_neurons])
            h_state = tf.placeholder(tf.float32,shape = [None,self.n_neurons])
            c_state2 = tf.placeholder(tf.float32,shape = [None,self.n_neurons])
            h_state2 = tf.placeholder(tf.float32,shape = [None,self.n_neurons])
            z2 = tf.placeholder(tf.bool)
        
        keep_probability = .75
        network = inception_resnet_v1
        
        #LSTMState = tf.nn.rnn_cell.LSTMStateTuple(c_state, h_state)
        if self.test == False : 
            LSTMState = self.lstm.zero_state(self.batch_size, tf.float32)
            if useDoubleLSTM: 
                LSTMState2 = self.lstm2.zero_state(self.batch_size, tf.float32)
        else : 
            LSTMState = tf.nn.rnn_cell.LSTMStateTuple(c_state, h_state)
            if useDoubleLSTM: 
                LSTMState2 = tf.nn.rnn_cell.LSTMStateTuple(c_state2, h_state2)
        #lst02 LSTMState2 = self.lstm2.zero_state(self.batch_size, tf.float32)
        
        rnn_outputs_temp, rnn_o_p= [],[]
        
        #l_gates = np.zeros([self.seq_length,self.batch_size*4,self.n_neurons])
        l_gates, final_y, xy,sample_image = [],[],[],[]
        
        
        for i in range(1): #self.seq_length) :
            
            t_lb = tf.maximum(y[:,self.seq_length-1],0)   
            the_images = []
            #Now subsetting manually. The last frame and current frame
            
            y_c_b, r_c_w, r_c_h,si_a = [],[],[],[]
            x_min_a, y_min_a = [],[]
                    
            for j in range(self.batch_size):
                #subset_images = tf.zeros([self.batch_size,crop_size,crop_size,self.channels])
                subset_images = []
                lImage = [x[j][i],x[j][i+1]]
                 
                x_min = 0; y_min = 0;x_max = 0; y_max = 0;
                
                #get the ratio of image to be resized and alter the ground truth to work in that resized image
                ratioCropWidth = 0
                ratioCropHeight = 0
                
                if dataType in [0]: 
                    t = t_lb[j]
                elif dataType in [2,4]:
                    #input is x1,x2,y1,y2
                    #x1,y1,x2,y2
                    t0,t1,t2,t3 = initial_BB[j,0],initial_BB[j,2],initial_BB[j,1],initial_BB[j,3]
                elif dataType in [1,3]: 
                    #input is  be x1,x2,y1,y2
                    #change to be x1,y1,x2,y2
                    t0,t1,t2,t3 = t_lb[j,0],t_lb[j,2],t_lb[j,1],t_lb[j,3]
                    
                for tImage in lImage :
                     
                    l_x = (t2-t0)/2; l_y = (t3-t1)/2
                    
                    x_min = tf.maximum(t0 - l_x,0); y_min = tf.maximum(t1 - l_y,0);
                    x1,y1 = tf.cast(x_min,tf.int32), tf.cast(y_min,tf.int32)
                    x2,y2 = tf.cast(tf.minimum(t2 + l_x,imWidth),tf.int32), tf.cast(tf.minimum(t3 + l_y,imHeight),tf.int32)
                    
                    croppedImage = tImage[y1:y2,x1:x2];
                    #xy.append(t_lb)

                    ratioCropWidth = truediv(crop_size,abs(x1-x2))
                    ratioCropHeight = truediv(crop_size,abs(y1-y2))
                    
                    subset_images.append(tf.image.resize_images(croppedImage, (crop_size,crop_size)))
                    si_a.append(tf.image.resize_images(croppedImage, (crop_size,crop_size)))
                    
                the_images.append(tf.stack(subset_images))
                
                #input is  be x1,x2,y1,y2
                #Correct the ground truth to work in cropped image
                #input is [x1,y1,x2,y2]
                #          0 ,1 ,2 ,3
                #we want it to be [x1,x2,y1,y2]
                #UPDATED, the input of ground truth now is x1,x2,y1,y2. Has been changed on the face_data_track.py
                
                x_1 = tf.subtract(tf.cast(y[j,i,0:int(n_o/2)],tf.float32),x_min)
                y_1 = tf.subtract(tf.cast(y[j,i,int(n_o/2):],tf.float32),y_min)
                
                x_min_a.append(x_min)
                y_min_a.append(y_min)
                
                #Correct the resiszed image to resized cropped 
                x_2 = tf.multiply(x_1,tf.cast(ratioCropWidth,tf.float32))
                y_2 = tf.multiply(y_1,tf.cast(ratioCropHeight,tf.float32))
                
                r_c_h.append(ratioCropHeight)
                r_c_w.append(ratioCropWidth)
                
                y_c_b.append(tf.concat([x_2,y_2],0))
                #si_a.append(tf.image.resize_images(croppedImage, (crop_size,crop_size)))
                #print y_c.shape
                
            sample_image.append(tf.stack(si_a))
            final_y.append(tf.stack(y_c_b))
            t_im = tf.stack(the_images)
            
            print(("Seqlength_train : {}".format(i)))
            t_out = []
            for j in range(0,2) :
                if self.channels > 3 :
                    pred= network.inference(t_im[:,j], keep_probability,z2,136,.0,tf.AUTO_REUSE,True)
                else : 
                    pred= network.inference(t_im[:,j], keep_probability,z2,136,.0,tf.AUTO_REUSE)
                    
                #   pred= network.inference(t_im[:,j], keep_probability,phase_train=z2,bottleneck_layer_size=136,reuse=tf.AUTO_REUSE)
                t_out.append(pred)    
                
            #Now merging the two 
            difference_cnn = tf.stack(t_out[0] - t_out[1])
            merged_cnn = tf.concat([t_out[0],t_out[1]],1)
            
            #embedding 
            merged = selu(tf.matmul(merged_cnn,self.w_fce + self.b_fce))

            with tf.variable_scope("LSTM") as scope:
                with tf.variable_scope("LSTM1") as scope:
                    output, LSTMState= self.lstm(merged, LSTMState)
                    
                if useDoubleLSTM:       
                    with tf.variable_scope("LSTM2") as scope:
                        output, LSTMState2 = self.lstm2(output, LSTMState2)
                
                #Now regressing to the output space      #Output is on the cropped coordinate
                comb_lstm_cnn = tf.concat([merged,output],1)
            
                o1 = tf.matmul(output,self.w_fc_o_edge) + self.b_fc_o_edge
                o2 = tf.matmul(output,self.w_fc_o_ebrows) + self.b_fc_o_ebrows
                o3 = tf.matmul(output,self.w_fc_o_nose) + self.b_fc_o_nose
                o4 = tf.matmul(output,self.w_fc_o_eyes) + self.b_fc_o_eyes
                o5 = tf.matmul(output,self.w_fc_o_mouths) + self.b_fc_o_mouths
                
                o = tf.concat([o1[:,:12],o2[:,:10],o3[:,:9],o4[:,:20],o5[:,:17], o1[:,12:],o2[:,10:],o3[:,9:],o4[:,20:],o5[:,17:]],1)
            
                
                rnn_outputs_temp.append(o)
                
                o_t = []
                for j in range(self.batch_size):
                    #Correct the resiszed image to resized cropped 
                    x_1 = tf.multiply(o[j,0:int(n_o/2)],tf.cast(tf.divide(1.0,r_c_w[j]),tf.float32))
                    y_1 = tf.multiply(o[j,int(n_o/2):],tf.cast(tf.divide(1.0,r_c_h[j]),tf.float32))
                    
                    x_2 = tf.add(x_1, x_min_a[j])
                    y_2  = tf.add(y_1,y_min_a[j])
                    
                    o_t.append(tf.concat([x_2,y_2],0))
                    
                rnn_o_p.append(tf.stack(o_t)) #in seq, batch 
        
        states = tf.transpose(rnn_outputs_temp, [1,0,2], name = "result_unwrapped_stats")
        states_to_ori = tf.transpose(rnn_o_p, [1,0,2], name = "result_unwrapped_states_to_ori")
        
        
        final_y = tf.stack(final_y)
        final_y = tf.transpose(final_y,[1,0,2])

        
        sample_image = tf.stack(sample_image)
        sample_image = tf.transpose(sample_image,[1,0,2,3,4])
        
        if useDoubleLSTM: 
            return x,c_state,c_state2,h_state,h_state2,y,states,states_to_ori,l_gates,xy,final_y,sample_image,LSTMState,z2,initial_BB,LSTMState2
        else : 
            return x,c_state,c_state2,h_state,h_state2,y,states,states_to_ori,l_gates,xy,final_y,sample_image,LSTMState,z2,initial_BB
        #return x,y,loss, training_op,o,states,l_gates
        
