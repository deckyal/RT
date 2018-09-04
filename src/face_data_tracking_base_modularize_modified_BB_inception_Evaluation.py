from operator import truediv
import tensorflow as tf
import numpy as np
import cv2
import utils
from datetime import datetime
from TrackerModifiedInception import frameTracking
from config import *
from pathlib import Path
import os
import glob
import bouncing_balls as b_b
from random import randint
from  face_localiser import face_localiser
from scipy.stats import multivariate_normal
import random 
import time
import gc

import math
from config import imWidth, imHeight

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_boolean("i_addChannel", False,
                        "Whether to train inception submodel variables.")
tf.flags.DEFINE_boolean("i_is3D", True,
                        "Whether to train inception submodel variables.")
tf.flags.DEFINE_integer("i_seqLength", 3,
                        "Whether to train inception submodel variables.")
tf.flags.DEFINE_boolean("i_upgrade",False,
                        "Whether to train inception submodel variables.")
tf.flags.DEFINE_boolean("i_continue", False,
                        "Whether to train inception submodel variables.")
tf.flags.DEFINE_integer("i_batch_number", 32,
                        "Whether to train inception submodel variables.")
tf.flags.DEFINE_integer("i_mode", 0,
                        "Whether to train inception submodel variables.")

#for traiing KP, step  : 
#1. Train on the localisation dataset 
#2. Train on the 2,4 up to 8 sequences. 

def training(dataType = 0,continuing = None):
    
    is3D = FLAGS.i_is3D
    addChannel = FLAGS.i_addChannel
    doUpgrade = FLAGS.i_upgrade
    
    if continuing is None : continuing = FLAGS.i_continue
    
    
    recover_lstm = False
    recover_cnn = True
    doSave = True
    toShowGT = False
    doAnalyze = False
    doTransformation = True
    channels = 3
    
    print(("Training datatype : {} continuing {}".format(dataType,continuing)))
    
    n_iterations = 1000
    
    batch_number =FLAGS.i_batch_number
    '''
    if useFullModel : 
        batch_number//=2
    '''
    if dataType == 4 :
        seq_length = 2
    else :  
        seq_length = FLAGS.i_seqLength#5;
    
    if (seq_length-1) %2 != 0 : 
        batch_size = 1 + serverAdder
    else : 
        batch_size = int(batch_number/(seq_length-1))
        batch_size += batch_size*serverAdder
    
    global_step =tf.Variable(0,trainable=False)
    root_log_dir = curDir+"tf_logs"
    logdir = curDir+"{}/run-{}/".format(root_log_dir,datetime.utcnow().strftime("%Y%m%d%H%M%S"))
    
    #0 is circle, 1 is bb, 2 is the kp (that we want). 3 is for bounding box. 
    #First get the list of input and bounding boxes. 
    n_adder = 2
    if dataType == 0: 
        name_save = "circle"
        n_o = 4
        
    elif dataType == 1:
        name_save = "bb"
        all_batch,all_labels = utils.get_bb_face(seq_length,True,"synthetic2/")
        #all_batch,all_labels = utils.get_bb_face(seq_length,False)
        all_labels = np.array(all_labels)
        
        batch_length = int(len(all_batch) / batch_size)
        n_o = 4
        n_adder = 0
    else : 
        name_save = "kp-transfer"
        err_file = "err"
        
        if is3D : 
            name_save += "-3D"
            err_file += "-3D"
        
        if useFullModel :
            name_save += "-full"
            err_file += "-full"
            
        if experimental : 
            name_save += "-exp"
            err_file += "-exp"
        
        if addChannel : 
            name_save += "-channel"
            err_file += "-channel"
            channels = 4
                
            
        if dataType == 3: 
            all_batch,all_labels,_ = utils.get_kp_face(seq_length,["300VW-Train"],False,3)
        elif dataType == 2: 
            if is3D : 
                all_batch,all_labels,_ = utils.get_kp_face(seq_length,["300VW-Train"],False,3,True)
            else : 
                all_batch,all_labels,_ = utils.get_kp_face(seq_length,["300VW-Train"],False,3)
        
        
        elif dataType == 4: 
            all_batch,all_labels,_ = utils.get_kp_face_localize(True,"toTest")
        
        batch_length = int(len(all_batch) / batch_size)
        n_o = 136
        n_adder = 0
    
    err_file += ('-'+str(int(seq_length-1)))
    err_file += '.txt'
    
    errFile = curDir + 'src/'+err_file
    f = open(errFile,'w+')
    f.write(' \n Error : ')
    f.close()
    
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "0"
    
    sess = tf.Session(config=config)
    
    g_t = tf.Graph() ## This is one graph
    g_l = tf.Graph() ## This is another graph
    sess = tf.InteractiveSession(graph = g_t,config = config)
    sess_l = tf.InteractiveSession(graph = g_l,config = config)
    
    
    with g_l.as_default():
        f = face_localiser(crop_size,True,3)
        x,y,pred = f.build()
        
        name_localiser = 'dt-inception'
        if is3D: 
            name_localiser+='-4D'
            
            
        if addChannel : 
            #err_name += "-4D"
            name_localiser+="-channel"
            channels = 4
        
        saver = tf.train.Saver()
        #saver.restore(sess,curDir + 'src/models/'+name_localiser)
        
        #saver.restore(sess_l, tf.train.latest_checkpoint(curDir + 'src/models/'+name_localiser))
        
        loss_g_l = tf.sqrt(tf.reduce_mean(tf.square(pred - y)))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            training_op_g_l = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss_g_l)#tf.train.MomentumOptimizer(0.02, momentum=0.01)
    
    
    start = time.time()
    
    with g_t.as_default():
        print ("*Initiating FT")
        ft = frameTracking(batch_size, seq_length, crop_size, channels, n_neurons, n_o,learning_rate=.01,test=False,model_name=name_save,dataType = dataType,n_adder=n_adder)
        print ("*Initiating builder")
        if True : #addChannel :
            if useFullModel or useDoubleLSTM:
                #x,c_state,c_state2,h_state,h_state2,y,z,loss, training_op,o,states,states_to_ori,l_gates,xy,final_y,sample_image,LSTMState,LSTMState2,phase_train
                x,c,h,c2,h2,y,pgt,loss, training_op, _,preds,preds_ori,the_gates,xy,finalY,_,_,phase_train= ft.buildTrainModel()
            else : 
                x,c,h,c2,h2,y,pgt,loss, training_op, _,preds,preds_ori,the_gates,xy,finalY,_,phase_train= ft.buildTrainModel()
        else : 
            
            if useFullModel or useDoubleLSTM: 
                x,c,c2,h,h2,y,pgt,loss, training_op, _,preds,preds_ori,the_gates,xy,finalY,_,phase_train,loss_bb,_,LSTMState2= ft.buildModel()
            else : 
                x,c,c2,h,h2,y,pgt,loss, training_op, _,preds,preds_ori,the_gates,xy,finalY,_,phase_train,loss_bb,_= ft.buildModel()
    
        reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionResnetV1") # regular expression
        reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
        restore_cnn = tf.train.Saver(reuse_vars_dict) # to restore all restore_variables_on_create
        
        reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Variable_LSTM|LSTM") # regular expression
        reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
        restore_lstm = tf.train.Saver(reuse_vars_dict)
        
        print ("Now summary saver")
        init = tf.global_variables_initializer()
        
        mse_summary = tf.summary.scalar('Loss',loss)
        file_writer = tf.summary.FileWriter(logdir,tf.get_default_graph())
        
        print ("Now training")
        saver = tf.train.Saver(max_to_keep=2,save_relative_paths=True,var_list=tf.trainable_variables(scope="(?!additional)"))
        
        #print(tf.trainable_variables(scope="(?!additional)"))
        #print(tf.trainable_variables())
    
        the_state = tf.contrib.rnn.BasicLSTMCell(n_neurons).zero_state(batch_size, tf.float32)
    
    end = time.time()
    print(end - start) 
    
    if continuing : 
        print((curDir + 'src/models/'+name_save))
        init.run(session = sess)
        name_to_upgrade = name_save;
        print("do upgrade : ",doUpgrade)
        if doUpgrade : 
            name_to_upgrade = name_save + "-"+ str(int((seq_length-1)/2))
            print("Upgrading from ",name_to_upgrade," to ",seq_length-1)
        else : 
            name_to_upgrade = name_save + "-"+ str(int(seq_length-1))
            print('resuming training to ',name_to_upgrade)
            
        saver.restore(sess, tf.train.latest_checkpoint(curDir + 'src/models/'+name_to_upgrade))    
        
    else : 
        init.run(session = sess)
        if recover_cnn : 
            if is3D :
                if not addChannel : 
                    restore_cnn.restore(sess,curDir + 'src/models/dt-inception-3D/dt-inception-3D-67')
                else : 
                    restore_cnn.restore(sess,curDir + 'src/models/dt-inception-3D-4D/dt-inception-3D-4D-4')
            elif addChannel :
                if useFullModel : 
                    restore_cnn.restore(sess,curDir + 'src/models/dt-inception-4D/dt-inception-4D-6')
                elif useDoubleLSTM: 
                    restore_cnn.restore(sess,curDir + 'src/models/dt-inception-4D/dt-inception-4D-6') 
                else : 
                    restore_cnn.restore(sess,curDir + 'src/models/dt-inception-4D/dt-inception-4D-6') 
                    
            else : 
                if useFullModel: 
                    restore_cnn.restore(sess,curDir + 'src/models/dt-inception/dt-inception-356')
                elif useDoubleLSTM:
                    restore_cnn.restore(sess,curDir + 'src/models/dt-inception/dt-inception-356')
                else : 
                    restore_cnn.restore(sess,curDir + 'src/models/dt-inception-full2/dt-inception-full-1')
        if recover_lstm : 
            restore_lstm.restore(sess,curDir + 'src/models/kp/kp-45')
    
    
    name_save += ('-'+str(int(seq_length-1)))
    
    start = time.time()
    for iteration in range(n_iterations) :
        
        print(("it-",iteration))
        counter, mean_error = 0,0
        cur_data = None
        
        for bt in range(0,batch_length) :
            stt = sess.run(the_state)
            
            if seq_length-1 == 2:
                p_own = 0
            else : 
                p_own = 0.25 * (np.log((seq_length-1)/2)/np.log(2))
            #p_own = 0
            
            prob_using_gt = np.asarray([utils.myRand(1, (1-p_own)*100, 0, (p_own)*100)])
            
            if prob_using_gt ==1 : 
                prob_using_gt = True 
            else : 
                prob_using_gt = False
                
            temp_batch,cBBPoints = [],[]
            
            for b_i in range(counter,counter+batch_size):
                #fetch the data for each batch
                t_b,y_b = [],[]
                
                sel = randint(0, 3)
                rad = randint(-3,3)#7
                
                for j_i in range(seq_length) :
                    
                    images = cv2.imread(all_batch[b_i][j_i])
                    y_t = all_labels[b_i][j_i].copy()
                    
                    #print sel
                    if doTransformation : 
                        if sel == 0 : 
                            out = [images,y_t]
                        elif sel == 1 : #Do flipping  
                            out = utils.transformation(images, y_t, 1, 1)
                        elif sel == 2 : #Do rotation
                            out = utils.transformation(images, y_t, 2, 5*rad)
                        elif sel == 3 : #Do occlusion 
                            out = utils.transformation(images, y_t, 3, 1)
                        images = out[0];y_t = out[1]
                    
                    if images is None : 
                        print((all_batch[b_i][j_i]))
                        
                    if addChannel:# and False: 

                        if is3D : 
                            add ="3D"
                        else : 
                            add ="2D"
                            
                        #get the recently calculated heatmap. If any use it, otherwise calculate it
                        tBase = os.path.basename(all_batch[b_i][j_i])
                        tName,tExt = os.path.splitext(tBase)
                        theDir =  os.path.dirname(all_batch[b_i][j_i])+"/heatmap-"+add+"/"
                        
                        if not os.path.exists(theDir):
                            os.makedirs(theDir)
                            
                        fName =theDir+tName+".npy"
                        #print(fName)
                        
                        b_channel,g_channel,r_channel = images[:,:,0],images[:,:,1],images[:,:,2]
                        
                        if os.path.isfile(fName): 
                            newChannel = np.load(fName)
                            #print("using saved npy")
                        else : 
                            
                            t0,t1,t2,t3 = utils.get_bb(y_t[0:int(n_o//2)], y_t[int(n_o//2):],68,False,random.uniform( .05, .25 ))
                            
                            #print(t0,t1,t2,t3)
                            #print(t2-t0,t3-t1)
                            
                            l_cd,rv = utils.get_list_heatmap(0,None,t2-t0,t3-t1,.1)
                            
                            #print(np.max(l_cd[:,0]),np.max(l_cd[:,1]))
                            
                            b_channel,g_channel,r_channel = images[:,:,0],images[:,:,1],images[:,:,2]
                            newChannel = b_channel.copy(); newChannel[:] = 0
                            
                            '''#Try drawing gausian image
                            mean = [0,0]; cov = [[1,0],[0,1]]
                            
                            tx = np.random.multivariate_normal(mean,cov,200).astype(int)
                            tx = np.unique(tx,axis=0)
                            rv = multivariate_normal.pdf(tx,mean = mean, cov = [1,1])'''
                            height, width,_ = images.shape
                            
                            scaler = 255/np.max(rv)
                            #addOne = randint(0,2),addTwo = randint(0,2)
                            for iter in range(68) :
                                #print(height,width)
                                ix,iy = int(y_t[iter])+randint(0,2),int(y_t[iter+68])+randint(0,2)
                                #Now drawing given the center
                                for iter2 in range(len(l_cd)) : 
                                    value = int(rv[iter2]*scaler)
                                    if newChannel[utils.inBound(iy+l_cd[iter2][0],0,height-1), utils.inBound(ix + l_cd[iter2][1],0,width-1)] < value : 
                                        newChannel[utils.inBound(iy+l_cd[iter2][0],0,height-1), utils.inBound(ix + l_cd[iter2][1],0,width-1)] = int(rv[iter2]*scaler)#int(heatmapValue/2 + rv[iter2] * heatmapValue)
                            
                            '''toShow = cv2.resize(cv2.merge((b_channel, newChannel,newChannel, newChannel)),(imWidth,imHeight)) 
                            cv2.imshow('test',toShow)
                            cv2.waitKey(0)'''
                            np.save(fName,newChannel)
                            
                        images = cv2.merge((b_channel, g_channel,r_channel, newChannel))  
                    
                    if addChannel : 
                        b_channel,g_channel,r_channel,n_channel = images[:,:,0],images[:,:,1],images[:,:,2],images[:,:,2].copy()
                        n_channel[:,:] = 0
                        images = cv2.merge((b_channel, g_channel,r_channel, n_channel))
                        
                    height, width,_ = images.shape
                
                    ratioHeight =truediv(imHeight,height)
                    ratioWidth =truediv(imWidth,width)
                    
                    
                    '''#Test to see the image and keypoints
                    t_image = images.copy()
                    #cv2.putText(t_image,'Test',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
                    for z22 in range(68) :
                        #cv2.circle(t_image,(int(y_t[z22]),int(y_t[z22+68])),3,(0,255,0))
                        if True : #z22 in range(60,68): 
                            cv2.putText(t_image,str(z22),(int(y_t[z22]),int(y_t[z22+68])),cv2.FONT_HERSHEY_SIMPLEX,.175,(255,255,255))
                            print(z22,int(y_t[z22]),int(y_t[z22+68]))
                        #cv2.circle(t_image,(int(listCenter[z][z22]),int(listCenter[z][z22+68])),3,(255,0,0))
                    cv2.imshow('test_input',t_image)
                    cv2.waitKey(0)'''
                    
                    
                    images = cv2.resize(images,(imWidth,imHeight)).copy()
                    
                    t_b.append(images)
                     
                    y_t[0:int(n_o/2)] = y_t[0:int(n_o/2)] * ratioWidth
                    y_t[int(n_o/2):] = y_t[int(n_o/2):] * ratioHeight
                    
                    
                    y_b.append(y_t)
                    
                temp_batch.append(t_b)
                cBBPoints.append(y_b)
                    
            counter+=batch_size
            
            x_batch = np.asarray(temp_batch)
            y_batch = np.asarray(cBBPoints).clip(0).copy()
            
            #print(x_batch.shape)
            #print(y_batch.shape)
            
            #print("Executing")
            #_,losses,l_bb,tg,fY,sI = sess.run([training_op,loss,loss_bb,the_gates,finalY,sampleImage],feed_dict = {x:x_batch, y:y_batch,c : stt.c,h : stt.h,c2 : stt.c,h2 : stt.h,pgt:prob_using_gt,z2:True})
            _,losses,tg,fY = sess.run([training_op,loss,the_gates,finalY],feed_dict = {x:x_batch, y:y_batch,pgt:prob_using_gt,phase_train:True})
            
            #print(end - start)
            #print("Done")
            '''print(fY.shape,sI.shape)
            
            image = np.array(sI[0,0], dtype = np.uint8)
            print(image[:,:,0])
            print("max",np.max(image))
            
            lP = fY[0,0]
            #print image
            print (lP)
            print (y_batch[0,0])
            #output is x1,x2,y1,y2
            #original ix x1,y1,x2,y2
            #cv2.rectangle(image,(lP[0],lP[2]),(lP[1],lP[3]),(255,0,255),1)
            cv2.imshow('test',image)
            cv2.waitKey(0)
            
            image = np.array(x_batch[0,0], dtype = np.uint8)
            lP = fY[0,0]
            #print image
            print (lP)
            print (y_batch[0,0])
            #output is x1,x2,y1,y2
            #original ix x1,y1,x2,y2
            #cv2.rectangle(image,(lP[0],lP[2]),(lP[1],lP[3]),(255,0,255),1)
            cv2.imshow('test',image)
            cv2.waitKey(0)
            
            
            image = np.array(sI[1,0], dtype = np.uint8)
            lP = fY[0,0]
            #print image
            print (lP)
            print (y_batch[0,0])
            #output is x1,x2,y1,y2
            #original ix x1,y1,x2,y2
            cv2.rectangle(image,(lP[0],lP[2]),(lP[1],lP[3]),(255,0,255),1)
            cv2.imshow('test',image)
            cv2.waitKey(0)
            
            image = np.array(x_batch[0,1], dtype = np.uint8)
            lP = fY[0,0]
            #print image
            print (lP)
            print (y_batch[0,0])
            #output is x1,x2,y1,y2
            #original ix x1,y1,x2,y2
            cv2.rectangle(image,(lP[0],lP[2]),(lP[1],lP[3]),(255,0,255),1)
            cv2.imshow('test',image)
            cv2.waitKey(0)'''
            
            mean_error += losses
            
            if toShowGT : 
                utils.showGates(np.asarray(tg), n_to_see=64,batch_index_to_see=0, n_neurons=n_neurons, toShow=True, toSave=False, fileName="gate-"+str(iteration)+"-"+str(bt))
               
            if True : #bt%(100//seq_length) == 0:
                summary_str = mse_summary.eval(session = sess,feed_dict = {x:x_batch, y:y_batch,c : stt.c,h : stt.h,c2 : stt.c,h2 : stt.h,pgt:prob_using_gt,phase_train:True})
                step = iteration*batch_length + bt
                file_writer.add_summary(summary_str,step)
                print(("Batch {}/{} err : {}".format(bt,batch_length,losses)))
                f = open(errFile,'a')
                f.write("Batch {}/{} err : {}\n".format(bt,batch_length,losses))
                f.close()
                    
                end = time.time()
                print("Time per iteration : ",end - start)
                start = time.time()
            
        print(("Mean : ",truediv(mean_error, batch_length)))
            
        if iteration % 1 == 0 : 
            mse = loss.eval(session = sess,feed_dict = {x:x_batch, y:y_batch,c : stt.c,h : stt.h,c2 : stt.c,h2 : stt.h,pgt:False,phase_train:False})
            
            f = open(errFile,'a')
            f.write('\n' +  str(iteration) + " Error : "+str(mse))
            f.close()
            
            
            #Get example of result 
            list_result,l_o = sess.run([preds_ori,preds],feed_dict = {x:x_batch, y:y_batch,c : stt.c,h : stt.h,c2 : stt.c,h2 : stt.h,pgt:False,phase_train:False})
            
            listCenter,listCenterPred =y_batch[0][1:],np.array(list_result[0], dtype = np.int32)
            
            print(listCenterPred,listCenter)
            
            for z in range(seq_length-1) :    
                tImage = x_batch[0][z+1]
                
                listToAnalyse = [42,48,28,34,67]
                
                #t_bb = utils.get_bb(listCenter[z][0:int(n_o/2)],listCenter[z][int(n_o/2):])
                t_bb_p = utils.get_bb(listCenterPred[z][0:int(n_o/2)],listCenterPred[z][int(n_o/2):])
                t_bb = utils.get_bb(listCenter[z][0:int(n_o/2)],listCenter[z][int(n_o/2):])
                
                cv2.rectangle(tImage,(int(t_bb_p[0]),int(t_bb_p[1])),(int(t_bb_p[2]),int(t_bb_p[3])),(255,0,255),1)
                cv2.rectangle(tImage,(int(t_bb[0]),int(t_bb[1])),(int(t_bb[2]),int(t_bb[3])),(0,0,255),1)
                
                if doAnalyze: 
                    for z22 in range(len(listToAnalyse)) :
                        cv2.circle(tImage,(int(listCenterPred[z][listToAnalyse[z22]]),int(listCenterPred[z][listToAnalyse[z22]+68])),3,(0,0,255))
                        #cv2.circle(tImage,(int(listCenter[z][z2]),int(listCenter[z][z2+68])),3,(0,0,255))
                else : 
                    for z22 in range(68) :
                        cv2.circle(tImage,(int(listCenterPred[z][z22]),int(listCenterPred[z][z22+68])),3,(0,255,0))
                        cv2.circle(tImage,(int(listCenter[z][z22]),int(listCenter[z][z22+68])),3,(255,0,0))
                        
                print(listCenter,listCenterPred)
                
                if not os.path.exists(curDir + "src/"+name_save+"-res/"):
                    os.makedirs(curDir + "src/"+name_save+"-res/")
                    
                cv2.imwrite(curDir + "src/"+name_save+"-res/Image"+str(iteration)+"_"+str(z)+".jpg", tImage)
                print((curDir + "src/"+name_save+"-res/Image"+str(iteration)+"_"+str(z)+".jpg"))
                
            
            if doSave :
                if not os.path.exists(curDir + 'src/models/'+name_save):
                    os.makedirs(curDir + 'src/models/'+name_save)
                     
                saver.save(sess,os.path.join(curDir + 'src/models/'+name_save, name_save),global_step=iteration)
            
            print(("Done for iteration "+str(iteration)))
        
    file_writer.close()
    
def track(dataType = 2,catTesting = 1,justDoErrCalc = False,is3D= False,trained_length = 32,useRefinedBB = False):
    
    doBBEvaluation = True
    useYolo = False
    useRefinedBB = useRefinedBB
    
    bb_mode = True
    
    addChannel = False
    addChannelLocaliser = False
    
    '''
    if useRefinedBB : 
        addChannelLocaliser = True
        '''
    prob_using_gt = False#0#Put 0 on actual test, put 1 put the testing purpose only regarding the ground truth 
    
    baseSize = 128 #this is for the classifier
    channels = 3
    n_o = 136
    
    
    addition = ""
    if addChannel : 
        addition = "_C_"
    if addChannelLocaliser : 
        addition += "_CL_"
        
    if useRefinedBB : 
        addition += "_RBB_"
    
    
    if useYolo :
        bb_err_ext=".RT_YOLO_err_"+str(trained_length)+addition+"txt"
        bb_ext = ".RT_YOLO_bbs_"+str(trained_length)+addition+"txt"
        flag_ext = ".RT_YOLO_FLAG_"+str(trained_length)+addition+"txt"
        ext = ".r_YOLO_"+str(trained_length)+addition+"txt"
        np_ext = "_r_YOLO_"+str(trained_length)+addition
        from lib.model import FaceDetectionRegressor
    else :    
        bb_err_ext=".RT_MTCNN_err_"+str(trained_length)+addition+"txt"
        bb_ext = ".RT_MTCNN_bbs_"+str(trained_length)+addition+"txt"
        flag_ext = ".RT_MTCNN_FLAG_"+str(trained_length)+addition+"txt"
        ext = ".r_MTCNN_"+str(trained_length)+addition+"txt"
        np_ext = "_r_MTCNN_"+str(trained_length)+addition
        if not justDoErrCalc: 
            from MTCNN import MTCNN
    
    if useFullModel : 
        baseSize = baseSize*2
        
    cropSize = baseSize + int(baseSize*.5)
    
    patchOSize = int(baseSize*.1)
    patchSize = int(patchOSize/2)
    
    arrName = ['300W-Test/01_Indoor','300W-Test/02_Outdoor','300W-Test/03_InOut']
    
    if justDoErrCalc : 
        

        arrName = ['300VW-Test/cat1','300VW-Test/cat2','300VW-Test/cat3']
        arrName3D = ['300VW-Test_M/cat1','300VW-Test_M/cat2','300VW-Test_M/cat3']
        
        for is3D in is3Dx : 
            l_all = []
            for catTesting in range(3): 
                
                if is3D : 
                    folderToTry = arrName3D[catTesting]#"Menpo-3D"#arrName[catTesting]
                else : 
                    folderToTry = arrName[catTesting]
                    
                #ext=".rt_"+u_bb+"txt"
                name = "RT_"+str(trained_length)
                
                if is3D: 
                    name+="_3D"
                    
                print(name)
                
                temp = utils.errCalc(catTesting+1,False,"images/"+folderToTry,name,is3D,ext=ext)
                
                temp = np.asarray(temp)
                for z in temp : 
                    l_all.append(z)
            
            #Make the combination 
            #flatten 
            all_err = np.asarray(l_all)
            print(all_err.shape)
            
            fileName = "src/result_compared/cat_c/"
            aboveT = utils.makeErrTxt(all_err,fileName= fileName+name+".txt",threshold = .08,lim = 1.1005)
            utils.plot_results(catTesting,resFolder= 'src/result_compared/cat_c',addition=[name],is3D=is3D,All = True)
                
        return
        
    from face_classifier_simple import face_classifier_simple
    
    name_save = "kp-transfer"
        
    if is3D : 
        name_save += "-3D"
    if useFullModel :
        name_save += "-full"
    if experimental : 
        name_save += "-exp"
    if addChannel : 
        name_save += "-channel"
        channels = 4
    
    name_save+="-"+str(trained_length)
            
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "0"
    config.gpu_options.per_process_gpu_memory_fraction = .2
    
    g_t = tf.Graph() ## This is graph for tracking
    g_c = tf.Graph() ## This is another graph for classifier
    g_l = tf.Graph() ## This is another graph for localiser
    
    sess = tf.InteractiveSession(graph = g_t,config=config)
    sess_c = tf.InteractiveSession(graph = g_c,config=config)
    sess_l = tf.InteractiveSession(graph = g_l,config=config)
    
    with g_t.as_default():
              
        print("*Initiating FT")
        ft = frameTracking(1, 2, crop_size, channels, n_neurons, n_o,learning_rate=.01,test=True,model_name=name_save,dataType = dataType,n_adder=0,CNNTrainable=True,realTime =True)
        print("*Initiating builder")
        #   x,c_state   ,c_state2   ,h_state,h_state2,  y,states,   states_to_ori,  l_gates     ,xy,final_y,sample_image,LSTMState,z2           ,initial_BB
        if useFullModel or useDoubleLSTM: 
            x,c         ,c2         ,h      ,h2,        y,preds     ,   preds_ori,      the_gates   ,xy,finalY, sampleImage ,LSTMState,phase_train  ,in_bb,     LSTMState2= ft.buildTracker()
        else : 
            x,c,c2,h,h2,y,pgt,preds_ori,the_gates,xy,finalY,sampleImage,LSTMState,phase_train,in_bb= ft.buildTracker()
        
        saver = tf.train.Saver(var_list=tf.trainable_variables(scope="(?!additional)"))
        print(tf.trainable_variables(scope="(?!additional)"))
        print(curDir + 'src/models/'+name_save)
        #print("Tensor1 : ",saver.saver_def.filename_tensor_name)
        saver.restore(sess, tf.train.latest_checkpoint(curDir + 'src/models/'+name_save))
        the_state = tf.contrib.rnn.BasicLSTMCell(n_neurons).zero_state(1, tf.float32)
    
    
    with g_c.as_default():
        print("Inititaing classifier")
        f = face_classifier_simple(patchOSize,1)
        x_c,_,pred_c = f.build()
    
    
        saver2 = tf.train.Saver()
        #print("Tensor2 : ",saver2.saver_def.filename_tensor_name)
        saver2.restore(sess_c, tf.train.latest_checkpoint(curDir + 'src/models/classifier'))
    
    channels_l = 3
    
    if addChannelLocaliser : 
        g_l2d = tf.Graph() ## This is another graph for localiser    
        sess_l2d = tf.InteractiveSession(graph = g_l2d,config=config)
        with g_l2d.as_default(): 
            f2d = face_localiser(crop_size,False,channels_l)
            x_l2d,y_l2d,pred_l2d = f2d.build()
            name_localiser = "dt-inception"
            
            if is3D : 
                name_localiser+= "-3D"
                
            saver32d = tf.train.Saver()
            #print("Tensor2 : ",saver2.saver_def.filename_tensor_name)
            saver32d.restore(sess_l2d, tf.train.latest_checkpoint(curDir + 'src/models/'+name_localiser))
        

        
    with g_l.as_default():
        print("Inititaing localiser")    
        
        if addChannelLocaliser : 
            channels_l = 4
        
        f = face_localiser(crop_size,False,channels_l)
        x_l,y_l,pred_l = f.build()
        
        name_localiser = "dt-inception"
        
        if is3D : 
            name_localiser+= "-3D"
        
        if addChannelLocaliser : 
            name_localiser+="-4D"
            
        saver3 = tf.train.Saver()
        #print("Tensor2 : ",saver2.saver_def.filename_tensor_name)
        saver3.restore(sess_l, tf.train.latest_checkpoint(curDir + 'src/models/'+name_localiser))
    
    
    
    if useYolo : 
        print("Initiating detector ")
        # FaceDetectionRegressor
        model = FaceDetectionRegressor()
        # load weights
        model.load_weights('./models/yolo')
        
    else :
        print("Using MTCNN")
        model = MTCNN()
    
    if is3D : 
        #testing/cat1
        folderToTry ="300VW-Test_M/cat"+str(catTesting)#testing/cat+str(catTesting)
    else : 
        folderToTry ="300VW-Test/cat"+str(catTesting)
    
    #folderToTry = "toTest_M"
    if is3D : 
        allVW = ['300VW-Test_M/cat1','300VW-Test_M/cat2','300VW-Test_M/cat3']
    else : 
        allVW = ['300VW-Test/cat1','300VW-Test/cat2','300VW-Test/cat3']
        
    l_hard = open(curDir + "src/hard.txt",'a') 
    
    debug = True
    
    for folderToTry in allVW : 
    
        if dataType == 2 : 
            if is3D : 
                all_batch,all_labels,_ = utils.get_kp_face(None,[folderToTry],per_folder=True,is3D = True)
            else :
                all_batch,all_labels,_ = utils.get_kp_face(None,[folderToTry],per_folder=True)
            print(("Total folder "+str(len(all_batch))))
        elif dataType == 4 : 
            folderToTry = arrName[catTesting]
            all_batch,all_labels,_ = utils.get_kp_face_localize(None,folderToTry)
            print(("Total folder "+str(len(all_batch))))
        
        totalData = 0
        
        list_err_file = []
        list_err_file_BB = []
        
        for folder_i in range(len(all_batch)): 
            
            folder_name = None 
            #now getting the folder name 
            sample_image = all_batch[folder_i][0]
            folder_name = os.path.split(os.path.split(os.path.dirname(sample_image))[0])[1]
            
            #This is for BB
            err_file_name = curDir + "images/"+folderToTry+"/"+str(folder_name)+"_TR"+bb_err_ext
            list_err_file_BB.append(err_file_name)
                
            my_file = Path(err_file_name)
            if my_file.is_file():
                print((err_file_name+" is exist "))
                if False : 
                    continue
            fileBB_err = open(err_file_name,'w')
            
            fileBB = open(curDir + "images/"+folderToTry+"/"+str(folder_name)+"_TR"+bb_ext,'w')  
            
            fileFlag = open(curDir + "images/"+folderToTry+"/"+str(folder_name)+"_TR"+flag_ext,'w')  
            
            list_err_file.append(curDir + "images/"+folderToTry+"/"+str(folder_name)+"_"+ext)
                    
            my_file = Path(curDir + "images/"+folderToTry+"/"+str(folder_name)+"_"+ext)
            if my_file.is_file():
                print((curDir + "images/"+folderToTry+"/"+str(folder_name)+"_"+ext+" is exist "))
                #continue
            file = open(curDir + "images/"+folderToTry+"/"+str(folder_name)+"_"+ext,'w') 
            
            np_name = curDir + "images/"+folderToTry+"/"+str(folder_name)+"_TR"+np_ext
            
            #Now taking the data and groundtruth for this folder
            list_images =[]
            cBBPoints2 = []
            y_batch = np.zeros([len(all_batch[folder_i]),n_o])
             
            for b_j in range(len(all_batch[folder_i])) : 
                #print len(all_batch[folder_i])
                #fetch the data for each batch 
                list_images.append(all_batch[folder_i][b_j])
                cBBPoints2.append(all_labels[folder_i][b_j])
                
            seq_length = len(list_images)
            totalData+=seq_length
            
            list_result = np.zeros([seq_length,n_o])
            counter , all_err = 0, []
            
            restartStateEvery = trained_length
            
            stt = sess.run(the_state) #restatrt the state
            stt2 = sess.run(the_state) #restatrt the state
            stt_before = sess.run(the_state)
            stt_before2 = sess.run(the_state)
            
            l_gen = []
            l_ori = []
            toRestart = False #configured in loop when to do restart
            
            doRestart = True #whter to restart or not. Parameter
            use_shifted_kp = True
            
            always_detect = False
            
            lastBB = None
            detected_BB = None
            
            l_r = None
            indexer = 0
            
            lp_BB = None
            distanceLength = 1
            bbDiagLength = .5
            
            countBeforeRestart = 0 
            
            whenToRestart = 0#restartStateEvery
            runningCount = 0
            prevDetect = False
            use_shadow = True#is to use the predicted KP as the basis of classifieer 
            firstBB = None
            
            anyHard = False
            
            choosen_kp = None #This is to select whether to use the localised from BB of predicted lstm kp or from detected bb
            
            b_channel,g_channel,r_channel = None,None,None
            images = None
            l_cd,rv = None,None
            newChannel = None
            
            the_res = []
            
            while (indexer < seq_length) :
                
                detected_BB = None
                if (toRestart and doRestart) or always_detect:
                    print("Now detecting")
                    prevDetect = True
                    
                    img = cv2.imread(list_images[indexer])
                        
                    height, width, _ = img.shape
                    ratioWidth = truediv(imWidth, width )
                    ratioHeight = truediv(imHeight,height)
                    
                    if useYolo : 
                        predictions = model.predict(img, merge=True)
                        print("Length predictions : ",len(predictions))
                    else : 
                        predictions,_ = model.doFaceDetection(img)
                        
                        
                    l_lastBB_p = []
                    l_distBB = []
                    l_diag_p = []
                    l_diag_pp = []
                    
                    if len(predictions) > 0:
                        for box in predictions : 
                            t=np.zeros(4)
                            
                            if useYolo : 
                                left = t[0]  = int((box['x'] - box['w'] / 2.))#left
                                right = t[2]  = int((box['x'] + box['w'] / 2.))#right 
                                top = t[1]= int((box['y'] - box['h'] / 2.)) #top 
                                bot = t[3] = int((box['y'] + box['h'] / 2.)) #bot
                            else :
                                left = t[0] = int(box[0])
                                right = t[2]  = int(box[2]) 
                                top = t[1]=  int(box[1]) 
                                bot = t[3] = int(box[3])
                            
                            img = cv2.resize(img,(imWidth,imHeight)).copy()
                            #cv2.rectangle(img,(int(left*ratioWidth), int(top*ratioHeight)), (int(right*ratioWidth), int(bot*ratioHeight)),(0,255,0),3)
                            '''cv2.rectangle(img,(int(left*ratioWidth), int(top*ratioHeight)), (int(right*ratioWidth), int(bot*ratioHeight)),(0,255,0),3)
                            cv2.imshow('test',img)
                            cv2.waitKey(0)'''
                            #input is  be x1,x2,y1,y2
                            lastBB_pp = np.expand_dims(np.array([left*ratioWidth,right*ratioWidth,top*ratioHeight,bot*ratioHeight]), axis=0)
                            cBB_p = np.array([(lp_BB[0,0]+lp_BB[0,1])/2,(lp_BB[0,2]+lp_BB[0,3])/2])
                            cBB_pp = np.array([(lastBB_pp[0,0]+lastBB_pp[0,1])/2,(lastBB_pp[0,2]+lastBB_pp[0,3])/2])
                            
                            diag_p = np.sqrt(np.square(lp_BB[0,0]-lp_BB[0,1]) + np.square(lp_BB[0,2]-lp_BB[0,3]))  
                            diag_pp = np.sqrt(np.square(lastBB_pp[0,0]-lastBB_pp[0,1]) + np.square(lastBB_pp[0,2]-lastBB_pp[0,3]))
                            
                            #distance between center of last bb and current detected BB
                            dist = np.sqrt(np.square(cBB_p[0]-cBB_pp[0]) + np.square(cBB_p[1]-cBB_pp[1]))
                            
                            l_distBB.append(dist)
                            l_lastBB_p.append(lastBB_pp)
                            l_diag_p.append(diag_p)
                            l_diag_pp.append(diag_pp)
                        
                        anyClose = False
                        n_close = 0
                        minDistance = 9999
                        
                        
                        if len(predictions) >= 1: 
                            print("multiple BB!",len(predictions))
                            
                            for lnx in range(0,len(l_distBB)) :
                                if ( (l_distBB[lnx] < l_diag_p[lnx] * distanceLength)):# and np.abs(l_diag_p[lnx] - l_diag_pp[lnx]) < bbDiagLength*l_diag_p[lnx]):
                                    print("got  the bb",lnx) 
                                    if (l_distBB[lnx] < minDistance): 
                                        lastBB = l_lastBB_p[lnx]
                                        minDistance =l_distBB[lnx]
                                         
                                    if not anyClose : 
                                        anyClose = True
                                    n_close+=1
                                    
                        if not anyClose : 
                            print("Does not found any close")
                            lastBB = lp_BB
                            
                    else :#if there's no face 
                        print("No face detected!")
                        lastBB = lp_BB 
                    #toRestart = False
                    
                    t_bb = lastBB[0]
                    detected_BB = [t_bb[0],t_bb[2],t_bb[1],t_bb[3]]
                
                if dataType == 4 : 
                    stt = sess.run(the_state)
                    stt2 = sess.run(the_state)
                
                flag_restart = 0;
                
                if (runningCount % restartStateEvery == 0 or toRestart) and doRestart :
                    print("Restarting the state")
                    runningCount = 0
                    #countBeforeRestart = 0
                    
                    flag_restart = 1
                    #stt = sess.run(the_state)
                    #stt2 = sess.run(the_state)
                    stt = stt_before
                    stt2 = stt_before2
                    #stt_before = None
                    #print "resetting STT"
                elif indexer == 1 : 
                    stt_before = stt
                    stt_before2 = stt2
                '''else: 
                    if stt_before is None : 
                        stt_before = stt;'''
                    
                temp_batch,cBBPoints = [],[]
                
                t_b,y_b,y_o,ori_y = [],[],[],[] 
                
                for j in range (2):   
                    if dataType == 4 :
                        images = cv2.imread(list_images[indexer][j])
                        y_t = np.asarray(cBBPoints2[indexer][j]).copy()
                    else : 
                        if indexer == 0: 
                            index = indexer 
                        else : 
                            index = indexer+j-1
                        images = cv2.imread(list_images[index])
                        y_t = np.asarray(cBBPoints2[index]).copy()
                            
                    if images is None : 
                        print((all_batch[indexer]))
                    
                    if addChannel:# and False: 
                        if indexer == 0 : 
                            l_kp_o = y_t
                        else:     #use detected landmark
                            if not useRefinedBB : 
                                l_kp_o = l_r[0][0] #choosen_kp
                            else : 
                                l_kp_o = choosen_kp
                        
                        if use_shifted_kp : 
                            l_kp = l_kp_o 
                        else :
                            t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 = utils.get_enlarged_bb(l_kp,2,2,images)
                            
                            croppedImage = images[y1:y2,x1:x2];
                            height, width, channels = croppedImage.shape
            
                            ratioHeight =truediv(image_size,height)
                            ratioWidth =truediv(image_size,width)
                            
                            r_image = cv2.resize(croppedImage, (image_size,image_size))
                            x_batch = np.expand_dims(r_image, axis=0)
                            l_kp = pred_l.eval(feed_dict = {x_l:x_batch},session = sess_l)[0]
                            
                        t0,t1,t2,t3 = utils.get_bb(l_kp[0:int(n_o//2)], l_kp[int(n_o//2):],68,False)
                        l_cd,rv = utils.get_list_heatmap(0,None,t2-t0,t3-t1,.1)
                        b_channel,g_channel,r_channel = images[:,:,0],images[:,:,1],images[:,:,2]
                        newChannel = b_channel.copy(); newChannel[:] = 0
                        height, width,_ = images.shape
                        
                        scaler = 255/np.max(rv)
                        #addOne = randint(0,2),addTwo = randint(0,2)
                        for iter in range(68) :
                            #print(height,width)
                            ix,iy = int(l_kp[iter]),int(l_kp[iter+68])
                            #Now drawing given the center
                            for iter2 in range(len(l_cd)) : 
                                value = int(rv[iter2]*scaler)
                                if newChannel[utils.inBound(iy+l_cd[iter2][0],0,height-1), utils.inBound(ix + l_cd[iter2][1],0,width-1)] < value : 
                                    newChannel[utils.inBound(iy+l_cd[iter2][0],0,height-1), utils.inBound(ix + l_cd[iter2][1],0,width-1)] = int(rv[iter2]*scaler)#int(heatmapValue/2 + rv[iter2] * heatmapValue)
                        
                        '''toShow = cv2.resize(cv2.merge((b_channel, newChannel,newChannel, newChannel)),(imWidth,imHeight)) 
                        cv2.imshow('test',toShow)
                        cv2.waitKey(0)'''
                            
                        images = cv2.merge((b_channel, g_channel,r_channel, newChannel))  
                    
                    height, width,_ = images.shape
                
                    ratioHeight_o =truediv(imHeight,height)
                    ratioWidth_o =truediv(imWidth,width)
                    
                    images = cv2.resize(images,(imWidth,imHeight)).copy()
                    t_b.append(images)
                    
                    #Correct the ground truth to work in resizpreds_oried image
                    
                    
                    if dataType == 1 : #synthetic bb. 
                        y_t[0] = y_t[0] * ratioWidth_o
                        y_t[1] = y_t[1] * ratioHeight_o
                        y_t[2] = y_t[2] * ratioWidth_o
                        y_t[3] = y_t[3] * ratioHeight_o
                    else :
                        y_t[0:68] = y_t[0:68] * ratioWidth_o
                        y_t[68:] = y_t[68:] * ratioHeight_o
                    
                    y_o.append(y_t)
                    
                    if dataType in [2,4]: #ori KP 
                        y_b.append(y_t)
                    elif dataType == 1 : #BB of landmark
                        dt = [y_t[0],y_t[2],y_t[1],y_t[3]]
                        y_b.append(dt)
                    
                temp_batch.append(t_b)
                cBBPoints.append(y_b)
                ori_y.append(y_o)
            
                x_batch = np.asarray(temp_batch)
                y_batch = np.asarray(cBBPoints).clip(0).copy()
                ori_y = np.asarray(ori_y)
                
                if dataType == 4 :
                    input_y = y_batch
                    input_y[:,1] = y_batch[:,0]
                    toFind = np.squeeze(y_batch[:,0])
                    lastBB = np.expand_dims(utils.get_bb(toFind[:68],toFind[68:],68,True),axis=0)
                else :
                    input_y = y_batch
                    if indexer == 0 : 
                        input_y[:,1] = y_batch[:,0]
                        
                        toFind = np.squeeze(y_batch[:,0])
                        lastBB = np.expand_dims(utils.get_bb(toFind[:68],toFind[68:],68,True),axis=0)
                        
                        #print(lastBB)
                    else : 
                        if lastBB is None : 
                            lastBB = np.expand_dims(utils.get_bb(np.squeeze(l_r[:,0,:68]),np.squeeze(l_r[:,0,68:]),68,True),axis = 0)
                
                #print(x_batch.shape,input_y.shape)
                
                #******* This is the tracking part 
                
                print("l_bb : ",lastBB)
                cb = lastBB[0]
                
                if np.abs(cb[0]-cb[1]) <= 50 or np.abs(cb[2]-cb[3]) <=50 or np.min(cb) <= 0: #check if the bb is 0
                    print ("less than 0")
                    if True :   
                        toFind = np.squeeze(y_batch[:,0])
                        lastBB = np.expand_dims(utils.get_bb(toFind[:68],toFind[68:],68,True),axis=0) #recover using the first BB information
                    else : 
                        #recover using the height and width of first BB information 
                        tempBB_GT = utils.get_bb(toFind[:68],toFind[68:],68,True)
                        tempBB= lastBB[0]
                        
                        l_x = np.abs(tempBB_GT[1] - tempBB_GT[0])/2
                        l_y = np.abs(tempBB_GT[3] - tempBB_GT[2])/2
                        
                        c_x = (tempBB[1] + tempBB[0])/2 
                        c_y = (tempBB[2] + tempBB[3])/2
                        
                        height, width,_ = cv2.imread(list_images[indexer]).shape
                        proposeBB = [max(c_x - l_x,0),min(c_x+l_x,width),max(c_y - l_y,0),min(c_y+l_y,height)  ]
                        print("Propose BB ", proposeBB)
                        
                        lastBB = np.expand_dims(proposeBB,axis = 0)
                    
                    
                        print("Detected bb : ",detected_BB)
                        print("Propose BB : ", proposeBB)
                    
                    
                    t_bb = lastBB[0]    
                    detected_BB = [t_bb[0],t_bb[2],t_bb[1],t_bb[3]]
                    
                if not (useFullModel or useDoubleLSTM) :
                    l_r,stt= sess.run([preds_ori,LSTMState],feed_dict = {x:x_batch, y:input_y,c : stt.c,h : stt.h,c2 : stt.c,h2 : stt.h,phase_train:False,in_bb: lastBB})
                else : 
                    print("using double lstm")
                    l_r,st,fy,stt,stt2= sess.run([preds_ori,preds,finalY,LSTMState,LSTMState2],feed_dict = {x:x_batch, y:input_y, c : stt.c, h : stt.h, c2 : stt2.c, h2 : stt2.h, phase_train:False,in_bb: lastBB})
                
                #Now dismissing the addded channel 
                if addChannel : 
                    images = images[:,:,:3]
                
                print(utils.get_enlarged_bb(l_r[0][0],2,2,images))
                if(detected_BB is not None) : 
                    print(utils.get_enlarged_bb(detected_BB,2,2,images,is_bb=True))
                
                if bb_mode : 
                    if prevDetect : 
                        numCheck = 2;
                    else :
                        numCheck = 1;
                        
                    if indexer == 0 : 
                        l_bbs = [utils.get_enlarged_bb(y_t,2,2,images),utils.get_enlarged_bb(y_t,2,2,images)]
                        the_kp = [y_t, y_t]
                    else : 
                        if detected_BB is None : 
                            detected_BB = utils.get_bb(l_r[0][0][:68],l_r[0][0][68:])
                            
                        #get the list of BB to be evaluated 
                        print(detected_BB,np.asarray(detected_BB).shape)
                        
                        '''l_bbs = [utils.get_enlarged_bb(l_r[0][0#],2,2,images),utils.get_enlarged_bb(detected_BB,2,2,images,is_bb=True)]
                        the_kp = [l_r[0][0],l_r[0][0]]#problem on the second one, that to use the localized one'''
                        
                        l_bbs = [utils.get_enlarged_bb(l_r[0][0],2,2,images),utils.get_enlarged_bb(detected_BB,2,2,images,is_bb=True)]
                        the_kp = [l_r[0][0],choosen_kp]#problem o#n the second one, that to use the localized one 
                    
                    l_predict = []
                    
                    
                    #****** This is the localisation part 
                    
                    for i_check in range(numCheck): #check to use between the detected BB or from KP
                        
                        #print(l_bbs,l_bbs[i_check], len(l_bbs), len(l_bbs[i_check]))
                        t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 = l_bbs[i_check][0],l_bbs[i_check][1],l_bbs[i_check][2],l_bbs[i_check][3],l_bbs[i_check][4],l_bbs[i_check][5],l_bbs[i_check][6],l_bbs[i_check][7],l_bbs[i_check][8]
                        # detectedBB
                        
                        #get the heatmap 
                        if is3D : 
                            add ="3D"
                        else : 
                            add ="2D"
                            
                        print(i_check,t)
                        
                        '''if (y2 < y1): 
                            y2 = y1 + 10;
                        if (x2 < x1) : 
                            x2 = x1 + 10'''
                        
                        croppedImage = images[y1:y2,x1:x2];
                        height, width, channels = croppedImage.shape
                    
                        ratioHeightR =truediv(height,crop_size)
                        ratioWidthR =truediv(width,crop_size)
                    
                        #print ratioHeight,import configratioWidth
                        r_image = cv2.resize(croppedImage, (crop_size,crop_size))[:,:] #dismiss the channel from the previous one

                        #get the recently calculated heatmap. If any use it, otherwise calculate it
                        if addChannelLocaliser :
                            b_channel,g_channel,r_channel = r_image[:,:,0],r_image[:,:,1],r_image[:,:,2]
                            
                            to_use_kp = pred_l2d.eval(feed_dict = {x_l2d:np.expand_dims(r_image, axis=0)},session = sess_l2d)[0]
                            
                            newChannel = utils.make_heatmap(list_images[indexer],r_image,add,to_use_kp,False,.1,.05)
                            r_image = cv2.merge((b_channel, g_channel,r_channel, newChannel))
                            #heatmaps = cv2.merge((b_channel, newChannel,newChannel, newChannel))
                            
                            
                            '''cv2.imshow('test',heatmaps)
                            cv2.waitKey(0)'''
                        
                        
                        
                        predicted = pred_l.eval(feed_dict = {x_l:np.expand_dims(r_image, axis=0)},session = sess_l)[0]
                        
                        '''for z22 in range(68) :
                                cv2.circle(r_image,(int(predicted[z22]),int(predicted[z22+68])),2,(0,255,0))
                                cv2.circle(r_image,(int(listCenter[z22]),int(listCenter[z22+68])),2,(0,0,255))
                                
                        cv2.imshow('test',r_image)
                        cv2.waitKey(0)'''
                        
                        predicted[:68] = predicted[:68]*ratioWidthR + x_min
                        predicted[68:] = predicted[68:]*ratioHeightR +y_min
                        l_predict.append(predicted)
                    
                
                #print("st : ",st,"fy",fy)
                
                if bb_mode : 
                    listCenterPred  = l_predict
                else : 
                    listCenterPred  = [np.array(l_r[0][0])]
                
                if use_shadow : 
                    listCenterPred.append(np.array(l_r[0][0])) #use shadow mean to use the predicted (direct) kp as also measurement of whether it is face or not
                    
                
                #after the detection is made. Usually happen on the hard problem.
                min_classifier = 1;
                
                l_f_classifier = []
                
                #print(listCenterPred)#
                
                #*****this is the classification part
                
                if indexer !=0 :#and indexer%restartStateEvery == 0 :
                    for i_chosen in range(0,len(listCenterPred)) :  
                        
                        images = x_batch[0][0].copy()
                        
                        b_channel,g_channel,r_channel = images[:,:,0],images[:,:,1],images[:,:,2]
                        tImage = cv2.merge((b_channel, g_channel,r_channel))
                        
                        exp_face = np.zeros([1,68,patchOSize,patchOSize,3])
                        
                        '''#Now check whether current tracked face is correcly face. 
                        y_t = utils.get_bb(y_t[0:68],y_t[68:])
                        #[xMin,yMin,xMax,yMax]
                        print(y_t)
                        #change it to be x1,x2,y1,y2
                        dt = [y_t[0],y_t[2],y_t[1],y_t[3]]
                        exp_face = np.expand_dims(cv2.resize(tImage[dt[2]:dt[3],dt[0]:dt[1]],(crop_size,crop_size)), axis=0)'''
                        
                        toEvalClassifier = listCenterPred[i_chosen]#l_r[0][0]
                        
                        t = utils.get_bb(toEvalClassifier[:68], toEvalClassifier[68:])
                        
                        l_x = (t[2]-t[0])/2 + (t[2]-t[0])/4
                        l_y = (t[3]-t[1])/2 + (t[3]-t[1])/4 
                        
                        x1 = int(max(t[0] - l_x,0))
                        y1 = int(max(t[1] - l_y,0))
                        
                        #print tImage.shape
                        x2 = int(min(t[2] + l_x,tImage.shape[1]))
                        y2 = int(min(t[3] + l_y,tImage.shape[0]))
                        
                        if (np.abs(y1-y2) <= 1): 
                            y2 = y1+10;
                        if (np.abs(x1-x2) <= 1): 
                            y2 = y1+10;
                        
                        
                        tImage = tImage[y1:y2,x1:x2].copy();
                        
                        height, width,_ = tImage.shape
                        
                        '''if height <= 0 or width <= 0 : 
                            print("Zero dimension, restart with the last one"); 
                            print(listCenterPred)
                            t2 = firstBB[0]
                            
                            l_x = (t2[2]-t2[0])/2 + (t2[2]-t2[0])/4
                            l_y = (t2[3]-t2[1])/2 + (t2[3]-t2[1])/4
                            
                            x1 = int(max(t[0] - l_x,0))
                            y1 = int(max(t[1] - l_y,0))
                            
                            #print tImage.shape
                            x2 = int(min(t[2] + l_x,tImage.shape[1]))
                            y2 = int(min(t[3] + l_y,tImage.shape[0]))
                            
                            tImage = tImage[y1:y2,x1:x2].copy();
                            height, width,_ = tImage.shape '''
                                
                            
                        ratioHeight =truediv(cropSize,height)
                        ratioWidth =truediv(cropSize,width)
                                    
                        tImage = cv2.resize(tImage,(cropSize,cropSize)).copy()
                        
                        #Now fixing the groundtruth 
                        kpX = (toEvalClassifier[:68] - x1)*ratioWidth
                        kpY = (toEvalClassifier[68:] - y1)*ratioHeight
                        
                        
                        for k_2 in range(68) : 
                            x_2,y_2 = int(kpX[k_2]),int(kpY[k_2])
                            t_image = np.zeros([patchOSize,patchOSize,3])
                            t_image[0:(utils.inBound(int(y_2+patchSize),0,tImage.shape[0]) - utils.inBound(int(y_2-patchSize),0,tImage.shape[0])),
                                    0:(utils.inBound(int(x_2+patchSize),0,tImage.shape[1]) - utils.inBound(int(x_2-patchSize),0,tImage.shape[1]))
                                    ] = tImage[utils.inBound(int(y_2-patchSize),0,tImage.shape[0]):utils.inBound(int(y_2+patchSize),0,tImage.shape[0]),
                                               utils.inBound(int(x_2-patchSize),0,tImage.shape[1]):utils.inBound(int(x_2+patchSize),0,tImage.shape[1])]
                            exp_face[0,k_2] = t_image
                            
                            '''cv2.rectangle(tImage,(utils.inBound(int(x_2-patchSize),0,tImage.shape[1]),utils.inBound(int(y_2-patchSize),0,tImage.shape[0])),
                                               (utils.inBound(int(x_2+patchSize),0,tImage.shape[1]),utils.inBound(int(y_2+patchSize),0,tImage.shape[0])),(0,255,0),1)'''
                            
                            #cv2.imwrite('test_'+str(k_2)+".jpg",t_image)
                        '''cv2.imshow("test",tImage)
                        cv2.waitKey(0)'''
                        
                        is_face = sess_c.run(pred_c,feed_dict = {x_c:exp_face})
                        
                        f_index = sigmoid(np.squeeze(is_face))
                        if f_index < min_classifier: 
                            min_classifier = f_index
                        
                        l_f_classifier.append(f_index)
                    
                    print("is face : \t \t",min_classifier, "CBR : ",countBeforeRestart," whentorestart ",whenToRestart)
                    if min_classifier < 0.5 :
                        print("Evaluating whether to restart or not **")
                        if countBeforeRestart > whenToRestart : 
                            toRestart = True
                            countBeforeRestart = 0 
                            continue
                        else :  
                            print("skipping restart**")
                            countBeforeRestart+=1
                            toRestart = False
                    else : 
                        print("Face is ok, keep tracking **")
                        prevDetect = False
                        toRestart = False
                
                if prevDetect: 
                    if l_f_classifier[0] > l_f_classifier[1]: 
                        choosen_kp =  listCenterPred[0]
                    else: 
                        choosen_kp =  listCenterPred[1]
                else : 
                    choosen_kp = l_predict[0]#l_r[0][0]
                
                lp_BB = np.expand_dims(utils.get_bb(np.squeeze(choosen_kp[:68]),np.squeeze(choosen_kp[68:]),68,True),axis = 0)
                
                if useRefinedBB : 
                    lastBB = lp_BB
                else : 
                    lastBB = np.expand_dims(utils.get_bb(np.squeeze(l_r[0][0][:68]),np.squeeze(l_r[0][0][68:]),68,True),axis = 0)#lp_BB
                
                if firstBB is None : 
                    firstBB = lp_BB
                
                print(list_images[index])
                #print(stt)
                listCenter=y_batch[0][1]
                tImage = x_batch[0][1].copy()#batch 0, seq 0 (0->1). 
                
                if prevDetect : 
                    color = (255,0,0)
                    colorBB = [255,0,0]
                    flag_data = 1.0
                else : 
                    color = (0,255,0)
                    colorBB = [0,255,0]
                    flag_data = 0.0
                
                useOriginalSize = True
                
                if dataType in [2,4] : 
                    
                    
                    listCenterR = listCenter.copy()
                    choosen_kpR = choosen_kp.copy()
                    l_rR = l_r[0][0].copy()
                    
                    if useOriginalSize : 
                        listCenterR[0:68] = listCenterR[0:68] * 1/ratioWidth_o
                        listCenterR[68:] = listCenterR[68:] * 1/ratioHeight_o
                        choosen_kpR[0:68] = choosen_kpR[0:68] * 1/ratioWidth_o
                        choosen_kpR[68:] = choosen_kpR[68:] * 1/ratioHeight_o
                        l_rR[0:68] = l_rR[0:68] * 1/ratioWidth_o
                        l_rR[68:] = l_rR[68:] * 1/ratioHeight_o
                        tImage = cv2.imread(list_images[indexer])
                    
                        
                    for z22 in range(68) :
                        cv2.circle(tImage,(int(listCenterR[z22]),int(listCenterR[z22+68])),2,(0,0,255))
                        cv2.circle(tImage,(int(choosen_kpR[z22]),int(choosen_kpR[z22+68])),2,color)
                        cv2.circle(tImage,(int(l_rR[z22]),int(l_rR[z22+68])),2,(0,255,255))
                        
                    if doBBEvaluation : 
                        bbox_gt = utils.get_bb(listCenterR[:68],listCenterR[68:])
                        bbox = utils.get_bb(l_rR[:68],l_rR[68:])
                            
                        cv2.rectangle(tImage,
                                (int(bbox_gt[0]), int(bbox_gt[1])),
                                (int(bbox_gt[2]), int(bbox_gt[3])),
                                [0,0,255], 2)
                        
                        cv2.rectangle(tImage,
                                (int(bbox[0]), int(bbox[1])),
                                (int(bbox[2]), int(bbox[3])),
                                colorBB, 2)
                        
                        
                        iou = utils.calc_bb_IOU(bbox, bbox_gt)
                        err = ( np.linalg.norm(np.asarray([bbox[0],bbox[1]])-np.asarray([bbox_gt[0],bbox_gt[1]])) + np.linalg.norm(np.asarray([bbox[2],bbox[3]])-np.asarray([bbox_gt[2],bbox_gt[3]])) )/2
                        print ("Err : ",err)
                        print("IOU : ",iou)
                        
                        fileBB_err.write("%.4f\t%4f\n" % (iou,err));
                        fileBB.write("%.4f\t%4f\t%4f\t%4f\n" % (bbox[0],bbox[1],bbox[2],bbox[3]));
                        fileFlag.write("%.4f\t%.4f\n" % (flag_data,flag_restart));
                    
                    if dataType == 2 : 
                        curr_err = utils.calcLandmarkError(choosen_kpR,listCenterR)
                    else : 
                        curr_err = utils.calcNormalizedDistance(choosen_kpR,listCenterR)
                    
                    print(("Curr err : \t\t\t\t"+str(curr_err)+"_"+str(indexer)+"/"+str(seq_length)))
                    file.write("%.4f\n" % (curr_err));
                    
                    the_res.append(choosen_kpR)
                    
                    runningCount+=1
                    
                    
                
                cv2.imshow('Benchmark '+bb_err_ext,tImage)
                
                if curr_err > 0.08 : 
                    cv2.waitKey(1)
                else : 
                    cv2.waitKey(1)
                    
                if curr_err > 0.3 and not anyHard: 
                    l_hard.write(list_images[indexer]+" " +str(curr_err)+" \n")
                    anyHard = True
                
                
                #print listCenterPred
                if True and indexer%4 == 0 : #  and i>750 and i<1250 :
                    l_gen.append(tImage)#[50:230,50:370])
                    l_ori.append(x_batch[0][1])#[50:230,50:370]) 
                    
                    #cv2.imwrite('./picts/res_%.3i'%(i)+'.jpg',tImage)#[50:230,50:370])
                    #cv2.imwrite('./picts/ori_%.3i'%(i)+'.png',x_batch[0][1])#[50:230,50:370])
                #print listCenterPred    
                indexer +=1
                
                
            if addChannel : 
                del b_channel; del g_channel; del r_channel
                del images
            
                del l_cd; del rv;
                del newChannel;
            
                gc.collect()
                
            #makeGIF(l_gen,'./gen.gif')
            #makeGIF(l_ori,'./ori.gif')
            the_res = np.asarray(the_res)
            np.save(np_name,the_res)
            
            file.close()
            if doBBEvaluation : 
                fileBB_err.close()
                fileBB.close()
                fileFlag.close()
            
    l_hard.close()
    #exit(0)
    
    arrName = ['300VW-Test/cat1','300VW-Test/cat2','300VW-Test/cat3']
    arrName3D = ['300VW-Test_M/cat1','300VW-Test_M/cat2','300VW-Test_M/cat3']
    

mode = FLAGS.i_mode
is3D = FLAGS.i_is3D
i = 0
if False : #mode == 0 :         
    training(2) #Training on 3D landmark
else : 
    list_trained = [2]
    is3Dx = [False,True]
    for is3D in is3Dx : 
        for x in list_trained : 
            print(i)
            track(2,3,True,is3D = is3D,trained_length=x,useRefinedBB=True) 
            i=i+1

#training(4,True) #localisation 

#training(2,True)
#training(1,True)


#0 is circle, 1 is bb, 2 is the kp (that we want). 3 is for bounding box.
#2,1 2,2 2,3.. 4.0, 4.1, 4.2
#track(4,0,False)
#track(1)
#track(2,1,False,is3D = True)

#track(2,1,False)

#Localize, 4 - 6. 4 : indoor
#datatype : 1 and 3 is bounding box. 2 is kp_face, 4 is localize  
#localize(4,0,False)
