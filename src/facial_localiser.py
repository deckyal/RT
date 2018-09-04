'''
Created on Dec 20, 2017

@author: deckyal
'''
from operator import truediv
import utils
import tensorflow as tf
from datetime import datetime
import numpy as np
import cv2
import os
import inception_resnet_v1
import tensorflow.contrib.slim as slim
import random
from random import randint
from  face_localiser import face_localiser
from scipy.stats import multivariate_normal
from pathlib import Path
from config import *
import glob
from MTCNN import MTCNN
    

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_boolean("isFullDimension", True,
                        "Whether to train inception submodel variables.")
tf.flags.DEFINE_boolean("is3D", False,
                        "Whether to train inception submodel variables.")

def train(continuing=False):
    
    gpu_fract = .25
    config = tf.ConfigProto()
    
    width = 720
    height = 480
    
    fullDimension = False 
    print(fullDimension)
    fourChannel = FLAGS.isFullDimension
    is3D = True#FLAGS.is3D
    
    print("Training continuing {}".format(continuing))
    #Now training 
    n_iterations = 1000
    batch_size = 64;
    
    channels = 3
    image_size = 128
    doTransformation = True
    
    global_step =tf.Variable(0,trainable=False)
    root_log_dir = "tf_logs"
    logdir = "{}/run-{}/".format(root_log_dir,datetime.utcnow().strftime("%Y%m%d%H%M%S"))

    #First get the list of input and bounding boxes. 
    
    name_save = "dt-inception_84"
    if is3D : 
        name_save += "-3D"
        
    n_o = 136
    n_adder = 0
    err_name = name_save+"-err"
    
    
    if fullDimension  : 
        image_size = 256
        name_save += "-full"
        err_name += "-full"
        batch_size //= 2
        gpu_fract = 1
    
    
    use_estimated_kp = True
    
    config.gpu_options.per_process_gpu_memory_fraction = gpu_fract
    config.gpu_options.visible_device_list = "0"
    
    print("Fetching all data")
    
    #"300W_LP"
    #all_batch, all_labels, _ = utils.get_kp_face(None, "300VW-Train",False, 10,True)
    if is3D: 
        folderToTry = ["Menpo_Challenge/3D","300W-Train"]#["Menpo_Challenge/3D","300W_LP","300W-Train"]#"Menpo-3D"#arrName[catTesting]
    else : 
        folderToTry = ["Menpo_Challenge/2D","300W-Train"]#["Menpo_Challenge/2D","300W_LP","300W-Train"]
        
        
    all_batch, all_labels, _ = utils.get_kp_face(None, folderToTry,False, 1,is3D)
        
    batch_length = len(all_batch) // batch_size
    
    
    if fourChannel :
        
        if True :
            
            g_2d = tf.Graph() ## This is one graph
            sess_2d = tf.InteractiveSession(graph = g_2d,config = config)
            
            with g_2d.as_default():       
                f_2D = face_localiser(image_size,False,3)
                x_2d,y_2d,pred_2d = f_2D.build()
                saver_2d = tf.train.Saver(max_to_keep=2,save_relative_paths=True)
                sess_2d = tf.InteractiveSession(config = config)
                saver_2d.restore(sess_2d, tf.train.latest_checkpoint('./models/'+name_save))
         
        err_name += "-4D"
        name_save+="-4D"
        channels = 4
    
    
    errFile = curDir + "src/"+err_name+".txt"
    f = open(errFile,'w')
    f.write(' \n Error : ')
    f.close()
    
    
    g_d = tf.Graph() ## This is graph for tracking
    
    with g_d.as_default() : 
        f = face_localiser(image_size,True,channels)
        x,y,pred = f.build()
    

        sess = tf.InteractiveSession(graph = g_d, config = config)
        
        
        if not continuing : 
            reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionResnetV1/(?![Bottleneck,Conv2d_0a_3x3])") # regular expression
            print(reuse_vars)
            reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
            saver = tf.train.Saver(reuse_vars_dict)
            saver.restore(sess,curDir + 'src/models/facenet/model-20170511-185253.ckpt-80000')
                    
        loss = tf.sqrt(tf.reduce_mean(tf.square(pred - y)))#tf.reduce_mean(tf.sqrt(tf.square(pred - y)))
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            training_op = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss)#tf.train.MomentumOptimizer(0.02, momentum=0.01)
        
        print("Now summary saver")
        
        mse_summary = tf.summary.scalar('Loss',loss)
        file_writer = tf.summary.FileWriter(logdir,tf.get_default_graph())
        
        print("Now training")
        saver = tf.train.Saver(var_list = (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)),max_to_keep=2,save_relative_paths=True)
        #sess.run(tf.global_variables_initializer())
        utils.initialize_uninitialized_global_variables(sess)
    
    if continuing :
        print(curDir + 'src/models/'+name_save) 
        saver.restore(sess, tf.train.latest_checkpoint(curDir + 'src/models/'+name_save))
        
    for iteration in range(n_iterations) :
        print("it-",iteration)
        counter = 0
        
        x_batch = np.zeros([batch_size,image_size,image_size,channels])
        y_batch = np.zeros([batch_size,n_o+n_adder])
        
        
        for bt in range(0,batch_length) :
            
            print("Batch {}/{}".format(bt,batch_length))
            temp_batch = [] 
            cBBPoints = []
            l_imgtr = []

            
            #print(counter,batch_length,len(all_batch))
            for b_i in range(counter,counter+batch_size):
                #fetch the data for each batch 
                temp_batch.append(all_batch[b_i])
                cBBPoints.append(all_labels[b_i])
                
            cBBPoints = np.array(cBBPoints)
            y_batch = cBBPoints.clip(0)
            counter+=batch_size

            lower_xy = np.zeros([batch_size,2])
            
            for j in range(batch_size):
    
                sel = randint(0,3)#iteration%2
                rad = randint(-3,3)
                
                x_min = 0; y_min = 0;
                
                tImage = cv2.imread(temp_batch[j])
                #print(temp_batch[j])
                
                try : 
                    if doTransformation : 
                        if sel == 0 : 
                            out = [tImage,y_batch[j]]
                        elif sel == 1 : #Do flipping  
                            out = utils.transformation(tImage,y_batch[j], 1, 1)
                        elif sel == 2 : #Do rotation
                            out = utils.transformation(tImage,y_batch[j], 2, 5*rad)
                        elif sel == 3 : #Do occlusion 
                            out = utils.transformation(tImage,y_batch[j], 3, 1)
                        tImage = out[0];y_batch[j] = out[1]
                except : 
                    print(temp_batch[j])
                
                
                allowed_diagonal = np.sqrt( np.square(250) + np.square(250)  )
                
                if fourChannel : 
                    
                    #now evaluate the image b shape, and recursively reshape until the bb becomes small enought 
                    
                    '''for i in range(68) :
                        cv2.circle(tImage,(int(y_batch[j][i]),int(y_batch[j][i+68])),3,(0,0,255))
                    cv2.imshow("Test",tImage)
                    cv2.waitKey(0)'''
                    
                    init_bb_size = 0
                    
                    t_bb = utils.get_bb(y_batch[j][:68], y_batch[j][68:])
                    init_bb_size = np.sqrt( np.square(t_bb[2]-t_bb[0]) + np.square(t_bb[3]-t_bb[1])  )
                    
                    #print("init_bb size 1 : ",init_bb_size," ad : ",allowed_diagonal," ",t_bb)
                    
                    while(init_bb_size > allowed_diagonal) : 
                        
                        #print("Resizing")
                        
                        height, width, channels = tImage.shape
                            
                        ratioHeight =truediv(int(height/2),height)
                        ratioWidth =truediv(int(width/2),width)
                        
                        
                        tImage = cv2.resize(tImage, (int(width/2),int(height/2)))
        
                        y_batch[j,0:n_o//2] *= ratioWidth
                        y_batch[j,n_o//2:] *= ratioHeight 
                        
                        t_bb = utils.get_bb(y_batch[j][:68], y_batch[j][68:])
                        init_bb_size = np.sqrt( np.square(t_bb[2]-t_bb[0]) + np.square(t_bb[3]-t_bb[1])  )
                        
                        '''for i in range(68) :
                            cv2.circle(tImage,(int(y_batch[j][i]),int(y_batch[j][i+68])),3,(0,0,255))
                        cv2.imshow("Test",tImage)
                        cv2.waitKey(0)'''
                    
                    '''for i in range(68) :
                        cv2.circle(tImage,(int(y_batch[j][i]),int(y_batch[j][i+68])),3,(0,0,255))
                    cv2.imshow("Test",tImage)
                    cv2.waitKey(0)'''
                            
                        
                    #print("init_bb size 2 : ",init_bb_size," ad : ",allowed_diagonal," ",t_bb)
                    
                    if is3D : 
                        add ="3D"
                    else : 
                        add ="2D"
                        
                    #get the recently calculated heatmap. If any use it, otherwise calculate it
                    tBase = os.path.basename(temp_batch[j])
                    tName,tExt = os.path.splitext(tBase)
                    theDir =  os.path.dirname(temp_batch[j])+"/../heatmap-"+add+"/"
                    theDir_KP =  os.path.dirname(temp_batch[j])+"/../kp-"+add+"/"
                    
                    if not os.path.exists(theDir):
                        os.makedirs(theDir)
                        
                    if not os.path.exists(theDir_KP):
                        os.makedirs(theDir_KP)
                        
                    fName =theDir+tName+".npy"
                    #print(fName)
                    
                    b_channel,g_channel,r_channel = tImage[:,:,0],tImage[:,:,1],tImage[:,:,2]
                    
                    passing = False
                    if os.path.isfile(fName) : 
                        newChannel = np.load(fName)
                        #print("using saved npy")
                        if b_channel.shape[0] == newChannel.shape[0] : 
                            passing = True
                            
                    if not passing :    
                        
                        print("making NPY ",temp_batch[j])
                        newChannel = b_channel.copy(); newChannel[:] = 0
                        
                        #obtain the approximated landmark 
                        if use_estimated_kp : 
                            #use the saved estimaged KP file if any 
                            
                            fName_kp =theDir_KP+tName+".npy"
                            
                            if not os.path.isfile(fName_kp): 
                            
                                t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 = utils.get_enlarged_bb(y_batch[j],2,2,tImage)
                                croppedImage = tImage[y1:y2,x1:x2].copy();
                                height, width, channels = croppedImage.shape
                                ratioHeightR =truediv(height,image_size)
                                ratioWidthR =truediv(width,image_size)
                                
                                r_image = cv2.resize(croppedImage, (image_size,image_size))
                                x_batch_2 = np.expand_dims(r_image, axis=0)
                                l_kp = pred_2d.eval(feed_dict = {x_2d:x_batch_2},session = sess_2d)[0]
                                l_kp[:68] = l_kp[:68]*ratioWidthR + x_min
                                l_kp[68:] = l_kp[68:]*ratioHeightR + y_min
                                
                                np.save(fName_kp,l_kp)
                                
                                y_t = l_kp
                            else : 
                                y_t = np.load(fName_kp)
                            #t0,t1,t2,t3 = utils.get_bb(y_t[0:int(n_o//2)], y_t[int(n_o//2):],68,False)
                            t0,t1,t2,t3 = utils.get_bb(y_t[0:int(n_o//2)], y_t[int(n_o//2):],68,False,random.uniform( -.25, .25 ),random.uniform( -.25, .25 ),random.uniform( -.25, .25 ),random.uniform( -.25, .25 ),random.uniform( -.25, .25 ))
                        else : 
                            y_t = y_batch[j]
                            t0,t1,t2,t3 = utils.get_bb(y_t[0:int(n_o//2)], y_t[int(n_o//2):],68,False,random.uniform( -.25, .25 ),random.uniform( -.25, .25 ),random.uniform( -.25, .25 ),random.uniform( -.25, .25 ),random.uniform( -.25, .25 ))
                        #print(t0,t1,t2,t3)
                        
                        
                        l_cd,rv = utils.get_list_heatmap(0,None,t2-t0,t3-t1,.15)
                        height, width,_ = tImage.shape
                        
                        wx = t2-t0
                        wy = t3-t1
                        
                        if not use_estimated_kp: 
                            distance = .035
                        else : 
                            distance = 0
                        
                        scaler = 255/np.max(rv)
                        #addOne = randint(0,2),addTwo = randint(0,2)
                        for iter in range(68) :
                            #print(height,width)
                            
                            #ix,iy = int(y_t[iter])+randint(0,2),int(y_t[iter+68])+randint(0,2)
                            ix,iy = int(y_t[iter])+int(random.uniform(-distance,distance)*wx),int(y_t[iter+68])+int(random.uniform(-distance,distance)*wy)
                            
                            #Now drawing given the center
                            for iter2 in range(len(l_cd)) : 
                                value = int(rv[iter2]*scaler)
                                if newChannel[utils.inBound(iy+l_cd[iter2][0],0,height-1), utils.inBound(ix + l_cd[iter2][1],0,width-1)] < value : 
                                    newChannel[utils.inBound(iy+l_cd[iter2][0],0,height-1), utils.inBound(ix + l_cd[iter2][1],0,width-1)] = int(rv[iter2]*scaler)#int(heatmapValue/2 + rv[iter2] * heatmapValue)
                        
                        np.save(fName,newChannel)
                        
                        if False : #( "aflw__face_39914.jpg" in temp_batch[j] ) :  
                            
                            print("saving npy")
                            print(tImage.shape)
                            print(newChannel.shape)
                        
                    '''tImage2 = cv2.merge((b_channel, g_channel,newChannel, newChannel))
                    cv2.imshow("combined",tImage2)
                    cv2.waitKey(0)'''
                    
                    try : 
                        tImage = cv2.merge((b_channel, g_channel,r_channel, newChannel))
                    except: 
                        print(temp_batch[j])
                        print(b_channel.shape,newChannel.shape)
                        print(tImage.shape)
                        print("init_bb size 3 : ",init_bb_size," ad : ",allowed_diagonal," ",t_bb)
                    
                    '''#Try drawing gausian image
                    mean = [0,0]; cov = [[10,0],[0,10]]
                    
                    tx = np.random.multivariate_normal(mean,cov,200).astype(int)
                    tx = np.unique(tx,axis=0)
                    rv = multivariate_normal.pdf(tx,mean = mean, cov = [.5,.5])
                    height, width,_ = tImage.shape
                    
                    heatmapValue = 256
                    for iter in range(68) :
                        #print(height,width)
                        ix,iy = int(y_batch[j,iter]),int(y_batch[j,iter+68])
                        #Now drawing given the center
                        for iter2 in range(len(tx)) : 
                            newChannel[utils.inBound(iy+tx[iter2][0],0,height-1), utils.inBound(ix + tx[iter2][1],0,width-1)] = int(heatmapValue/2 + rv[iter2] * heatmapValue)
                    '''
                    '''tImage = cv2.merge((b_channel,newChannel,newChannel, newChannel))
                    cv2.imshow('test',tImage)
                    cv2.waitKey(0)'''
                    
                    '''for i in range(68) :
                        cv2.circle(tImage,(int(x_list[i]),int(y_list[i])),3,(0,0,255))
                    cv2.imshow('test',tImage)
                    cv2.waitKey(0)'''
                    
                #print("Image Shape : ", tImage.shape)
                t = utils.get_bb(y_batch[j,0:n_o//2], y_batch[j,n_o//2:],68,False,random.uniform( -.25, .25 ),random.uniform( -.25, .25 ),random.uniform( -.25, .25 ),random.uniform( -.25, .25 ),random.uniform( .05, .25 ))#print(t)
                '''cv2.imshow('test2',tImage)
                cv2.waitKey(0)'''
                
                l_x = (t[2]-t[0])/2
                l_y = (t[3]-t[1])/2
                
                x1 = int(max(t[0] - l_x,0))
                y1 = int(max(t[1] - l_y,0))
                
                x_min = x1; y_min = y1;
                
                #print tImage.shape
                x2 = int(min(t[2] + l_x,tImage.shape[1]))
                y2 = int(min(t[3] + l_y,tImage.shape[0]))
                
                croppedImage = tImage[y1:y2,x1:x2];
                #print temp_batch[j],x1,y1,x2,y2
                
                
                height, width, channels = croppedImage.shape
                
                '''print("HELLO",sel,out)
                print(tImage.shape)
                print(l_x,l_y)
                print(croppedImage.shape,t,x1,y1,x2,y2,y_batch[j])'''
                    
                ratioHeight =truediv(image_size,height)
                ratioWidth =truediv(image_size,width)
                
                #print ratioHeight,import configratioWidth
                
                r_image = cv2.resize(croppedImage, (image_size,image_size))
                
                l_imgtr.append(r_image)
                x_batch[j] = np.asarray(r_image,np.float32)
                
                #print x_batch[j]
                
                lower_xy[j] = np.array([x_min,y_min])
                

                y_batch[j,0:n_o//2] -= x_min
                y_batch[j,n_o//2:] -= y_min
                
                y_batch[j,0:n_o//2] *= ratioWidth
                y_batch[j,n_o//2:] *= ratioHeight
                
                
                if False : 
                    x_list = y_batch[j,0:68]
                    y_list = y_batch[j,68:136]
                    #getting the bounding box of x and y
                    print(x_list, y_list)
                    
                    bb = utils.get_bb(x_list,y_list)
                    
                    cv2.rectangle(r_image,(bb[0],bb[1]),(bb[2],bb[3]),(255,0,255),1)
                    for i in range(68) :
                        cv2.circle(r_image,(int(x_list[i]),int(y_list[i])),3,(0,0,255))
                    
                    cv2.imshow('jim',r_image)
                    cv2.waitKey(0)
                
            #Now calculating the loss 
            t_loss,_,_= sess.run([loss,training_op,update_ops],feed_dict = {x:x_batch, y:y_batch})
            print(t_loss)
            #print "gv:",gv2
            
            if bt%10 == 0: 
                summary_str = mse_summary.eval(feed_dict = {x:x_batch, y:y_batch})
                step = iteration*batch_length + bt
                file_writer.add_summary(summary_str,step)
                        
        if iteration % 1 == 0 : 
            print(len(temp_batch))
            
            mse = loss.eval(feed_dict = {x:x_batch, y:y_batch})
            predicted = pred.eval(feed_dict = {x:x_batch})
            
            print((iteration,"\t ",mse))
            
            f = open(errFile,'a')
            f.write('\n' +  str(iteration) + " Error : "+str(mse))
            f.close()
            
            print(predicted[0])
            print(y_batch[0])
            
            x_list = predicted[0,0:68]
            y_list = predicted[0,68:136]
            
            
            x_listg = y_batch[0,0:68]
            y_listg = y_batch[0,68:136]
            #getting the bounding box of x and y
            
            bb = utils.get_bb(x_list,y_list)
            im2 = x_batch[0]
            
            cv2.rectangle(im2,(bb[0],bb[1]),(bb[2],bb[3]),(255,0,255),1)
            for i in range(68) :
                cv2.circle(im2,(int(x_list[i]),int(y_list[i])),3,(0,255,0))
                cv2.circle(im2,(int(x_listg[i]),int(y_listg[i])),3,(0,0,255))
            
            if not os.path.exists(curDir + "src/"+name_save+"-res/"):
                os.makedirs(curDir + "src/"+name_save+"-res/")
            
            cv2.imwrite(curDir + "src/"+name_save+"-res/image-"+str(iteration)+'.jpg',im2)
            print(curDir + "src/"+name_save+"-res/image-"+str(iteration)+'.jpg')
            
            f = open(errFile,'a')
            f.write('\n' +  str(iteration) + " Error : "+str(mse))
            f.close()
            
            if not os.path.exists(curDir + "src/models/"+name_save):
                os.makedirs(curDir + "src/models/"+name_save)
            
            saver.save(sess,os.path.join(curDir+'src/models/'+name_save, name_save),global_step=iteration)
                
    
    file_writer.close()


def localise(imageName="test.jpg"):
    
    channels = 3
    image_size = 128
    name_save = "dt-inception"
    
    is3D = False
    
    if is3D : 
        name_save += "-3D"
    n_o = 136
    err_name = name_save+"-err"
    
    f = face_localiser(image_size,False,channels)
    x,y,pred = f.build()
    
    print("Now testing")
    saver = tf.train.Saver(max_to_keep=2,save_relative_paths=True)
    
    
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = gpu_fract
    config.gpu_options.visible_device_list = "0"
    
    sess = tf.InteractiveSession(config = config)
    
    print(name_save)
    saver.restore(sess, tf.train.latest_checkpoint('./models/'+name_save))
    
    model = MTCNN()
    
    '''    
    x_min = 0; y_min = 0;
    t = utils.get_bb(y_batch[j,0:n_o/2], y_batch[j,n_o/2:])
        
    l_x = (t[2]-t[0])/4
    l_y = (t[3]-t[1])/4
    
    x1 = max(t[0] - l_x,0)
    y1 = max(t[1] - l_y,0)
    
    x_min = x1; y_min = y1;
    
    #print tImage.shape
    x2 = min(t[2] + l_x,tImage.shape[1])rue
    y2 = min(t[3] + l_y,tImage.shape[0])
    
    croppedImage = tImage[y1:y2,x1:x2];
    #print temp_batch[j],x1,y1,x2,y2
    
    
    height, width, channels = croppedImage.shape

    ratioHeight =truediv(image_size,height)
    ratioWidth =truediv(image_size,width)
    '''
    #print ratioHeight,ratioWidth
    
    #It's assumed that the image is cropped in the center.
    tImage = cv2.imread(imageName)
    
    predictions,_ = model.doFaceDetection(tImage)
        
    if len(predictions) > 0: 
        
        for box in predictions : 
            print(box)
            
            t=np.zeros(4)
            left = t[0] = int(box[0])
            right = t[2]  = int(box[2]) 
            top = t[1]=  int(box[1]) 
            bot = t[3] = int(box[3])
            
            '''cv2.rectangle(tImage,(left, top), (right, bot),(0,255,0),3)
            cv2.imshow('test',tImage)
            cv2.waitKey(0)'''
            
            l_x = (t[2]-t[0])/2 
            l_y = (t[3]-t[1])/2  
            
            x1 = int(max(t[0] - l_x,0))
            y1 = int(max(t[1] - l_y,0))
            x1a = int(t[0] - l_x)
            y1a = int(t[1] - l_y)
            
            #print tImage.shape
            x2 = int(min(t[2] + l_x,tImage.shape[1]))
            y2 = int(min(t[3] + l_y,tImage.shape[0]))
            x2a = int(t[2] + l_x)
            y2a = int(t[3] + l_y)
            
            tIm = np.zeros((y2a-y1a,x2a-x1a,3))
            
            tImage = tImage[int(y1):int(y2),int(x1):int(x2)].copy();#cv2.rectangle(tImage,(left, top), (right, bot),(0,255,0),3)
            
            dx_min,dx_max,dy_min,dy_max = 0,0,0,0
            if x1a < 0 : 
                dx_min = -x1a
                
            if x2a > tImage.shape[1]: 
                dx_max = x2a - tImage.shape[1]
                
            if y1a < 0 : 
                dy_min = -y1a 
                
            if y2a > tImage.shape[2]: 
                dy_max = y2a -tImage.shape[2]
            
            print(dy_min,dy_max,dx_min,dx_max)
                
            '''cv2.imshow('test',tIm)
            cv2.waitKey(0)
            
            '''    
            
            print(tImage.shape,tIm.shape)
            
            print(tIm[0+dy_min:(y2-y1)+dy_min,0+dx_min :(x2-x1)+dx_min].shape)
            tIm[0+dy_min:(y2-y1)+dy_min,0+dx_min :(x2-x1)+dx_min] = tImage
            
            #tImage = utils.padding(tImage)
                    
            '''cv2.imshow('test',tIm)
            cv2.waitKey(0)'''
            
            '''cv2.imshow('test',tImage)
            cv2.waitKey(0)
            cv2.imshow('test',tIm)
            cv2.waitKey(0)'''
            
            
            
            
            #tIm[0:(y2-y1),0:(x2-x1)] = tImage
            tImage = tIm.astype(np.uint8)
    
    print(tImage.shape)
    '''cv2.imwrite('temp.jpg',tImage)
    newImage = cv2.imread('temp.jpg')
    tImage = newImage.copy()'''
    
    height, width, channels = tImage.shape
    ratioHeightR =truediv(height,image_size)
    ratioWidthR =truediv(width,image_size)
    
    print(ratioHeightR,ratioWidthR)
    
    r_image = cv2.resize(tImage, (image_size,image_size))
    
    
    cv2.imshow('test',r_image)
    cv2.waitKey(0)
            
    predicted = pred.eval(feed_dict = {x:np.expand_dims(r_image, axis=0)})
    
    print(predicted)
    #Now recovering from the resized image to original image size
    
    tImage2 = r_image
    x_list = predicted[0,0:68]
    y_list = predicted[0,68:136]

    
    bb = utils.get_bb(x_list,y_list)
    
    cv2.rectangle(tImage2,(bb[0],bb[1]),(bb[2],bb[3]),(255,0,255),1)
    for i in range(68) :
        cv2.circle(tImage2,(int(x_list[i]),int(y_list[i])),3,(0,255,0))
    
    cv2.imshow('result',tImage2)
    cv2.waitKey(0)
     
    x_list = predicted[0,0:68]*ratioWidthR
    y_list = predicted[0,68:136]*ratioHeightR

    
    bb = utils.get_bb(x_list,y_list)
    
    cv2.rectangle(tImage,(bb[0],bb[1]),(bb[2],bb[3]),(255,0,255),1)
    for i in range(68) :
        cv2.circle(tImage,(int(x_list[i]),int(y_list[i])),3,(0,255,0))
    
    cv2.imshow('result',tImage)
    cv2.waitKey(0)
    cv2.imwrite('./data/result.jpg',tImage)
    
    print("done")

localise(imageName)
