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
from config import *
import random
from random import randint
import sys
from face_classifier_simple import face_classifier_simple
from PIL.ImageOps import crop


FLAGS = tf.app.flags.FLAGS
    
tf.flags.DEFINE_boolean("is3D", False,
                        "Whether to train inception submodel variables.")
def evaluate(operation,value):
    if operation == 1 :
        return value 
    else : 
        return -value
    
def inBound(input,min,max):
    if input < min : 
        return int(min) 
    elif input > max : 
        return int(max) 
    return int(input)

def eval(input):
    if input < 0 : 
        return 0
    else :
        return input

def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1, 1)

def initialize_uninitialized_global_variables(sess):
    """
    Only initializes the variables of a TensorFlow session that were not
    already initialized.
    :param sess: the TensorFlow session
    :return:
    """
    # List all global variables
    global_vars = tf.global_variables()

    # Find initialized status for all variables
    is_var_init = [tf.is_variable_initialized(var) for var in global_vars]
    is_initialized = sess.run(is_var_init)

    # List all variables that were not initialized previously
    not_initialized_vars = [var for (var, init) in
                            zip(global_vars, is_initialized) if not init]

    # Initialize all uninitialized variables found, if any
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars)) 
        
def next_batch(num, data, labels):
    '''
    Return a total of maximum `num` random samples and labels.
    NOTE: The last batch will be of size len(data) % num
    '''
    num_el = data.shape[0]
    while True: # or whatever condition you may have
        idx = np.arange(0 , num_el)
        np.random.shuffle(idx)
        current_idx = 0
        while current_idx < num_el:
            batch_idx = idx[current_idx:current_idx+num]
            current_idx += num
            data_shuffle = [data[ i,:] for i in batch_idx]
            labels_shuffle = [labels[ i] for i in batch_idx]
            yield np.asarray(data_shuffle), np.asarray(labels_shuffle)

def train(continuing=False):
    
    is3D = FLAGS.is3D
    
    print(("Training continuing {}".format(continuing)))
    #Now training 
    n_iterations = 1000
    batch_size = 32#1#32;
    if runServer : 
        batch_size = 128
    
    baseSize = 128
    
    if useFullModel : 
        baseSize = baseSize*2
        
    cropSize = baseSize + int(baseSize*.5)
    
    patchOSize = int(baseSize*.1)
    
    print(cropSize)
    
    channels = 3
    d = cv2.HOGDescriptor()
    
    global_step =tf.Variable(0,trainable=False)
    root_log_dir = "tf_logs"
    logdir = "{}/run-{}/".format(root_log_dir,datetime.utcnow().strftime("%Y%m%d%H%M%S"))

    #First get the list of input and bounding boxes. 
    
    name_save = "classifier"
    
    if useFullModel : 
        name_save += "-full"
    
    if is3D : 
        name_save += "-3D"
    
    print("Fetching all data")
    if is3D: 
        all_batch, all_labels, _ = utils.get_kp_face(None, "300W_LP",False, 1,True)
    else : 
        all_batch, all_labels, _ = utils.get_kp_face(None, "300W_LP",False, 1,False)
        
    batch_length = len(all_batch) // batch_size

    errFile = curDir + 'src/err_dt'
    if is3D : 
        errFile += "-3D"
    errFile+=".txt"
    f = open(errFile,'w')
    f.write(' \n Error : ')
    f.close()
    n_augmented = 6#4 rotation, 1 random, 1 gray area
    
    n_rot = 8 
    
    print("Now facial clasifier")
    f = face_classifier_simple(patchOSize, batch_size+n_rot*n_augmented*batch_size)
    x_b,y_b,pred = f.build()
    
    
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "0"
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    
    sess = tf.Session(config = config)
        
    print("Now summary saver")
    
    '''loss  = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))'''
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = pred, labels=y_b))#tf.reduce_mean(-(y_b*tf.log(pred)+(1-y_b)*tf.log(1-pred)))#tf.reduce_mean(tf.square(pred - y_b))
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before performing the train_step
        training_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)#tf.train.MomentumOptimizer(0.02, momentum=0.01)
        
    correct_prediction = tf.equal(y_b,tf.round(tf.nn.sigmoid(pred)))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
    mse_summary = tf.summary.scalar('Loss',loss)
    file_writer = tf.summary.FileWriter(logdir,tf.get_default_graph())
    
    print("Now training")
    saver = tf.train.Saver(max_to_keep=2,save_relative_paths=True)
    #sess.run(tf.global_variables_initializer())
    initialize_uninitialized_global_variables(sess)
    
    doTransformation = True
    
    if continuing : 
        saver.restore(sess, tf.train.latest_checkpoint('./models/'+name_save))
    
    patchSize = int(patchOSize/2)
    
    for iteration in range(n_iterations) :
        print(("it-",iteration))
        counter = 0
                
        
        for bt in range(0,batch_length) :
            op = np.array([[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1],[0,1]])            
            x_batch = np.zeros([batch_size+n_rot*n_augmented*batch_size,68,patchOSize,patchOSize,channels])
            y_batch = np.zeros([batch_size+n_rot*n_augmented*batch_size,1])
            
            l_image = np.zeros([batch_size+n_rot*n_augmented*batch_size,cropSize,cropSize,channels])
            
            temp_batch = [] 
            cBBPoints = []

            #print counter,batch_length,len(all_batch)
            for b_i in range(counter,counter+batch_size):
                #fetch the data for each batch 
                temp_batch.append(all_batch[b_i])
                cBBPoints.append(all_labels[b_i])
                
            cBBPoints = np.array(cBBPoints)
            
            
            counter+=batch_size
            
            for j in range(batch_size):
                
                gt = cBBPoints[j]
                
                t = utils.get_bb(gt[:68], gt[68:])
                
                try :
                    tImage = cv2.imread(temp_batch[j])
                except : 
                    print(temp_batch[j])
                    
                sel = randint(0,2)#iteration%2
                rad = randint(-3,3)
                
                
                tImage = cv2.imread(temp_batch[j])
                #print(temp_batch[j])
                
                if doTransformation : 
                    if sel == 0 : 
                        out = [tImage,gt]
                    elif sel == 1 : #Do flipping  
                        out = utils.transformation(tImage,gt, 1, 1)
                    elif sel == 2 : #Do rotation
                        out = utils.transformation(tImage,gt, 2, 5*rad)
                    '''elif sel == 3 : #Do occlusion 
                        out = utils.transformation(tImage,gt, 3, 1)'''
                    tImage = out[0];gt = out[1]
                    
                
                
                l_x = (t[2]-t[0])/2 + (t[2]-t[0])/4
                l_y = (t[3]-t[1])/2 + (t[3]-t[1])/4 
                
                x1 = int(max(t[0] - l_x,0))
                y1 = int(max(t[1] - l_y,0))
                
                #print tImage.shape
                x2 = int(min(t[2] + l_x,tImage.shape[1]))
                y2 = int(min(t[3] + l_y,tImage.shape[0]))
                
                tImage = tImage[y1:y2,x1:x2]
                
                height, width,_ = tImage.shape
                ratioHeight =truediv(cropSize,height)
                ratioWidth =truediv(cropSize,width)
                            
                tImage = cv2.resize(tImage,(cropSize,cropSize))
                
                
                #Now fixing the groundtruth 
                kpX = (gt[:68] - x1)*ratioWidth
                kpY = (gt[68:] - y1)*ratioHeight
                
                
                copyImage = tImage.copy()
                
                #for each point take the fixed size of patch
                for k in range(68) : 
                    x,y = int(kpX[k]),int(kpY[k])
                    t_image = np.zeros([patchOSize,patchOSize,3])
                    x1,x2 = inBound(int(x-patchSize),0,tImage.shape[1]),inBound(int(x+patchSize),0,tImage.shape[1])
                    y1,y2 = inBound(int(y-patchSize),0,tImage.shape[0]), inBound(int(y+patchSize),0,tImage.shape[0]) 
                    try : 
                        t_image[0:(y2-y1),0:(x2-x1)] = tImage[y1:y2,x1:x2]
                    except : 
                        print("error : ")
                        print("x,y : ",x,y)
                        print("l_x, l_y : ",l_x,l_y)
                        print(x1,":",x2,", ",y1,":",y2)
                    finally : 
                        x_batch[j,k] = t_image.copy()
                
                    if True : #bt == batch_length-1 : 
                        #x1,y1, x2,y2 
                        cv2.rectangle(copyImage,(inBound(x-patchSize,0,tImage.shape[1]), inBound(y+-patchSize,0,tImage.shape[0])),
                                       (inBound(x+patchSize,0,tImage.shape[1]), inBound(y+ patchSize,0,tImage.shape[0])),(0,255,0),1)
                        
                        cv2.circle(copyImage,(inBound(x,0,tImage.shape[1]),
                                           inBound(y,0,tImage.shape[0])),3,(0,0,255))
                        
                        #cv2.imwrite("image_"+str(j)+"_"+str(k)+".jpg",t_image)
                    
                        '''cv2.imshow('teest',copyImage)
                        cv2.waitKey(0)'''
                
                
                if bt == batch_length-1 : 
                    l_image[j] = copyImage
                    #print("getting ori ",j)
                y_batch[j,0] = 1
                
                #print j
                
                t = utils.get_bb(gt[:68], gt[68:])
                l_x = int((t[2]-t[0])/2)
                l_y = int((t[3]-t[1])/2)
                
                for ix in range(n_rot*n_augmented) : 
                    if ix < n_rot : 
                        displacement = random.uniform( .025, .05 )
                        #displacement = 0.005
                        y_batch[j*n_rot*n_augmented+batch_size+ix,0] = 1
                    else : 
                        displacement = random.uniform( .075, .20 )
                        #displacement = random.uniform( .5, .75 )
                        y_batch[j*n_rot*n_augmented+batch_size+ix,0] = 0
                    disp = op[ix%8]
                        
                    copyImage = tImage.copy()
                    
                    for k in range(68) :
                        if ix > (n_rot-2)*n_augmented : 
                            x,y = randint(0,cropSize),randint(0,cropSize)
                        else :
                            x,y = int(kpX[k]),int(kpY[k])
                        t_image = np.zeros([patchOSize,patchOSize,3])
                        x1,x2 = inBound(x+ disp[1]*int(l_x*displacement)-patchSize,0,tImage.shape[1]),inBound(x+ disp[1]*int(l_x*displacement)+patchSize,0,tImage.shape[1])
                        y1,y2 = inBound(y+ disp[0]*int(l_y*displacement)-patchSize,0,tImage.shape[0]),inBound(y+ disp[0]*int(l_y*displacement)+patchSize,0,tImage.shape[0]) 
                        try : 
                            if ix < (n_rot-1) * n_augmented: 
                                t_image[0:(y2-y1),0:(x2-x1)] = tImage[y1:y2,x1:x2]
                            else :
                                t_image = np.zeros([patchOSize,patchOSize,3]) + randint(0,255)
                        except: 
                            print ("Error1 ")
                            print("error : ")
                            print("x,y : ",x,y)
                            print("l_x, l_y : ",l_x,l_y)
                            print(displacement)
                            print(x1,":",x2,", ",y1,":",y2)
                        finally :       
                            x_batch[j*n_rot*n_augmented+batch_size+ix,k] = t_image.copy()
                        
                        
                        if bt == batch_length-1 : 
                            cv2.rectangle(copyImage,(inBound(x+ disp[1]*int(l_x*displacement)-patchSize,0,tImage.shape[1]), inBound(y+ disp[0]*int(l_y*displacement)-patchSize,0,tImage.shape[0])),
                                       (inBound(x+ disp[1]*int(l_x*displacement)+patchSize,0,tImage.shape[1]), inBound(y+ disp[0]*int(l_y*displacement)+patchSize,0,tImage.shape[0])),(0,255,0),1)
                        
                            cv2.circle(copyImage,(inBound(x+ disp[1]*int(l_x*displacement),0,tImage.shape[1]),
                                           inBound(y+ disp[0]*int(l_y*displacement),0,tImage.shape[0])),3,(0,0,255))
                            
                            
                        
                        #x1,y1, x2,y2 
                        '''cv2.rectangle(tImage,(inBound(x+ disp[1]*int(l_x*displacement)-patchSize,0,tImage.shape[1]), inBound(y+ disp[0]*int(l_y*displacement)-patchSize,0,tImage.shape[0])),
                                       (inBound(x+ disp[1]*int(l_x*displacement)+patchSize,0,tImage.shape[1]), inBound(y+ disp[0]*int(l_y*displacement)+patchSize,0,tImage.shape[0])),(0,255,0),1)
                        
                        cv2.circle(tImage,(inBound(x+ disp[1]*int(l_x*displacement),0,tImage.shape[1]),
                                           inBound(y+ disp[0]*int(l_y*displacement),0,tImage.shape[0])),3,(0,0,255))
                        
                        cv2.imshow('teest',tImage)
                        print('oy',ix,(n_rot-2)*n_augmented)
                        cv2.waitKey(0)'''
                            
                    if bt == batch_length-1 : 
                        l_image[j*n_rot*n_augmented+batch_size+ix] = copyImage
                        #print("getting the other ",j*n_rot*3+batch_size+ix)
                
            #print x_batch.shape,y_batch.shape
            #Now calculating the loss 
            #shuffle x and y batch 
            
            permutation=np.random.permutation(len(x_batch))
            batch=[x_batch[permutation],y_batch[permutation]]
            
            t_loss,predict,acc,_= sess.run([loss,pred,accuracy,training_op],feed_dict = {x_b:batch[0], y_b:batch[1]})
            #print "gv:",gv2
            #print(predict.shape)
            
            if bt%5 == 0: 
                print(("Batch {}/{}".format(bt,batch_length)))
                summary_str = mse_summary.eval(feed_dict = {x_b:batch[0], y_b:batch[1]},session = sess)
                step = iteration*batch_length + bt
                file_writer.add_summary(summary_str,step)
                print((t_loss,'-',acc))
                #print(predict,batch[1])
                #print predict,y_batch
                '''if t_loss > 0: 
                    print predict, y_batch
                    '''
                #print predict,y_batch
                f = open(errFile,'a')
                f.write("Batch {}/{} err : {} \t".format(bt,batch_length,t_loss))
                f.close()
            
            if bt%10000 == 0 : 
                saver.save(sess,os.path.join(curDir+'src/models/'+name_save, name_save),global_step=iteration)
                        
        if iteration % 1 == 0 :
            mse = loss.eval(feed_dict = {x_b:batch[0], y_b:batch[1]},session = sess)
            predicted = pred.eval(feed_dict = {x_b:batch[0]},session = sess)
            
            
            print((iteration,"\t ",mse))
            
            f = open(errFile,'a')
            f.write('\n' +  str(iteration) + " Error : "+str(mse))
            f.close()
            
            #print(predicted)
            #print(y_batch)
            
            f = open(errFile,'a')
            f.write('\n' +  str(iteration) + " Error : "+str(mse))
            f.close()
            
            saver.save(sess,os.path.join(curDir+'src/models/'+name_save, name_save),global_step=iteration)
            
            for iter in range (batch_size+n_rot*n_augmented*batch_size) : 
                t_pred = int(predicted[iter])
                #print(curDir+'src/res-dt/train_dt_s_'+str(iteration)+"_"+str(iter)+"["+str(t_pred)+"-"+str(int(y_batch[iter]))+"].jpg")
                theImage = l_image[iter];
                #cv2.imwrite(curDir+'src/res-dt/train_dt_s_'+str(iteration)+"_"+str(iter)+"["+str(t_pred)+"-"+str(int(y_batch[iter]))+"].jpg",theImage)
                #cv2.imshow("Predicted : "+str(predicted[iter]),theImage)
                #cv2.waitKey(0)
                
    
    file_writer.close()


def test(imageName=curDir+"src/picts/back1.png"):
    
    image_size = 128

    name_save = "classifier"
    
    f = face_classifier(image_size,True)
    x,y,pred = f.build()
    
    print("Now testing")
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    initialize_uninitialized_global_variables(sess)
    saver.restore(sess, tf.train.latest_checkpoint(curDir+'src/models/'+name_save))
        
    #print ratioHeight,ratioWidth
    
    #It's assumed that the image is cropped in the center.
    tImage = cv2.imread(imageName)
    r_image = cv2.resize(tImage, (image_size,image_size))
    
    x_input = np.expand_dims(r_image, axis=0)
    print((x_input.shape))
    
    patch = [] 
    for k in range(68) : 
        x,y = cBBPoints[j,k],cBBPoints[j,k+68]
        patch.append(tImage[int(y-patchSize):int(y+patchSize),int(x-patchSize):int(x+patchSize)])
        
    x_batch[j] = np.asarray(patch,np.float32)
    
    
    t = utils.get_bb(cBBPoints[j,:68], cBBPoints[j,68:])
    l_x = (t[2]-t[0])/2
    l_y = (t[3]-t[1])/2
    
    predicted = pred.eval(feed_dict = {x:x_input})
    
    print(predicted)
    print("done")

#test()
train(True)
#test("train_dt_s145.jpg")

    