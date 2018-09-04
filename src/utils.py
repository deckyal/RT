import numpy as np
import file_walker
import re
import cv2
from operator import truediv
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import random
from config import *
from scipy.integrate.quadrature import simps
import math
from scipy.stats import multivariate_normal
import os
from random import randint
import glob
from scipy.integrate import simps

def calc_bb_IOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou

def read_kp_file(filename):
     
    x = []
    
    if ('pts' in filename) :
        with open(filename) as file:
            data2 = [re.split(r'\t+',l.strip()) for l in file]
        for i in range(len(data2)) :
            if(i not in [0,1,2,len(data2)-1]):
                x.append([ float(j) for j in data2[i][0].split()] )
    return np.asarray(x)

def errCalc(catTesting = 1, localize = False, t_dir = "300W-Test/01_Indoor/",name='Re3A',is3D=False,ext = ".txt",makeFlag = False):
    catTesting = catTesting;
    l_error = []
    
    if localize is False :#image dir 
        if is3D: 
            dir = curDir + 'images/300VW-Test_M/cat'+str(catTesting)+'/'
        else : 
            dir = curDir + 'images/300VW-Test/cat'+str(catTesting)+'/'
    else :  
        dir = curDir + t_dir +'/'
        
    list_txt = glob.glob1(dir,"*"+ext)
    
    for x in list_txt: 
        print(("Opening " +dir+x))
        
        file = open(dir+x)
        for line in file : 
            #print float(line)
            l_error.append(float(line))
            
        file.close()
    all_err = np.array(l_error)
    
    if makeFlag : 
        list_txt = glob.glob1(dir,"*"+ext)
        l_tr = []
        l_d = []
        
        for x in list_txt: 
            print(("Opening " +dir+x))
            
            file = open(dir+x)
            for line in file : 
                data = [ float(j) for j in line.split()] 
                #print(data)
                l_tr.append(float(data[1]))
                l_d.append(float(data[0]))
                
            file.close()
        
    
    if localize is False :    
        fileName = "src/result_compared/cat"+str(catTesting)+"/"
        aboveT = makeErrTxt(all_err,fileName= fileName+name+".txt",threshold = .08,lim = 1.1005)
        
        if makeFlag : 
            l_tr = np.asarray(l_tr);
            l_d = np.asarray(l_d);
            
            f = open(curDir+fileName+"flag.txt",'w')
            
            am_r = truediv(len(l_tr[np.where(l_tr > 0 )]),len(l_tr));
            am_d = truediv(len(l_d[np.where(l_d == 0 )]),len(l_d));
            
            f.write("%.4f %.4f\n" % (am_r,am_d));    
            f.close()
            
        print(("Above T ",name," : "+str(aboveT)))
        plot_results(catTesting,resFolder= 'src/result_compared/cat'+str(catTesting),addition=[name],is3D=is3D)
    else : #error dir 
        arrName = ['src/result_compared/300W/Indoor','src/result_compared/300W/Outdoor','src/result_compared/300W/InOut']
        aboveT = makeErrTxt(all_err,fileName= arrName[catTesting]+"/"+name+".txt",threshold = .08)
        print(("Above T ",name," : "+str(aboveT)))
        plot_results(catTesting+4,resFolder= arrName[catTesting],addition=[name],is3D=is3D)
    
    return all_err
    #print(("All error : "+str(all_err)))

def makeErrTxt(error,fileName = 'result_compared/Decky.txt',threshold = .08,lim = .35005):
    bin = np.arange(0,lim,0.0001)#0.35005,0.0005), 300vw 1.1005
    
    #res = np.array([len(bin)])
    
    #creating the file 
    f = open(curDir+fileName,'w')    
    f.write('300W Challenge 2013 Result\n');
    f.write('Participant: Decky.\n');
    f.write('-----------------------------------------------------------\n');
    f.write('Bin 68_all 68_indoor 68_outdoor 51_all 51_indoor 51_outdoor\n');

    for i in range(len(bin)) : 
        err = truediv(len(error[np.where(error < bin[i])]),len(error))
        f.write("%.4f %.4f %.4f %.4f %.4f %.4f %.4f\n" % (bin[i],err, err,err, err, err, err));    
    f.close()
    err_above = truediv(len(error[np.where(error > threshold )]),len(error));
    print((error[np.where(error > threshold )]))
    return err_above 


    
def plot_results(version, resFolder = 'result_compared',x_limit=0.08, colors=None, markers=None, linewidth=3,
                 fontsize=12, figure_size=(11, 6),addition = None,is3D = False,All = False):
    """
    Method that generates the 300W Faces In-The-Wild Challenge (300-W) results
    in the form of Cumulative Error Distributions (CED) curves. The function
    renders the indoor, outdoor and indoor + outdoor results based on both 68
    and 51 landmark points in 6 different figures.

    Please cite:
    C. Sagonas, E. Antonakos, G. Tzimiropoulos, S. Zafeiriou, M. Pantic. "300
    Faces In-The-Wild Challenge: Database and Results", Image and Vision
    Computing, 2015.
    
    Parameters
    ----------
    version : 1 or 2
        The version of the 300W challenge to use. If 1, then the reported
        results are the ones of the first conduct of the competition in the
        ICCV workshop 2013. If 2, then the reported results are the ones of
        the second conduct of the competition in the IMAVIS Special Issue 2015.
    x_limit : float, optional
        The maximum value of the horizontal axis with the errors.
    colors : list of colors or None, optional
        The colors of the lines. If a list is provided, a value must be
        specified for each curve, thus it must have the same length as the
        number of plotted curves. If None, then the colours are linearly sampled
        from the jet colormap. Some example colour values are:

                'r', 'g', 'b', 'c', 'm', 'k', 'w', 'orange', 'pink', etc.
                or
                (3, ) ndarray with RGB values

    linewidth : float, optional
        The width of the rendered lines.
    fontsize : int, optional
        The font size that is applied on the axes and the legend.
    figure_size : (float, float) or None, optional
        The size of the figure in inches.
    """
    
    if not is3D :
        title = "300VW 2D "
    else : 
        title = "300VW 3DA-2D "  
    # Check version
    if version == 1:
        participants = ['Dlssvm_Cfss', 'MD_CFSS', 'Mdnet_DlibERT', 'Meem_Cfss', 'Spot_Cfss']
        if not All : 
            title += 'category 1' 
    elif version == 2:
        participants = ['ccot_cfss', 'MD_CFSS', 'spot_cfss', 'srdcf_cfss']
        if not All : 
            title += 'category 2'
    elif version == 3:
        participants = ['ccot_cfss', 'MD_CFSS', 'meem_cfss', 'srdcf_cfss','staple_cfss']
        if not All : 
            title += 'category 3'
    elif version in [4,5,6]:
        if is3D : 
            participants=[]
            l_participants = ['Re3A_3D','Re3A_C_3D','FA_3D']
            for z in l_participants : 
                if z not in participants : 
                    participants.append(z)
        else: 
            #participants = ['Baltrusaitis', 'Hasan',  'Jaiswal','Milborrow','Yan','Zhou']
            participants = []
            participants.append('Re3A')
            participants.append('Re3A_C')
            participants.append('FA')
            
        arrName = ['Indoor','Outdoor','Indoor + Outdoor']
        if not All : 
            title = arrName[version - 4]
    else:
        raise ValueError('version must be either 1 or 2')
    
    if All : 
        title += " All Category "
    participants = []
    if version in [1,2,3]:
        participants = []
        if is3D :  
            #participants = []
            #participants.append('Re3A_3D')
            #participants.append('Re3A_C_3D')
            participants.append('RT_MT_3D')
            participants.append('RT_2_3D')
            participants.append('RT_4_3D')
            participants.append('RT_8_3D')
            participants.append('RT_16_3D')
            participants.append('RT_32_3D')
            
            participants.append('FA_MD_3D')
            participants.append('FA_MT_3D')
            participants.append('3DFFA_MD_3D')
            participants.append('3DFFA_MT_3D')
            colors = ['b','red','orange','yellow','yellow','yellow','green','brown','k','purple']
        else: 
            participants.append('RT_MT')
            participants.append('RT_2')
            participants.append('RT_4')
            participants.append('RT_8')
            participants.append('RT_16')
            participants.append('RT_32')
            participants.append('YANG')
            participants.append('MD_CFSS')
            participants.append('ME_CFSS')
            #participants.append('FA_MD')
            #participants.append('FA_MT')
            colors = ['b','red','orange','yellow','yellow','yellow','g','brown','k']
            
            #participants.append('Re3A')
            #participants.append('Re3A_C')
            #participants.append('FA_MD')
        #participants = []
    if addition is not None :
        for i in addition :  
            if i not in participants : 
                participants.append(i)
        
    # Initialize lists
    ced68 = []
    ced68_indoor = []
    ced68_outdoor = []
    ced51 = []
    ced51_indoor = []
    ced51_outdoor = []
    legend_entries = []

    # Load results
    results_folder = curDir+resFolder
    for f in participants:
        # Read file
        if 'Re3A' in f  or version in  [1,2,3,6]:
            index = 1
        elif version == 4 :#indoor 
            index = 2;
        elif version == 5 :#outdoor
            index = 3;
            
        filename = f + '.txt'
        tmp = np.loadtxt(str(Path(results_folder) / filename), skiprows=4)
        # Get CED values
        bins = tmp[:, 0]
        ced68.append(tmp[:, index])
        
        '''ced68_indoor.append(tmp[:, 2])
        ced68_outdoor.append(tmp[:, 3])
        ced51.append(tmp[:, 4])
        ced51_indoor.append(tmp[:, 5])
        ced51_outdoor.append(tmp[:, 6])'''
        # Update legend entries
        legend_entries.append(f)# + ' et al.')
    
    idx = [x[0] for x in np.where(bins==.0801)] #.0810
    real_bins =  bins[:idx[0]]
    
    
    print(idx,real_bins)
    
    for i in range(len(ced68)) : 
        
        real_ced =  ced68[i][:idx[0]]
        #print(real_ced)
        #AUC = str(round(simps(real_ced,real_bins) * (1/x_limit),3))
        AUC = str(round(simps(real_ced,real_bins) * (1/x_limit),5))
        FR = str(round(1. - real_ced[-1],5)) #[-3]
        
        #print(real_bins[-1])
        
        #print(legend_entries[i] + " : "+str(simps(real_ced,real_bins) * (1/x_limit)))
        
        print(legend_entries[i] + " : " +AUC+" FR : "+FR) 
        #legend_entries[i]+=" [AUC : "+AUC+"]"#+"] [FR : "+FR+"]"
        
        #plt.plot(real_bins,real_ced)
        #plt.show()
    # 68 points, indoor + outdoor    
    _plot_curves(bins, ced68, legend_entries, title, x_limit=x_limit,
                 colors=colors, linewidth=linewidth, fontsize=fontsize,
                 figure_size=figure_size)
    
    '''# 68 points, indoor
    title = 'Indoor, 68 points'
    _plot_curves(bins, ced68_indoor, legend_entries, title, x_limit=x_limit,
                 colors=colors, linewidth=linewidth, fontsize=fontsize,
                 figure_size=figure_size)
    # 68 points, outdoor
    title = 'Outdoor, 68 points'
    _plot_curves(bins, ced68_outdoor, legend_entries, title, x_limit=x_limit,
                 colors=colors, linewidth=linewidth, fontsize=fontsize,
                 figure_size=figure_size)
    # 51 points, indoor + outdoor
    title = 'Indoor + Outdoor, 51 points'
    _plot_curves(bins, ced51, legend_entries, title, x_limit=x_limit,
                 colors=colors, linewidth=linewidth, fontsize=fontsize,
                 figure_size=figure_size)
    # 51 points, indoor
    title = 'Indoor, 51 points'
    _plot_curves(bins, ced51_indoor, legend_entries, title, x_limit=x_limit,
                 colors=colors, linewidth=linewidth, fontsize=fontsize,
                 figure_size=figure_size)
    # 51 points, outdoor
    title = 'Outdoor, 51 points'
    _plot_curves(bins, ced51_outdoor, legend_entries, title, x_limit=x_limit,
                 colors=colors, linewidth=linewidth, fontsize=fontsize,
                 figure_size=figure_size)'''
    
    
def _plot_curves(bins, ced_values, legend_entries, title, x_limit=0.08,
                 colors=None, linewidth=3, fontsize=12, figure_size=None):
    # number of curves
    n_curves = len(ced_values)
    
    # if no colors are provided, sample them from the jet colormap
    if colors is None:
        cm = plt.get_cmap('jet')
        colors = [cm(1.*i/n_curves)[:3] for i in range(n_curves)]
        
    # plot all curves
    fig = plt.figure()
    ax = plt.gca()
    for i, y in enumerate(ced_values):
        plt.plot(bins, y, color=colors[i],
                 linestyle='-',
                 linewidth=linewidth, 
                 label=legend_entries[i])
        #print bins.shape, y.shape
    # legend
    ax.legend(prop={'size': fontsize}, loc=4)
    
    # axes
    for l in (ax.get_xticklabels() + ax.get_yticklabels()):
        l.set_fontsize(fontsize)
    ax.set_xlabel('Normalized Point-to-Point Error', fontsize=fontsize)
    ax.set_ylabel('Images Proportion', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    # set axes limits
    ax.set_xlim([0., x_limit])
    ax.set_ylim([0., 1.])
    ax.set_yticks(np.arange(0., 1.1, 0.1))
    
    # grid
    plt.grid('on', linestyle='--', linewidth=0.5)
    
    # figure size
    if figure_size is not None:
        fig.set_size_inches(np.asarray(figure_size))
    
    plt.show()

def make_heatmap(image_name,t_image,add,y_batch,isRandom = True,percent_heatmap = .1,percent_heatmap_e = .05):
    
    tBase = os.path.basename(image_name)
    tName,tExt = os.path.splitext(tBase)
    theDir =  os.path.dirname(image_name)+"/../heatmap-"+add+"/"
    
    if not os.path.exists(theDir):
        os.makedirs(theDir)
        
    fName =theDir+tName+".npy"
    
    #print(fName)
    try : 
        b_channel,g_channel,r_channel = t_image[:,:,0],t_image[:,:,1],t_image[:,:,2]
    except : 
        print(image_name)
    
    if os.path.isfile(fName) and isRandom: 
        newChannel = np.load(fName)
        print("using saved npy")
    else :    
        print("make npy "+add)
        newChannel = b_channel.copy(); newChannel[:] = 0
        y_t = y_batch
        
        if isRandom : 
            t0,t1,t2,t3 = get_bb(y_t[0:int(n_o//2)], y_t[int(n_o//2):],68,False,
                                 random.uniform( -.25, .25 ),
                                 random.uniform( -.25, .25 ),
                                 random.uniform( -.25, .25 ),
                                 random.uniform( -.25, .25 ),
                                 random.uniform( -.25, .25 ))
        else : 
            t0,t1,t2,t3 = get_bb(y_t[0:int(n_o//2)], y_t[int(n_o//2):],68,False)
        #print(t0,t1,t2,t3)
        
        l_cd,rv = get_list_heatmap(0,None,t2-t0,t3-t1,percent_heatmap)
        l_cd_e,rv_e = get_list_heatmap(0,None,t2-t0,t3-t1,percent_heatmap_e)
        
        height, width,_ = t_image.shape
        
        scaler = 255/np.max(rv)
        #addOne = randint(0,2),addTwo = randint(0,2)
        for iter in range(68) :
            #print(height,width)
            if random: 
                ix,iy = int(y_t[iter]),int(y_t[iter+68])
            else : 
                ix,iy = int(y_t[iter])+randint(0,2),int(y_t[iter+68])+randint(0,2)
            #Now drawing given the center
            if iter in range(36,48): 
                l_cd_t = l_cd_e
                rv_t = rv_e
            else : 
                l_cd_t = l_cd
                rv_t = rv
            
            for iter2 in range(len(l_cd_t)) : 
                value = int(rv_t[iter2]*scaler)
                if newChannel[inBound(iy+l_cd_t[iter2][0],0,height-1), inBound(ix + l_cd_t[iter2][1],0,width-1)] < value : 
                    newChannel[inBound(iy+l_cd_t[iter2][0],0,height-1), inBound(ix + l_cd_t[iter2][1],0,width-1)] = int(rv_t[iter2]*scaler)#int(heatmapValue/2 + rv[iter2] * heatmapValue)
        
        #np.save(fName,newChannel)
    
    return newChannel

def get_enlarged_bb(the_kp = None, div_x = 2, div_y = 2, images = None,is_bb = False):
    
    if not is_bb : 
        t = get_bb(the_kp[:68],the_kp[68:])
    else : 
        t = the_kp
                
    l_x = (t[2]-t[0])/div_x
    l_y = (t[3]-t[1])/div_y
    
    x1 = int(max(t[0] - l_x,0))
    y1 = int(max(t[1] - l_y,0))
    
    x_min = x1; y_min = y1;
    
    #print tImage.shape
    x2 = int(min(t[2] + l_x,images.shape[1]))
    y2 = int(min(t[3] + l_y,images.shape[0]))
    
    return t,l_x,l_y,x1,y1,x_min,y_min,x2,y2

def inBoundN(input,min,max):
    if input < min : 
        return min 
    elif input > max : 
        return max 
    return input

def inBound(input,min,max):
    if input < min : 
        return int(min) 
    elif input > max : 
        return int(max) 
    return int(input)

def inBound_tf(input,min,max):
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


def addPadd(im) : 
    #im = cv2.imread("./test-frontal.png")
    height, width, channels =im.shape
    desired_size = np.max(np.array([height,width]))
    
    add_x,add_y = 0,0
    
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    if height > width : #so shift x 
        add_x = left
    else:
        add_y = top 
        
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    #print top,bottom,left,right
    '''cv2.imshow("image", new_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    return new_im,add_x,add_y

def transformation(input, gt, type, info,length = 68 ):
    mapping =[ 
        [0,16], 
        [1,15], 
        [2,14], 
        [3,13], 
        [4,12], 
        [5,11], 
        [6,10], 
        [7,9], 
        [8,8], 
        [9,7], 
        [10,6], 
        [11,5], 
        [12,4], 
        [13,3], 
        [14,2], 
        [15,1], 
        [16,0],
         
        [17,26], 
        [18,25], 
        [19,24], 
        [20,23], 
        [21,22], 
        [22,21], 
        [23,20], 
        [24,19], 
        [25,18], 
        [26,17], 
        
        [27,27], 
        [28,28], 
        [29,29], 
        [30,30], 
        
        [31,35], 
        [32,34], 
        [33,33], 
        [34,32], 
        [35,31],
         
        [36,45], 
        [37,44], 
        [38,43], 
        [39,42], 
        [40,47], 
        [41,46],
         
        [42,39], 
        [43,38], 
        [44,37], 
        [45,36], 
        [46,41], 
        [47,40],
        
         
        [48,54], 
        [49,53], 
        [50,52], 
        [51,51], 
        [52,50], 
        [53,49], 
        [54,48],
         
        [55,59], 
        [56,58], 
        [57,57], 
        [58,56], 
        [59,55],
         
        [60,64], 
        [61,63], 
        [62,62], 
        [63,61], 
        [64,60],
         
        [65,67], 
        [66,66], 
        [67,65],
        ]
    
    mapping84 =[ 
        [0,32], 
        [1,31], 
        [2,30], 
        [3,29], 
        [4,28], 
        [5,27], 
        [6,26], 
        [7,25], 
        [8,24], 
        [9,23], 
        [10,22], 
        [11,21], 
        [12,20], 
        [13,19], 
        [14,18], 
        [15,17], 
        [16,16],
        [17,15], 
        [18,14], 
        [19,13], 
        [20,12], 
        [21,11], 
        [22,10], 
        [23,9], 
        [24,8], 
        [25,7], 
        [26,6], 
        [27,5], 
        [28,4], 
        [29,3], 
        [30,2], 
        [31,1], 
        [32,0], 
         
        [33,42], 
        [34,41], 
        [35,40], 
        [36,39], 
        [37,38], 
        [38,37], 
        [39,36], 
        [40,35], 
        [41,34], 
        [42,33], 
        
        [43,46], 
        [44,45], 
        [45,44], 
        [46,43], 
        
        [47,51], 
        [48,50], 
        [49,49], 
        [50,48], 
        [51,47],
         
        [52,57], 
        [53,56], 
        [54,55], 
        [55,54], 
        [56,53], 
        [57,52],
         
        [58,63], 
        [59,62], 
        [60,61], 
        [61,60], 
        [62,59], 
        [63,58],
        
         
        [64,70], 
        [65,69], 
        [66,68], 
        [67,67], 
        [68,66], 
        [69,65], 
        [70,64],
         
        [71,75], 
        [72,74], 
        [73,73], 
        [74,72], 
        [75,71],
         
        [76,80], 
        [77,79], 
        [78,78], 
        [79,77], 
        [80,76],
         
        [81,83], 
        [82,82], 
        [83,81],
        ]
    if length > 68: 
        mapping = np.asarray(mapping84)
    else :
        mapping = np.asarray(mapping)
        
    if type == 1 : 
        #print("Flippping") #info is 0,1
        
        gt_o = gt.copy()
        height, width,_ = input.shape
        
        if info == 0 : #vertical 
            #print("Flipping vertically ^v")
            output = cv2.flip(input,0)
            
            for i in range(length) : 
                    
                if gt_o[i+length] > (height/2) : #y 
                    gt_o[i+length] = height/2 -  (gt[i+length] -(height/2))
                if gt_o[i+length] < (height/2) : #y 
                    gt_o[i+length] = height/2 + ((height/2)-gt[i+length])

        elif info == 1 : #horizontal 
            t_map = mapping[:,1]
            
            #gt_o_t = gt.copy()
            
            #print("Flipping Horizontally <- ->  ")
            #return np.fliplr(input)
            output = cv2.flip(input,1)
            
            for i in range(length) : 
                    
                if gt[i] > (width/2) : #x 
                    #gt_o_t[i] = (width/2) - (gt[i] - (width/2))
                    gt_o[t_map[i]] = (width/2) - (gt[i] - (width/2))
                if gt[i] < (width/2) : #x 
                    #gt_o_t[i] = (width/2) + ((width/2) - gt[i])
                    gt_o[t_map[i]] = (width/2) + ((width/2) - gt[i])
                #get the new index 
                #gt_o[t_map[i]] = gt_o_t[i]
                
                gt_o[t_map[i]+length] = gt[i+length]
                    
            #needs to be transformed. 
        
        return [output,gt_o]
    elif type == 2 : 
        #print("Rotate") # info is 1,2,3
        #output = np.rot90(input,info)
        rows,cols,_ = input.shape
    
        M = cv2.getRotationMatrix2D((cols/2,rows/2),info,1)
        output = cv2.warpAffine(input,M,(cols,rows))
        
        gt_o = np.array([gt[:length]-(cols/2),gt[length:]-(rows/2)])
        
        theta = np.radians(-info)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c,-s), (s, c)))
        gt_o = np.dot(R,gt_o)
        
        gt_o =  np.concatenate((gt_o[0]+(cols/2),gt_o[1]+(rows/2)),axis = 0)
        
        '''
        print R.shape, gt_o.shape
        print gt_o.shape'''
        
        
        return [output,gt_o]
        
    elif type == 3 : #info is 0 to 1
        #print("Occlusion")
        
        output = input.copy()
        gt_o = gt.copy()
        
        lengthW = 0.5
        lengthH = 0.4
        
        s_row = 15
        s_col = 7
        
        imHeight,imWidth,_ = input.shape
        
        #Now filling the occluder 
        l_w = imHeight//s_row 
        l_h = imWidth//s_col 
         
        for ix in range(s_row):
            for jx in range(s_col):
                #print ix,jx,l_w,l_h
                #y1:y2, x1:x2
                
                #print(ix*b_size,outerH ,jx*b_size,outerW,'--',outerImgH,',',outerImgW )
                #print(ix*l_w,ix*l_w+l_w ,jx*l_h,jx*l_h+l_h )
                output[ix*l_w:ix*l_w+int(l_w*lengthH) ,jx*l_h:jx*l_h+int(l_h*lengthW) ] = np.full([int(l_w*lengthH),int(l_h*lengthW),3],255)
        
        return [output,gt_o]
    


def calcListNormalizedDistance(pred,gt):
    '''
    input : 
        pred : num_images,num points
        gt : num_images, num points 
    '''
    err = np.zeros(len(pred))
    
    print((pred.shape))
    
    num_points = pred.shape[2]
    
    for i in range(len(pred)) : 
        
        if num_points == 68 : 
            i_d = np.sqrt(np.square(pred[i,36] - gt[i,45]))
        else : 
            i_d = np.sqrt(np.square(pred[i,19] - gt[i,28]))
        
        sum = 0
        for j in range(num_points) : 
            sum += np.sqrt(np.square(pred[i,j]-gt[i,j]))
        
        err[i] = sum/(num_points * i_d)
        
    return err

def calcNormalizedDistance(pred,gt):
    '''
    input : 
        pred : 1,num points
        gt : 1, num points 
    '''
    
    num_points = pred.shape[0]
    #print(num_points)
    
    '''if num_points == 68*2 : 
        i_d = np.sqrt(np.square(pred[36] - gt[45]) + np.square(pred[36+68] - gt[45+68]))
    else : 
        i_d = np.sqrt(np.square(pred[19] - gt[28]) + np.square(pred[19+68] - gt[28+68]))
    '''
    if num_points == 68*2 : 
        i_d = np.sqrt(np.square(gt[36] - gt[45]) + np.square(gt[36+68] - gt[45+68]))
    else : 
        i_d = np.sqrt(np.square(gt[19] - gt[28]) + np.square(gt[19+68] - gt[28+68]))
    
    t_sum = 0
    num_points_norm = num_points//2
    
    for j in range(num_points_norm) : 
        t_sum += np.sqrt(np.square(pred[j]-gt[j])+np.square(pred[j+num_points_norm]-gt[j+num_points_norm]))
    
    err = t_sum/(num_points_norm * i_d)
        
    return err

#assumes p_a and p_b are both positive numbers that sum to 100
def myRand(a, p_a, b, p_b):
    return a if random.uniform(0,100) < p_a else b 


def calcLandmarkErrorListTF(pred,gt):
    
    all_err = []
    batch = pred.get_shape()[0]
    seq = pred.get_shape()[1]
    
    for i in range(batch) :
        for z in range(seq):  
            bb = get_bb_tf(gt[i,z,0:68],gt[i,z,68:])
            
            width = tf.abs(bb[2] - bb[0])
            height = tf.abs(bb[3] - bb[1])
            
            gt_bb = tf.sqrt(tf.square(width) + tf.square(height)) 
            
            num_points = pred.get_shape()[2]
            num_points_norm = num_points//2
            
            sum = []
            for j in range(num_points_norm) : 
                sum.append( tf.sqrt(tf.square(pred[i,z,j]-gt[i,z,j])+tf.square(pred[i,z,j+num_points_norm]-gt[i,z,j+num_points_norm])))
            
            err = tf.divide(tf.stack(sum),gt_bb*num_points_norm)
            
            all_err.append(err)
        
    return tf.reduce_mean(tf.stack(all_err))

def calcLandmarkError(pred,gt): #for 300VW
    '''
    input : 
        pred : 1,num points
        gt : 1, num points 
        
        according to IJCV
        Normalized by bounding boxes
    '''
    
    #print pred,gt
    
    
    num_points = pred.shape[0]
    
    num_points_norm = num_points//2
    
    bb = get_bb(gt[:68],gt[68:])
    
    #print(gt)
    #print(bb)
    
    '''width = np.abs(bb[2] - bb[0])
    height = np.abs(bb[3] - bb[1])
    
    gt_bb = np.sqrt(np.square(width) + np.square(height))
    
    
    print("1 : ",width,height,gt_bb)'''
    
    width = np.abs(bb[2] - bb[0])
    height = np.abs(bb[3] - bb[1])
    
    gt_bb = math.sqrt((width*width) +(height*height))
    
    #print("2 : ",width,height,(width^2) +(height^2),gt_bb)
    '''print(bb) 
    print(gt_bb)
    print("BB : ",gt)
    print("pred : ",pred)'''
    
    '''print(num_points_norm)
    print("BB : ",bb)
    print("GT : ",gt)
    print("PR : ",pred)'''
    #print(num_points)
    
    '''error = np.mean(np.sqrt(np.square(pred-gt)))/gt_bb
    return error''' 
    
    summ = 0
    for j in range(num_points_norm) : 
        #summ += np.sqrt(np.square(pred[j]-gt[j])+np.square(pred[j+num_points_norm]-gt[j+num_points_norm]))
        summ += math.sqrt(((pred[j]-gt[j])*(pred[j]-gt[j])) + ((pred[j+num_points_norm]-gt[j+num_points_norm])*(pred[j+num_points_norm]-gt[j+num_points_norm])))
    #err = summ/(num_points_norm * (gt_bb))
    err = summ/(num_points_norm*gt_bb)
    
        
    return err

def showGates(tg = None, batch_index_to_see = 0, n_to_see = 64, n_neurons = 1024,toShow = False, toSave = False, fileName = "gates.jpg"):
    #Total figure : 1024/64 data per image : 16 row per gate then *6 gate : 96
    
    
    t_f_row = n_neurons/n_to_see
    n_column = 6

    fig = plt.figure()
    
    for p_i in range(t_f_row) :
        
        inputGate =     tg[:,0,batch_index_to_see,p_i * n_to_see:p_i*n_to_see+n_to_see] #all sequence, gate 1, batch 0, 200 neurons
        newInputGate=   tg[:,1,batch_index_to_see,p_i * n_to_see:p_i*n_to_see+n_to_see] #all sequence, gate 2, batch 0, 200 neurons
        forgetGate =    tg[:,2,batch_index_to_see,p_i * n_to_see:p_i*n_to_see+n_to_see] #all sequence, gate 3, batch 0, 200 neurons
        outputGate =    tg[:,3,batch_index_to_see,p_i * n_to_see:p_i*n_to_see+n_to_see] #all sequence, gate 4, batch 0, 200 neurons
        cellState =     tg[:,4,batch_index_to_see,p_i * n_to_see:p_i*n_to_see+n_to_see]
        outputState =   tg[:,5,batch_index_to_see,p_i * n_to_see:p_i*n_to_see+n_to_see]
        
        #print p_i
        ax = fig.add_subplot(t_f_row,n_column,p_i*(n_column) + 1)
        if p_i == 0 : 
            ax.set_title('Input Gate')
        plt.imshow(inputGate,vmin=0,vmax=1)
        '''
        for temp in inputGate : 
            for temp2 in temp : 
                if temp2 < 0 : 
                    print temp2'''
        
        ax = fig.add_subplot(t_f_row,n_column,p_i*(n_column) + 2)
        if p_i == 0 : 
            ax.set_title('New Input Gate')
        plt.imshow(newInputGate,vmin=0,vmax=1)
        
        ax = fig.add_subplot(t_f_row,n_column,p_i*(n_column) + 3)
        if p_i == 0 : 
            ax.set_title('Forget Gate')
        plt.imshow(forgetGate,vmin=0,vmax=1)
        
        ax = fig.add_subplot(t_f_row,n_column,p_i*(n_column) + 4)
        if p_i == 0 : 
            ax.set_title('Output Gate')
        plt.imshow(outputGate,vmin=0,vmax=1)
        
        ax = fig.add_subplot(t_f_row,n_column,p_i*(n_column) + 5)
        if p_i == 0 : 
            ax.set_title('Cell State')
        plt.imshow(cellState,vmin=0,vmax=1)
        
        ax = fig.add_subplot(t_f_row,n_column,p_i*(n_column) + 6)
        if p_i == 0 : 
            ax.set_title('Output')
        plt.imshow(outputState,vmin=0,vmax=1)
        
        #plt.colorbar(orientation='vertical')
    if toShow : 
        plt.show()
    if toSave : 
        fig.savefig(fileName)

def get_list_heatmap(center,cov,image_size_x,image_size_y,percent_radius,exact_radius = None) :
    
    radius_x = int(image_size_x * percent_radius)
    radius_y = int(image_size_y * percent_radius)
    
    #print(radius_x,radius_y)
    
    l_cd = []
    
    t_radius_x = radius_x
    t_radius_y = radius_y
    
    if t_radius_x <= 0 : 
        t_radius_x = 1
    if t_radius_y <= 0 : 
        t_radius_y = 1 
        
    
    if exact_radius is not None : 
        t_radius_x = cov
        t_radius_y = cov
        
    #print(t_radius_x,t_radius_y,"radius")
    
    for x in range(center-t_radius_x,center+t_radius_x) :
        '''print((center-x)/t_radius_y)
        print(math.acos((center-x)/t_radius_y))    
        print(math.sin(math.acos((center-x)/t_radius_y)))'''
        
        yspan = t_radius_y*math.sin(math.acos(inBoundN((center-x)/t_radius_y,-1,1)));
        for y in range (int(center-yspan),int(center+yspan))  : 
            l_cd.append([x,y])
            
    l_cd = np.asarray(l_cd)
    
    mean = [center,center]
    
    if cov is None : 
        rv = multivariate_normal.pdf(l_cd,mean = mean, cov = [t_radius_x,t_radius_y])
    else :
        rv = multivariate_normal.pdf(l_cd,mean = mean, cov = [cov,cov])
        
    return l_cd,rv
    
def get_bb(x_list, y_list, length = 68,swap = False,adding = 0,adding_xmin=None, adding_xmax = None,adding_ymin = None, adding_ymax = None):
    #print x_list,y_list
    xMin = 999999;xMax = -9999999;yMin = 9999999;yMax = -99999999;
    
    for i in range(length): #x
        if xMin > x_list[i]: 
            xMin = int(x_list[i])
        if xMax < x_list[i]: 
            xMax = int(x_list[i])
    
        if yMin > y_list[i]: 
            yMin = int(y_list[i])
        if yMax < y_list[i]: 
            yMax = int(y_list[i])
    
    l_x = xMax - xMin
    l_y = yMax - yMin
    
    if swap : 
        return [xMin,xMax,yMin,yMax]
    else : 
        if adding_xmin is None: 
            return [xMin-adding*l_x,yMin-adding*l_y,xMax+adding*l_x,yMax+adding*l_y]
        else :
            return [xMin+adding_xmin*l_x,yMin+adding_ymin*l_y,xMax+adding_xmax*l_x,yMax+adding_ymax*l_y] 

def get_bb_tf(x_list, y_list, length = 68,adding = 0, axMin = None, axMax = None, ayMin = None, ayMax = None):
    #print x_list,y_list
    xMin = tf.constant(999999.0);xMax = tf.constant(-9999999.0);yMin = tf.constant(9999999.0);yMax = tf.constant(-99999999.0);
    
    for i in range(length): #x
        xMin = tf.minimum(x_list[i],xMin)
        xMax = tf.maximum(x_list[i],xMax)
        yMin = tf.minimum(y_list[i],yMin)
        yMax = tf.maximum(y_list[i],yMax)
    
    l_x = xMax - xMin
    l_y = yMax - yMin
    
    #adding ranging from 0 to 1
    if axMin is None : 
        return xMin-adding*l_x,yMin-adding*l_y,xMax+adding*l_x,yMax+adding*l_y
    else :  
        return xMin+axMin*l_x,yMin+ayMin*l_y,xMax+axMax*l_x,yMax+ayMax*l_y

def padding(image):

    def get_padding_size(image):
        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    return constant

def get_bb_face(seq_size=2,synthetic = False,path= "images/bb/"):
    
    list_gt = []
    list_labels = []
    list_labels_t = []
    
    for f in file_walker.walk(curDir +path):
        #print(f.name, f.full_path) # Name is without extension
        if f.isDirectory: # Check if object is directory
            for sub_f in f.walk():
                if sub_f.isFile:
                    if('txt' in sub_f.full_path): 
                        #print(sub_f.name, sub_f.full_path) #this is the groundtruth
                        list_labels_t.append(sub_f.full_path)
                if sub_f.isDirectory: # Check if object is directory
                    list_img = []
                    for sub_sub_f in sub_f.walk(): #this is the image
                        list_img.append(sub_sub_f.full_path)
                    list_gt.append(sorted(list_img))
    
    list_gt = sorted(list_gt)
    list_labels_t = sorted(list_labels_t)
    
    
    for lbl in list_labels_t : 
        
        with open(lbl) as file:
            x = [re.split(r',+',l.strip()) for l in file]
        y = [ list(map(int, i)) for i in x]
        list_labels.append(y)
    
    
    if seq_size is not None : 
        list_images = []
        list_ground_truth = []
        for i in range(0,len(list_gt)): 
            counter = 0
            for j in range(0,int(len(list_gt[i])/seq_size)):
                
                temp = []
                temp2 = []
                for z in range(counter,counter+seq_size):
                    temp.append(list_gt[i][z])
                    #temp2.append([list_labels[i][z][2],list_labels[i][z][3],list_labels[i][z][0],list_labels[i][z][1],list_labels[i][z][0]+list_labels[i][z][2],list_labels[i][z][1]+list_labels[i][z][3]])
                    if not synthetic : 
                        temp2.append([list_labels[i][z][0],list_labels[i][z][1],list_labels[i][z][0]+list_labels[i][z][2],list_labels[i][z][1]+list_labels[i][z][3]])
                    else : 
                        #temp2.append([list_labels[i][z][0],list_labels[i][z][1],list_labels[i][z][2],list_labels[i][z][3]])
                        temp2.append([list_labels[i][z][0],list_labels[i][z][1],list_labels[i][z][2],list_labels[i][z][3]])
                
                counter+=seq_size
                #print counter
                    
                list_images.append(temp) 
                list_ground_truth.append(temp2) 
    else : 
        list_images = []
        list_ground_truth = []
        for i in range(0,len(list_gt)): #per folder 
            temp = []
            temp2 = []
            for j in range(0,len(list_gt[i])):#per number of seq * number of data/seq_siz 
                
                temp.append(list_gt[i][j])
                #temp2.append([list_labels[i][z][2],list_labels[i][z][3],list_labels[i][z][0],list_labels[i][z][1],list_labels[i][z][0]+list_labels[i][z][2],list_labels[i][z][1]+list_labels[i][z][3]])
                if not synthetic : 
                    temp2.append([list_labels[i][j][0],list_labels[i][j][1],list_labels[i][j][0]+list_labels[i][j][2],list_labels[i][j][1]+list_labels[i][j][3]])
                else : 
                    #temp2.append([list_labels[i][z][0],list_labels[i][z][1],list_labels[i][z][2],list_labels[i][z][3]])
                    temp2.append([list_labels[i][j][0],list_labels[i][j][1],list_labels[i][j][2],list_labels[i][j][3]])
            
                    
            list_images.append(temp) 
            list_ground_truth.append(temp2)
        
    '''
    print len(list_images)
    print len(list_ground_truth)
            
    print (list_images[0])
    print (list_ground_truth[0])

    img = cv2.imread(list_images[0][0])
    
    cv2.rectangle(img,(list_ground_truth[0][0][2],list_ground_truth[0][0][3]),(list_ground_truth[0][0][4],list_ground_truth[0][0][5]),(255,0,255),1)
    cv2.imshow('jim',img)
    cv2.waitKey(0)
    '''
    return[list_images,list_ground_truth]#2d list of allsize, seqlength, (1 for image,6 for bb)


def makeGIF(files,filename):
    import imageio
    image = []
    for i in files :
        cv2_im = cv2.cvtColor(i,cv2.COLOR_BGR2RGB) 
        image.append(cv2_im)
        #pil_im = Image.fromarray(cv2_im)   
    #print np.asarray(image).shape
    imageio.mimsave(filename,image,'GIF')    

def get_kp_face(seq_size=None,data_list = ["300VW-Train"],per_folder = False,n_skip = 1,is3D = False,is84 = False):
        
    list_gt = []
    list_labels_t = []
    list_labels = []
    
    counter_image = 0
    
    i = 0
    
    if is84 :
        annot_name = 'annot84'
    elif is3D : 
        annot_name = 'annot2'
    else : 
        annot_name = 'annot' 
        
    for data in data_list : 
        print(("Opening "+data))
        
        for f in file_walker.walk(curDir + "images/"+data+"/"):
            if f.isDirectory: # Check if object is directory
                
                print((f.name, f.full_path)) # Name is without extension
                for sub_f in f.walk():
                    
                    if sub_f.isDirectory: # Check if object is directory
                        
                        list_dta = []
                        
                        #print sub_f.name
                        
                        for sub_sub_f in sub_f.walk(): #this is the data
                            if(".npy" not in sub_sub_f.full_path):
                                list_dta.append(sub_sub_f.full_path)
                        
                    
                        if(sub_f.name == annot_name) : #If that's annot, add to labels_t 
                            list_labels_t.append(sorted(list_dta))
                        elif(sub_f.name == 'img'): #Else it is the image
                            list_gt.append(sorted(list_dta))
                            counter_image+=len(list_dta)
    '''
    print len(list_gt[2])
    print len(list_labels_t[2])
    '''               
                       
    #print list_gt 
    #print list_labels_t     
    
    print("Now opening keylabels")
                        
    for lbl in list_labels_t : 
        #print lbl
        lbl_68 = [] #Per folder
        for lbl_sub in lbl : 
            
            print lbl_sub
            
            if ('pts' in lbl_sub) : 
                x = []
                
                with open(lbl_sub) as file:
                    data2 = [re.split(r'\t+',l.strip()) for l in file]
                
                #print data
                
                for i in range(len(data2)) :
                    if(i not in [0,1,2,len(data2)-1]):
                        x.append([ float(j) for j in data2[i][0].split()] )
                #y = [ list(map(int, i)) for i in x]
                
                #print len(x)
                lbl_68.append(x) #1 record
                
        list_labels.append(lbl_68)
        
    #print len(list_gt[2])           #dim  : numfolder, num_data
    #print len(list_labels[2])  #dim  : num_folder, num_data, 68
    
    list_images = []
    
    max_width = max_height = -9999
    min_width = min_height = 9999
    mean_width = mean_height = 0
    
    print(("Total data : "+str(counter_image)))
    
    print("Now partitioning data if required")
    
    if seq_size is not None : 
        
        list_ground_truth = np.zeros([int(counter_image/(seq_size*n_skip)),seq_size,136])
        indexer = 0;
        
        for i in range(0,len(list_gt)): #For each dataset
            counter = 0
            for j in range(0,int(len(list_gt[i])/(seq_size*n_skip))): #for number of data/batchsize
                
                temp = []
                temp2 = np.zeros([seq_size,136])
                i_temp = 0
                
                for z in range(counter,counter+(seq_size*n_skip),n_skip):#1 to seq_size 
                    temp.append(list_gt[i][z])
                    temp2[i_temp] = np.array(list_labels[i][z]).flatten('F')
                    i_temp+=1
                    
                list_images.append(temp)
                list_ground_truth[indexer] = temp2
                    
                indexer += 1
                counter+=seq_size*n_skip
                #print counter
    else : 
        if per_folder : #divide per folder
            print("Per folder")
            list_ground_truth = []
            
            indexer = 0;
            for i in range(0,len(list_gt)): #For each dataset
                temp = []
                temp2 = []
                
                '''print(len(list_gt[i]))
                print(list_gt[i][0])
                print(len(list_labels[i]))'''
                
                for j in range(0,len(list_gt[i]),n_skip): #for number of data/batchsize
                    #print len(list_gt[i])
                    #print len(list_labels[i])
                    #print(list_gt[i][j],list_labels[i][j])
                    temp.append(list_gt[i][j])
                    temp2.append(np.array(list_labels[i][j]).flatten('F'))
                    
                list_images.append(temp) 
                list_ground_truth.append(temp2)
            
        else : #make as one long list, for localisation
        
            if is84: 
                list_ground_truth = np.zeros([counter_image,168])
            else : 
                list_ground_truth = np.zeros([counter_image,136])
            indexer = 0;
            for i in range(0,len(list_gt)): #For each dataset
                for j in range(0,len(list_gt[i]),n_skip): #for number of data
                    
                    #print(("{}/{} {}/{}".format(i,len(list_gt),j,len(list_gt[i]))))
                    tmpImage = cv2.imread(list_gt[i][j])
                    '''height, width, channels = tmpImage.shape
                    
                    mean_width+=width;
                    mean_height+=height;
                    
                    if max_width<width : 
                        max_width = width
                    if max_height<height : 
                        max_height = height
                        
                    if min_width>width : 
                        min_width = width
                    if min_height>height : 
                        min_height = height'''
                        
                    list_images.append(list_gt[i][j])
                    #print(list_gt[i][j])
                    list_ground_truth[indexer] = np.array(list_labels[i][j]).flatten('F')
                        
                    indexer += 1
                    #print counter
            mean_width/= indexer
            mean_height/= indexer
        
        '''
        im_width = 240
        im_height = 180
        
        img = cv2.imread(list_images[500])
        height, width, channels = img.shape
        img = cv2.resize(img,(im_width,im_height))
        
        ratioWidth = truediv(im_width,width)
        ratioHeight = truediv(im_height,height)
        
        print ratioWidth,im_width,width
        print ratioHeight,im_height,height
        
        x_list = list_ground_truth[500,0:68] * ratioWidth
        y_list = list_ground_truth[500,68:136] * ratioHeight
        #getting the bounding box of x and y
        
        
        bb = get_bb(x_list,y_list)
        
        cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),(255,0,255),1)
        for i in range(68) :
            cv2.circle(img,(int(x_list[i]),int(y_list[i])),3,(0,0,255))
        
        cv2.imshow('jim',img)
        cv2.waitKey(0)'''
        
    return list_images,list_ground_truth,[mean_width,mean_height, min_width,max_width, min_height, max_height]


def get_kp_face_localize(seq_size=None,data = "300W/01_Indoor"):
        
    list_gt = []
    list_labels_t = []
    list_labels = []
    
    counter_image = 0
    
    i = 0
    
    print(("Opening "+data))
    
    for f in file_walker.walk(curDir + "images/"+data+"/"):
        print((f.name, f.full_path)) # Name is without extension
        if f.isDirectory: # Check if object is directory
            
            for sub_f in f.walk():
                
                if sub_f.isDirectory: # Check if object is directory
                    
                    list_dta = []
                    
                    #print sub_f.name
                    
                    for sub_sub_f in sub_f.walk(): #this is the data
                        list_dta.append(sub_sub_f.full_path)
                    
                    if(sub_f.name == 'annot') : #If that's annot, add to labels_t 
                        list_labels_t.append(sorted(list_dta))
                    elif(sub_f.name == 'img'): #Else it is the image
                        list_gt.append(sorted(list_dta))
                        counter_image+=len(list_dta)
    '''
    print len(list_gt[2])
    print len(list_labels_t[2])
    '''
    print("Now opening keylabels")
                        
    for lbl in list_labels_t : 
        #print lbl
        lbl_68 = [] #Per folder
        for lbl_sub in lbl : 
            
            #print lbl_sub
            
            if ('pts' in lbl_sub) : 
                x = []
                
                with open(lbl_sub) as file:
                    data = [re.split(r'\t+',l.strip()) for l in file]
                
                #print data
                
                for i in range(len(data)) :
                    if(i not in [0,1,2,len(data)-1]):
                        x.append([ float(j) for j in data[i][0].split()] )
                #y = [ list(map(int, i)) for i in x]
                
                #print len(x)
                lbl_68.append(x) #1 record
                
        list_labels.append(lbl_68)
        
    #print len(list_gt[2])           #dim  : numfolder, num_data
    #print len(list_labels[2])  #dim  : num_folder, num_data, 68
    
    list_images = []
    list_ground_truth = []
    
    max_width = max_height = -9999
    min_width = min_height = 9999
    mean_width = mean_height = 0
    
    print(("Total data : "+str(counter_image)))
    
    print("Now partitioning data if required")
    
        
    indexer = 0;
    if seq_size is None : 
        for i in range(0,len(list_gt)): #For each dataset
            
            temp = []
            temp2 = []
            
            for j in range(0,len(list_gt[i])): #for number of data/batchsize
                t_temp = []
                t_temp2 = []
                
                for k in range (2) : 
                    t_temp.append(list_gt[i][j])
                    t_temp2.append(np.array(list_labels[i][j]).flatten('F'))
                    
                temp.append(t_temp)
                temp2.append(t_temp2)
                
            list_images.append(temp) 
            list_ground_truth.append(temp2)
    else : 
        for i in range(0,len(list_gt)): #For each dataset
            
            for j in range(0,len(list_gt[i])): #for number of data/batchsize
                t_temp = []
                t_temp2 = []
                
                for k in range (2) : 
                    t_temp.append(list_gt[i][j])
                    t_temp2.append(np.array(list_labels[i][j]).flatten('F'))
                    
                list_images.append(t_temp)
                list_ground_truth.append(t_temp2)
                
        
    
    #num_all_data,2,image_size,image_size,3
    #num_all_data,2,136
    
        
        '''
    if seq_size is not None : 
        
        list_ground_truth = np.zeros([counter_image/(seq_size),seq_size,136])
        indexer = 0;
        
        for i in range(0,len(list_gt)): #For each dataset
            counter = 0
            for j in range(0,len(list_gt[i])): #for number of data/batchsize
                
                temp = []
                temp2 = np.zeros([seq_size,136])
                i_temp = 0
                
                for z in range(counter,counter):#1 to seq_size 
                    temp.append(list_gt[i][z])
                    temp2[i_temp] = np.array(list_labels[i][z]).flatten('F')
                    i_temp+=1
                    
                list_images.append(temp)
                list_ground_truth[indexer] = temp2
                    
                indexer += 1
                counter+=1
                #print counter
    else : 
        #divide per folder
        print("Per folder")
        list_ground_truth = []
        
        indexer = 0;
        for i in range(0,len(list_gt)): #For each dataset
            
            temp = []
            temp2 = []
            
            for j in range(0,len(list_gt[i])): #for number of data/batchsize
                t_temp = []
                t_temp2 = []
                
                for k in range (2) : 
                    t_temp.append(list_gt[i][j])
                    t_temp2.append(np.array(list_labels[i][j]).flatten('F'))
                    
                temp.append(t_temp)
                temp2.append(t_temp2)
                
            list_images.append(temp) 
            list_ground_truth.append(temp2)'''
        
        '''
        im_width = 240
        im_height = 180
        
        img = cv2.imread(list_images[500])
        height, width, channels = img.shape
        img = cv2.resize(img,(im_width,im_height))
        
        ratioWidth = truediv(im_width,width)
        ratioHeight = truediv(im_height,height)
        
        print ratioWidth,im_width,width
        print ratioHeight,im_height,height
        
        x_list = list_ground_truth[500,0:68] * ratioWidth
        y_list = list_ground_truth[500,68:136] * ratioHeight
        #getting the bounding box of x and y
        
        
        bb = get_bb(x_list,y_list)
        
        cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),(255,0,255),1)
        for i in range(68) :
            cv2.circle(img,(int(x_list[i]),int(y_list[i])),3,(0,0,255))
        
        cv2.imshow('jim',img)
        cv2.waitKey(0)'''
    
    #print list_images
    return list_images,list_ground_truth,[mean_width,mean_height, min_width,max_width, min_height, max_height]
    
#print get_bb_face(20)
#print len(get_kp_face(20)[0])
#print get_kp_face(None)[2]
#print makeErrTxt(np.array([0.02,0.3,0.01]))
#plot_results(1)

'''dirTrain = [
       '001','002','003','004','007','009','010','011','013','015',
       '016','017','018','019','020','022','025','027','028','029',
       '031','033','034','035','037','039','041','043','044','046',
       '047','048','049','053','057','059','112','113','115','119',
       '120','123','138','143','144','160','204','205','223','225'] 

dirCatA = [
        '114','124','125','126','150','158','401','402','505','506',
        '507','508','509','510','511','514','515','518','519','520',
        '521','522','524','525','537','538','540','541','546','547',
        '548'
    ]

dirCatB = [
        '203','208','211','212','213','214','218','224','403','404',
        '405','406','407','408','409','412','550','551','553'
    ]

dirCatC = [
        '410','411','516','517','526','528','529','530','531','533',
        '557','558','559','562'
    ]

for i in dirCatC : 
    print "cd ./"+i;
    print "mv *.jpg ./img"
    print "mv *.t7 ./annotOri"
    print "mv *.pts ./annot2"
    print "cd ../"'''