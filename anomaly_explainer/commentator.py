import os
curdir=os.getcwd()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--retrain", help="train the model again with bag files in ./data/train/ directory",action="store_true")
args = parser.parse_args()

retrain=args.retrain

import numpy as np 
from scipy.io import loadmat, savemat
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm
import keras
from os import listdir
from PIL import Image
from os.path import join, isdir
from keras.layers import Conv2DTranspose, ConvLSTM2D, TimeDistributed, Conv2D, LSTM, Conv1D, Dense, Reshape, Bidirectional
from keras.models import Sequential, load_model
from keras_layer_normalization import LayerNormalization
from keras_self_attention import SeqSelfAttention
from keras import backend as K
import cv2
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

fea=['Pitch ','Roll ','Yaw ','w_x ','w_y ','w_z ','a_x','a_y ','a_z','m_x ','m_y ','m_z']
cmnt0=['Pitch= ','Roll= ','Yaw= ','w_x= ','w_y= ','w_z= ','a_x= ','a_y= ','a_z= ','m_x= ','m_y= ','m_z= ']
cmnt1=[': Normal ',': Slightly abnormal ',': Highly abnormal ']
cmnt2=['pitch',
        'roll',
        'yaw',
        'angular velocity',
        'angular velocity ',
        'angular velocity',
        'accelaration ',
        'accelaration ',
        'accelaration ',
        'magnetic field ',
        'magnetic field ',
        'magnetic field ']
     

ra = 500
ca = 1700
if not os.path.isdir(curdir+'/result/output_video'):
    os.mkdir(curdir+'/result/output_video')     
    


config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


plt.close('all')
nw=15

if not os.path.isdir(curdir+'/result'):
   os.mkdir(curdir+'/result')
   
matfile1=loadmat(curdir+'/data/train/train_extracted.mat')
matfile2=loadmat(curdir+'/data/test/test_extracted.mat')
matfile3=loadmat(curdir+'/anomaly_explainer/model/threshold.mat')

th1=matfile3['th1']
th2=matfile3['th2']

X_normal=matfile1['A'][:,:,:-2] 
X_abnormal=matfile2['A'][:,:,:-2] 
print(X_normal.shape)
print(X_abnormal.shape)

imgimuflag = matfile2['A'][:,:,-2]

timenum=X_normal.shape[1]

X_normal=shuffle(X_normal)
X_train = X_normal



############################################
imgdir_train=curdir+"/data/train/img"
imgdir_test=curdir+"/data/test/img"
img_model=curdir+"/anomaly_explainer/model/imgmodel.hdf5"
img_model_retrained=curdir+"/anomaly_explainer/model/imgmodel_retrained.hdf5"
par_model=curdir+"/anomaly_explainer/model/parmodel.hdf5"
par_model_retrained=curdir+"/anomaly_explainer/model/parmodel_retrained.hdf5"
############################################

class Config:
    BATCH_SIZE = 1
    EPOCHS = 30
    SZ = 7
    IMGDIM = 64
    

def get_clips_by_stride(stride, frames_list, sequence_size):
    clips = []
    sz = len(frames_list)
    clip = np.zeros(shape=(sequence_size, Config.IMGDIM, Config.IMGDIM, 1))
    cnt = 0
    for start in range(0, stride):
        for i in range(start, sz, stride):
            clip[cnt, :, :, 0] = frames_list[i]
            Image.fromarray(clip[cnt,:,:,0])
            cnt = cnt + 1
            if cnt == sequence_size:
                clips.append(clip)
                cnt = 0
    return clips



def get_training_set():
    clips = []
    for f in sorted(listdir(imgdir_train)):
        if isdir(join(imgdir_train, f)):
            all_frames = []
            for c in sorted(listdir(join(imgdir_train, f)), key = len):
                if str(join(join(imgdir_train, f), c))[-3:] == "jpg":
                    img = Image.open(join(join(imgdir_train, f), c)).resize((Config.IMGDIM, Config.IMGDIM))
                    img = np.array(img, dtype=np.float32) / 256.0
                    all_frames.append(img)
            for stride in range(1, 3):
                clips.extend(get_clips_by_stride(stride=stride, frames_list=all_frames, sequence_size=Config.SZ))
    return clips

def get_model(reload_model=True):
    if not reload_model:
        return load_model(img_model,custom_objects={'LayerNormalization': LayerNormalization, 'SeqSelfAttention': SeqSelfAttention})
    training_set = get_training_set()
    training_set = np.array(training_set)
    training_set = training_set.reshape(-1,Config.SZ,Config.IMGDIM,Config.IMGDIM,1)
    seq = Sequential()
    seq.add(TimeDistributed(Conv2D(128, (11, 11), strides=4, padding="same"), batch_input_shape=(None, Config.SZ, Config.IMGDIM, Config.IMGDIM, 1)))
    seq.add(LayerNormalization())
    seq.add(LayerNormalization())
    seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2DTranspose(128, (11, 11), strides=4, padding="same")))
    seq.add(TimeDistributed(Conv2D(1, (11, 11), activation="sigmoid", padding="same")))
    
    print(seq.summary())
    seq.compile(loss='mse', optimizer=keras.optimizers.Adam())
    imgHistory=seq.fit(training_set, training_set,
            batch_size=Config.BATCH_SIZE, 
            epochs=Config.EPOCHS, 
            shuffle=False)
    seq.save(img_model_retrained)
    return seq

def get_test():
    
    test = []
    testreal=[]
    cont = 0
    maxframes = 0
    for s in sorted(listdir(imgdir_test),key=len):
        cnt = 0 
        sz = len(listdir(imgdir_test+ '/' + s))
        
        tes = np.zeros(shape=(sz, Config.IMGDIM, Config.IMGDIM, 1))
        tesreal = np.zeros(shape=(sz, 500, 500))
        for f in sorted(listdir(imgdir_test+ '/' + s),key=len):
            if str(join(imgdir_test, f))[-3:] == "jpg":
                
                imgreal = Image.open(join(imgdir_test, s , f))
                img = imgreal.resize((Config.IMGDIM, Config.IMGDIM))
                
                img = np.array(img, dtype=np.float32) / 256.0
                imgreal = np.array(imgreal, dtype=np.float32) / 256.0
                tes[cnt, :, :, 0] = img
                tesreal[cnt, :, :] = imgreal
                cnt = cnt + 1
        maxframes = max(cnt, maxframes)
        test.append(tes)
        testreal.append(tesreal)
        cont = cont + 1
    return test, maxframes, testreal

n_slice=0

for i in range(0,timenum-nw,int(nw/2)):
    for j in range(X_train.shape[0]):
        if np.sum(X_train[j,int(i+nw/2):i+nw,:])!=0:
            n_slice=n_slice+1
        
Xfit=np.zeros((n_slice,nw,X_train.shape[2])).astype(np.float32)

n_slice=0

for i in range(0,timenum-nw,int(nw/2)):
    for j in range(X_train.shape[0]):
        if np.sum(X_train[j,int(i+nw/2):i+nw,:])!=0:
            Xfit[n_slice,:,:]= X_train[j,i:i+nw,:]
            n_slice=n_slice+1

X_ab=X_abnormal
X_inlier=Xfit
xmin, xmax = X_inlier.min(axis=(0,1)).reshape(1,1,X_train.shape[2]), X_inlier.max(axis=(0,1)).reshape(1,1,X_train.shape[2])
rng = (0, 1)
X_inlier = ((X_inlier - xmin) / (xmax - xmin)) * (rng[1] - rng[0]) + rng[0]
X_train2 = ((X_train - xmin) / (xmax - xmin)) * (rng[1] - rng[0]) + rng[0]
X_abnormal = ((X_abnormal - xmin) / (xmax - xmin)) * (rng[1] - rng[0]) + rng[0]



if not retrain:
    clf=load_model(par_model,custom_objects={'LayerNormalization': LayerNormalization, 'SeqSelfAttention': SeqSelfAttention})
else:
    
    clf = Sequential()

    clf.add(TimeDistributed(Dense(12), batch_input_shape=(None, X_inlier.shape[1], X_inlier.shape[2])))
    clf.add(LayerNormalization())
    clf.add(Conv1D(256, 3, strides=1, padding="same"))
    clf.add(LayerNormalization())
    clf.add(Bidirectional(LSTM(128, return_sequences=True)))
    clf.add(LayerNormalization())
    clf.add(Bidirectional(LSTM(64, return_sequences=True)))
    clf.add(LayerNormalization())
    clf.add(Bidirectional(LSTM(32, return_sequences=True)))
    clf.add(LayerNormalization())
    clf.add(Bidirectional(LSTM(64, return_sequences=True)))
    clf.add(LayerNormalization())
    clf.add(Bidirectional(LSTM(128, return_sequences=True)))
    clf.add(LayerNormalization())
    clf.add(Bidirectional(LSTM(256, return_sequences=True)))
    clf.add(LayerNormalization())
    clf.add(SeqSelfAttention(attention_width=32,attention_activation='sigmoid'))
    clf.add(LSTM(128, return_sequences=True))
    clf.add(LayerNormalization())
    clf.add(LSTM(64, return_sequences=True))
    clf.add(LayerNormalization())
    clf.add(LSTM(32, return_sequences=True))
    clf.add(LayerNormalization())
    clf.add(LSTM(64, return_sequences=True))
    clf.add(LayerNormalization())
    clf.add(LSTM(128, return_sequences=True))
    clf.add(LayerNormalization())
    clf.add(LSTM(256, return_sequences=True))
    clf.add(LayerNormalization())
    clf.add(SeqSelfAttention(attention_width=32,attention_activation='sigmoid'))
    clf.add(Conv1D(128, 3, strides=1, padding="same"))
    clf.add(LayerNormalization())
    clf.add(TimeDistributed(Dense(12))) 
    clf.add(LayerNormalization())
    print(clf.summary())
    clf.compile(loss='mse', optimizer=keras.optimizers.Adam())
    senHistory=clf.fit(X_inlier, X_inlier,
            batch_size=200, 
            epochs=750, 
            shuffle=False)
    
    clf.save(par_model_retrained)


testpreds = np.zeros(X_abnormal.shape).astype(np.float32)
testoutlier = np.zeros(X_abnormal.shape)
p = np.zeros(X_abnormal.shape).astype(np.float32)
s1=0
s2=0
model = get_model(retrain)
print("got model")
test, maxframes, testreal = get_test()

plt.figure(figsize=(20,20))

clip = np.zeros((Config.SZ, Config.IMGDIM, Config.IMGDIM, 1)).astype(np.float32)
rec_clip = np.zeros((Config.SZ, Config.IMGDIM, Config.IMGDIM, 1)).astype(np.float32)

vidpreds = np.zeros((X_abnormal.shape[0],maxframes)).astype(np.float32)

aa=np.ones((ra,ca))
if not os.path.isdir(curdir+'/result/scaled_abnormality_score'):
    os.mkdir(curdir+'/result/scaled_abnormality_score')

print('---------Predicting in Real-Time-------')

for k in range(X_abnormal.shape[0]):
    sca=np.empty((X_abnormal.shape[1],10))
    sca[:]=np.NaN
    fnum=sum(imgimuflag[k,0:nw-1]).astype(np.uint8)
    imflag=0
    m=np.array(X_ab[k])
    iflag=np.array(imgimuflag[k])
    rec_dir = curdir+'/result/img/'+str(k)
    maxkk=int(np.sum(imgimuflag[k]))
    kk =int(np.sum(imgimuflag[k,:nw])-1)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(curdir+'/result/output_video/outputFinal_'+str(k)+'.mp4',fourcc,5, (ca,ra),0) 
    for i in tqdm(range(nw,X_abnormal.shape[1])):
        n=np.array(m[i])
        test_pred=clf.predict(X_abnormal[k,i-nw:i,:].reshape((1,X_inlier.shape[1],X_inlier.shape[2])))
        p[k,i,:]=test_pred[0,nw-1,:]
        testpreds[k,i,:]=(X_abnormal[k,i,:]-test_pred[0,nw-1,:])**2
        q_s=testpreds[k,i,:]
        cmnt=''
        bb = []
        sl_abnor=0
        hg_abnor=0
        for ii in range(9): # 12 in place of 9
            if (q_s[ii] <= th1[0,ii]):
                s=0
            elif((q_s[ii]>th1[0,ii]) & (q_s[ii]<=th2[0,ii])):
                s=1
                sl_abnor=sl_abnor+1
            else:
                s=2
                hg_abnor=hg_abnor+1
                
            sc=min(1,q_s[ii]/th2[0,ii])
            
            sca[i,ii]=sc
                
            bb.append(cmnt0[ii]+np.array2string(round(n[ii],3))+' A S: '+str(round(sc,2))+cmnt1[s]+cmnt2[ii])
        
        bb.append('No Image at This Timestamp')
        
        if iflag[i] :
            img1 = testreal[k][kk-1,:,:]
            
            try:
                img2 = testreal[k][kk,:,:]
            except:
                img2 = np.ones((500,500))
            kk=kk+1
            aa[:, :500] = img1
            cv2.putText(aa[:, :500],'Latest Frame',(150,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2,(255,255,255))
            aa[:, -500:] = img2
            cv2.putText(aa[:, -500:],'Upcoming Frame',(190,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2,(255,255,255))
            
            if fnum>=Config.SZ:
                clip = test[k][(fnum-Config.SZ):fnum,:,:,:]
                rec_clip = model.predict(clip.reshape((1,Config.SZ, Config.IMGDIM, Config.IMGDIM, 1)))
                vidpreds[k,fnum]=np.sum((clip[-1]-rec_clip[0,-1])**2)
                #rec_img = rec_clip[0,-1,:,:,0]
                if (vidpreds[k,fnum] <= th1[0,-1]):
                    s=0
                elif((vidpreds[k,fnum]>th1[0,-1]) & (vidpreds[k,fnum]<=th2[0,-1])):
                    s=1
                else:
                    s=2
                    
                sca[i,9]=min(1,vidpreds[k,fnum]/th2[0,-1])    
                bb[-1]='Latest Image A.S:'+str(round(sca[i,9],2))+' '+cmnt1[s]
                
                
            fnum=fnum+1
        
        if ((hg_abnor>2) |(sl_abnor>4)):
            bb.append('System Status: Highly Abnormal')
        elif(((hg_abnor>=1)&(hg_abnor<=2))|((sl_abnor>=2)&(sl_abnor<=4))):
            bb.append('System Status: Slightly Abnormal')
        else:
            bb.append('System Status: Normal')
            
        for tex in range(len(bb)): # 12 in place of 9
            cv2.putText(aa[:, 500:-500],bb[tex],(5,tex*35+60), cv2.FONT_HERSHEY_COMPLEX_SMALL, .8,255)
            
        cv2.putText(aa,'Current Sensor Data',(740,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,255)
        cv2.imshow('BUET_ENDGAME: SP CUP 2020', (aa*255).astype(np.uint8))
        
        out.write((aa*255).astype(np.uint8))
        
        if kk == maxkk-1:
            aa[:, -500:] = np.ones((500,500))
            cv2.putText(aa[:, -500:],'No Image',(190,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2,(255,255,255))
        aa[:, 500:-500] = np.ones((500,ca-1000))
        if cv2.waitKey(27) == ord('q'):
                break
    np.savetxt(curdir+'/result/scaled_abnormality_score/'+'file_'+str(k)+'.csv',sca,fmt='%1.2f',delimiter=',')
    out.release()
    

cv2.destroyAllWindows()

print('-------Prediction Accomplished---------')
print('Creating Graphs for Better Understanding !!')

if not os.path.isdir(curdir+'/result/reconstructed_signals'):
    os.mkdir(curdir+'/result/reconstructed_signals')

k=np.linspace(1,X_abnormal.shape[1],X_abnormal.shape[1])
for i in range(X_abnormal.shape[0]):
    fig=plt.figure(figsize=(50,50))
    fig.suptitle('Reconstructed Signal for file no: '+str(i),fontsize=32,fontweight='bold')
    for j in range(9):
        fig.add_subplot(3,3,j+1).set_title(fea[j],fontsize=18,fontweight='bold')
        plt.subplots_adjust(hspace=.3)        
        plt.plot(k,p[i,:,j],label='Reconstructed',linewidth=2)
        plt.plot(np.linspace(1,X_abnormal.shape[1],X_abnormal.shape[1]),X_abnormal[i,:,j],label='Test')
        plt.plot(np.linspace(1,X_train2.shape[1],X_train2.shape[1]),X_train2[i,:,j],label='Normal')
        plt.legend(fontsize='x-large',loc='upper right')
    figName=curdir+'/result/reconstructed_signals/'+'reconstructed_signal_'+str(i)+'.png'
    plt.savefig(figName)

if not os.path.isdir(curdir+'/result/abnormality_score_&_thresholding'):
    os.mkdir(curdir+'/result/abnormality_score_&_thresholding')
    
for i in range(9):
    fig=plt.figure(figsize=(50,50))
    fig.suptitle('Abnormality Score & Tresholding of feature '+str(i),fontsize=32,fontweight='bold')
    pp=testpreds[:,:,i]
    for j in range(testpreds.shape[0]):
        qq=pp[j]
        plt.subplot(testpreds.shape[0],1,j+1)
        plt.plot(qq,linewidth=2,label='Abnormality Score')
        plt.axhline(y=th1[0,i], color='g', linestyle='-',linewidth=2,label='Threshold 1')
        plt.axhline(y=th2[0,i], color='r', linestyle='-',linewidth=2,label='Threshold 2')
        plt.legend(fontsize='x-large',loc='upper right')
    figName=curdir+'/result/abnormality_score_&_thresholding/'+'feature_'+str(i)+'.png'
    plt.savefig(figName)

fig=plt.figure(figsize=(50,50))
fig.suptitle('Abnormality Score & Tresholding of Image ',fontsize=32,fontweight='bold')

for i in range(vidpreds.shape[0]):
    fig.add_subplot(vidpreds.shape[0],1,i+1)
    plt.plot(vidpreds[i],linewidth=2,label='Abnormality Score')
    plt.axhline(y=th1[0,-1], color='g', linestyle='-',linewidth=2,label='Threshold 1')
    plt.axhline(y=th2[0,-1], color='r', linestyle='-',linewidth=2,label='Threshold 2')
    plt.legend(fontsize='x-large',loc='upper right')
    figName=curdir+'/result/abnormality_score_&_thresholding/'+'Image'+'.png'
    plt.savefig(figName)
    