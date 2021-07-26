function read_bag(dirname)
testdir="..\data\"+dirname+"\";

FileStructs=dir(testdir+"*.bag");
Total_files=numel(FileStructs);

A=[];

%%
padlen=0;
for p=1:Total_files
    filename=testdir+FileStructs(p).name;
    bag=rosbag(filename);
    bagSelect=select(bag,'Topic','mavros/imu/data');
    msgs=readMessages(bagSelect,'DataFormat','Struct');
    MsgLen1=length(msgs);
    bagSelect=select(bag,'Topic','/mavros/imu/mag');
    msgs = readMessages(bagSelect, 'DataFormat', 'struct');
    MsgLen2=length(msgs);
    padlen=max(padlen,max(MsgLen1,MsgLen2));
    %     Insert Image Code HERE
end

%%
for p=1:Total_files
    filename=testdir+FileStructs(p).name;
    bag=rosbag(filename);
    bagSelect=select(bag,'Topic','mavros/imu/data');
    msgs=readMessages(bagSelect,'DataFormat','Struct');
    MsgLen=length(msgs);
    string = bag.MessageList.Topic;
    topic1 = '/pylon_camera_node/image_raw';
    topic2 = '/mavros/imu/data';
    [ID1,~] = find(string==topic1);
    timeStamps1 = bag.MessageList.Time(ID1);
    [ID2,~] = find(string==topic2);
    timeStamps2 = bag.MessageList.Time(ID2);
    if length(ID1)<=length(ID2)
        fs = ID1;
        ms = ID2;
    else
        fs = ID2;
        ms = ID1;
    end
    
    %     nj = [];
    imuimgexist=zeros(padlen,1);
    for i = 1:length(fs)
        temp = abs(min(min(fs),min(ms))-max(max(fs),max(ms)));
        for j = 1:length(ms)
            if abs(fs(i)-ms(j))<=temp
                
                temp = abs(fs(i)-ms(j));
                
                %             timestamp_value = ms(j);
                indx = j;
                
            else
                break
            end
        end
        imuimgexist(indx)=1;
        %         nj = [nj; indx];
    end
    for j=1:MsgLen
        IMU_orientation(j,1)=msgs{j}.Orientation.X;
        IMU_orientation(j,2)=msgs{j}.Orientation.Y;
        IMU_orientation(j,3)=msgs{j}.Orientation.Z;
        IMU_orientation(j,4)=msgs{j}.Orientation.W;
        Euler_Angles_IMU(j,:)=quat2eul(IMU_orientation(j,:));
        IMU_AngularVelocity(j,1)=msgs{j}.AngularVelocity.X;
        IMU_AngularVelocity(j,2)=msgs{j}.AngularVelocity.Y;
        IMU_AngularVelocity(j,3)=msgs{j}.AngularVelocity.Z;
        IMU_LinearAcc(j,1)=msgs{j}.LinearAcceleration.X;
        IMU_LinearAcc(j,2)=msgs{j}.LinearAcceleration.Y;
        IMU_LinearAcc(j,3)=msgs{j}.LinearAcceleration.Z;
    end
    
    topic1 = '/pylon_camera_node/image_raw';
    topic2 = '/mavros/imu/mag';
    bagSelect=select(bag,'Topic','/mavros/imu/mag');
    msgs = readMessages(bagSelect, 'DataFormat', 'struct');
    MsgLen=length(msgs);
    CovLen=length(msgs{1}.MagneticFieldCovariance);
    [ID1,~] = find(string==topic1);
    timeStamps1 = bag.MessageList.Time(ID1);
    [ID2,~] = find(string==topic2);
    timeStamps2 = bag.MessageList.Time(ID2);
    if length(ID1)<=length(ID2)
        fs = ID1;
        ms = ID2;
    else
        fs = ID2;
        ms = ID1;
    end
    %     nj = [];
    magimgexist=zeros(padlen,1);
    for i = 1:length(fs)
        temp = abs(min(min(fs),min(ms))-max(max(fs),max(ms)));
        for j = 1:length(ms)
            if abs(fs(i)-ms(j))<=temp
                
                temp = abs(fs(i)-ms(j));
                %             timestamp_value = ms(j);
                indx = j;
                
            else
                break
            end
        end
        %         nj = [nj; indx];
        magimgexist(indx)=1;
    end
    
    for k=1:MsgLen
        
        IMU_magField(k,1)=msgs{k}.MagneticField.X;
        IMU_magField(k,2)=msgs{k}.MagneticField.Y;
        IMU_magField(k,3)=msgs{k}.MagneticField.Z;
        i=i+1;
    end
    bagselect = select(bag, 'Topic', '/pylon_camera_node/image_raw');
    msgs = readMessages(bagselect, 'DataFormat', 'struct');
    imgdir=testdir+"img\"+num2str(p)+"\";
    mkdir(imgdir);
    for i = 1:length(msgs)
        a = reshape(msgs{i}.Data, [2048, 1536]);
        b = imresize(a, [500,500]);
        b = rot90(b);
        imwrite(b, imgdir+num2str(i)+".jpg");
        
    end
    
    imudata=[Euler_Angles_IMU IMU_AngularVelocity IMU_LinearAcc];
    for l=1:9
        %         x=singlefile(:,i);
        %         resampled=resample(x,50,length(x));
        %         resampled=[resampled resampled];
        %length(singlefile(:,l))
        %x(:,l)=resample(singlefile(:,l),50,length(singlefile(:,l)));
        x(:,l)=[imudata(:,l); zeros(padlen-length(imudata(:,l)),1)];
    end
    for l=1:3
        y(:,l)= [IMU_magField(:,l); zeros(padlen-length(IMU_magField(:,l)),1)];
    end
    A(p,:,:)=[x y imuimgexist magimgexist];
    
    
end

extention=".mat";
name=dirname+"_extracted";

%matname = fullfile(testdir, [name extention]);
matname=testdir+name+extention;

save(matname, 'A');
end
