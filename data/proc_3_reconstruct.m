load('params.mat')
load('remain.mat')
load('pointcnn_test.mat')
[~,tempIdx] = sort(index(:,:,2),2);
pointcnn_result=[];
for id=1:length(params)
    data=squeeze(val_data(id,:,:));
    x=data(:,1);x=(x+1)/2*params(id,4)+params(id,1);
    y=data(:,2);y=(y+1)/2*params(id,4)+params(id,2);
    z=data(:,3);z=(z+params(id,3))*params(id,4);    
    pred=double(pred_label(id,tempIdx(id,:)));    
    curr_result=[x,y,z,pred'];
    pointcnn_result=[pointcnn_result;curr_result];
end
remain(:,4)=0;
save2txt([pointcnn_result;remain],'pointcnn_test.txt',4)