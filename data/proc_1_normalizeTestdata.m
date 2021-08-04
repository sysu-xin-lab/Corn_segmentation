
inputFile='testdata_original.txt';
outputFile='testdata_nor.txt';
%% read data
data=importdata(inputFile);
%% z coordinate
z=data(:,3);
scale=max(z)-min(z); % train 3.02
data(:,3)=(z-min(z))./scale;
%% x,y coordinate
xoffset=1.2;x=data(:,1);
yoffset=0.8;y=data(:,2);
xstart=min(x)-0.01; xlabel=floor((x-xstart)/xoffset);
ystart=min(y)-0.01; ylabel=floor((y-ystart)/yoffset);
clear x y 
gridLabel=xlabel*max(ylabel+1)+ylabel;
labels=unique(gridLabel);
result=[];% points in each grid
params=[];% parameter of each grid
remain=[];% grid without enough points( num_points<2048 )
for i=1:length(labels)    
    index=labels(i)==gridLabel;
    currGrid=[data(index,:),gridLabel(index)];
    if length(currGrid)<2048
        currGrid(:,3)=currGrid(:,3)*scale;
        remain=[remain;currGrid];
        continue;
    end    
    x=currGrid(:,1);
    currGrid(:,1)=(x-min(x))./scale*2-1;
    y=currGrid(:,2);
    currGrid(:,2)=(y-min(y))./scale*2-1;
    currPara=[min(x),min(y),min(z),scale];
    result=[result;currGrid];
    params=[params;currPara];
end
save2txt(result,outputFile,4)
save params.mat params
save remain.mat remain
clear all