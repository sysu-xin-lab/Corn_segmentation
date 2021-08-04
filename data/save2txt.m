function save2txt( data,filename,num )
%SAVE2TXT Summary of this function goes here
%   Detailed explanation goes here
fid=fopen(filename,'w');
if (~iscell(data))
fmt=[repmat(['%.' num2str(num) 'f '],[1,size(data,2)]) '\n'];
fprintf(fid,fmt,data');
%transpose([pts,nor_]));
else
    for i=1:length(data)
        linedata=data{i};
        fmt=[repmat(['%.' num2str(num) 'f '],[1,size(linedata,2)]) '\n'];
        fprintf(fid,fmt,linedata');
    end
end
fclose(fid);

end

