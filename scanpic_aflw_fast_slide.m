clear;
setup;
% Load the CNN learned before
global net12;
global net12_c;
global net24;
global net24_c;
global net48;
global net48_c;
net12 = load('data/12net-aflw-experiment-gpu-new/f12net.mat') ;
net12_c = load('data/12netc-experiment/f12net_c.mat') ;
net24 = load('data/24net-aflw-experiment-gpu-new/f24net.mat') ;
net24_c = load('data/24netc-experiment/f24net_c.mat') ;
net48 = load('data/48net-aflw-experiment-gpu/f48net.mat') ;
net48_c = load('data/48netc-aflw-experiment-gpu/f48net_c.mat') ;
% Load the sentence
origin_im = imread('data/find5.jpg');
[oh,ow,oc] = size(origin_im);
bias12 = 0.5;
bias24 = 0.5;
bias48 = 30;
%calibration ´ò±í
xn = [0,0,0];
yn = [0,0,0];
sn = [1,1,1,1,1];
%{
xn = [-0.17,0,0.17];
yn = [-0.17,0,0.17];
sn = [0.83,0.91,1.0,1.10,1.21];
%}
chang_count = 1;
for m = 1:5 %adverse
       for n = 1:3
             for k = 1:3
                 chang(:,chang_count)=[xn(k),yn(n),sn(m)];
                 chang_count = chang_count + 1;
             end
       end
end

%-------------------12net-------------------
disp('12net');
tic;
win12=[];
win24=[];
win48=[];
boxes12=[];
boxes24=[];
boxes48=[];
for k=1:20
    ss = 12/(oh/20+oh*(k-1)/20);% oh/10<f<oh   
    im = imresize(origin_im,ss); 
    [h, w ,c] = size(im);
    im = im2single(im) ;
    im = 256 * (im - net12.imageMean) ;
    cim = 256 * (im - net12_c.imageMean) ;
    %12net
    
    if(h<=12)
        break;
    end;
     if(w<=12)
        break;
    end;
    
    im1 = im;
    im2 = im; im2(:,1,:) = [];% delete first col
    im3 = im; im3(1,:,:) = [];% delete first row
    im4 = im; im4(1,:,:) = [];im4(:,1,:) = [];% delete first row and col
    res12_last1=[];
    res12_last2=[];
    res12_last3=[];
    res12_last4=[];
    if( ~isempty(im1) ) 
        res12_1 = vl_simplenn(net12,im1) ;
        %imshow(res12_1(end).x(:,:,1)); %heatmap
        res12_last1 = reshape(res12_1(end).x,[],2)'; 
    end;
    if( ~isempty(im2) ) 
        res12_2 = vl_simplenn(net12,im2) ;
        res12_last2 = res12_2(end).x;
        [x,y,~] = size(res12_last2);
        res12_last2(:,y,:)=[];
        res12_last2 = reshape(res12_last2,[],2)';
    end;
    if( ~isempty(im3) ) 
        res12_3 = vl_simplenn(net12,im3) ;
        res12_last3 = res12_3(end).x;
        [x,y,~] = size(res12_last3);
        res12_last3(x,:,:)=[];
        res12_last3 = reshape(res12_last3,[],2)';
    end;
    if( ~isempty(im4) ) 
        res12_4 = vl_simplenn(net12,im4) ;
        res12_last4 = res12_4(end).x;
        [x,y,~] = size(res12_last4);
        res12_last4(x,:,:)=[];
        res12_last4(:,y,:)=[];
        res12_last4 = reshape(res12_last4,[],2)';
    end;
    %local (1,1) (3,1) (5,1)...
    %im1
    l2 = 1:2:(w-12+1);%109 55
    local1_2 = sort(repmat(l2,1,floor((h-12)/2)+1));
    l1 = 1:2:(h-12+1);%181 91
    local1_1 = repmat(l1,1,floor((w-12)/2)+1);
    local_1 = [local1_1;local1_2];
    %im2
    l2 = 2:2:(w-12+1);%109
    local2_2 = sort(repmat(l2,1,floor((h-12)/2)+1));
    l1 = 1:2:(h-12+1);%181
    local2_1 = repmat(l1,1,floor((w-1-12)/2)+1);
    local_2 = [local2_1;local2_2];
    %im3
    l2 = 1:2:(w-12+1);%109
    local3_2 = sort(repmat(l2,1,floor((h-1-12)/2)+1));
    l1 = 2:2:(h-12+1);%181
    local3_1 = repmat(l1,1,floor((w-12)/2)+1);
    local_3 = [local3_1;local3_2];
    %im4
    l2 = 2:2:(w-12+1);%109
    local4_2 = sort(repmat(l2,1,floor((h-1-12)/2)+1));
    l1 = 2:2:(h-12+1);%181
    local4_1 = repmat(l1,1,floor((w-1-12)/2)+1);
    local_4 = [local4_1;local4_2];
    
    res12_last = [res12_last1,res12_last2,res12_last3,res12_last4];
    
    local = [local_1,local_2,local_3,local_4];
    chosen = res12_last(1,:)>(res12_last(2,:) + bias12);
    value = res12_last(1,:)-res12_last(2,:);
    value = value(chosen);
    local = local(:,chosen)./ss;
    width = repmat(12./ss,1,sum(chosen));
    win12_tmp= [local;width;value];
    win12 = [win12;win12_tmp'];
    boxes12_tmp = [local(1,:);local(2,:);local(1,:)+width;local(2,:)+width;value];
    boxes12 = [boxes12;boxes12_tmp'];
end

%nms
pick = nms(boxes12,0.8);
win12 = win12(pick(:),:);
toc;
%-------------------------------24net-----------------------
disp('24net');

count = 1;
s = size(win12);
x1 = round(win12(:,1));
y1 = round(win12(:,2));
w = win12(:,3);
x2 = round(x1 + w);
y2 = round(y1 + w);
x2(x2>oh) = oh;
y2(y2>ow) = ow;
x1(x1<1) = 1;
y1(y1<1) = 1; 
process24_im = im2single(origin_im) ;
process24_im = 256 * (process24_im - net24.imageMean) ;
process24_cim = 256 * (process24_im - net24_c.imageMean) ;
tic;
imbatch24.im = single(zeros(24,24,3,s(1)));
for i=1:s(1)
    win = process24_im(x1(i):x2(i),y1(i):y2(i),:);
    winc = process24_cim(x1(i):x2(i),y1(i):y2(i),:);
    if( isempty(win) ) 
        continue; 
    end;
    imbatch24.im(:,:,:,i) = imresize(win,[24 24]);
    imbatch24.cim(:,:,:,i) = imresize(winc,[24 24]);
    imbatch24.local(:,i)=[x1(i),y1(i)];
end
toc;
%24net
res24 = vl_simplenn(net24, imbatch24.im) ;
res24_last = reshape(res24(end).x,2,[]);
chosen = res24_last(1,:)>(res24_last(2,:) + bias24);
value = res24_last(1,:)-res24_last(2,:);
value = value(chosen);
w = w(chosen);
w = w';
%24-net-c
posbatch24.im = imbatch24.cim(:,:,:,chosen);
posbatch24.local = imbatch24.local(:,chosen);
res24c = vl_simplenn(net24_c, posbatch24.im) ;
res24c_last = reshape(res24c(end).x,45,[]);
[~,index_c]=max(res24c_last,[],1);
index_c = reshape(index_c,1,[]);
xn = chang(1,index_c);
yn = chang(2,index_c);
sn = chang(3,index_c);
posbatch24.local(1,:) = posbatch24.local(1,:) - (xn.*w)./sn; 
posbatch24.local(2,:) = posbatch24.local(2,:) - (yn.*w)./sn;
posbatch24.width = w./sn;
posbatch24.value = value;
win24 = [posbatch24.local;posbatch24.width;posbatch24.value];
win24 = win24';
boxes24 = [posbatch24.local(1,:);posbatch24.local(2,:);posbatch24.local(1,:)+posbatch24.width;posbatch24.local(2,:)+posbatch24.width;posbatch24.value];
boxes24 = boxes24';

pick = nms(boxes24,0.8);
win24 = win24(pick(:),:);

xn = [-0.17,0,0.17];
yn = [-0.17,0,0.17];
sn = [0.83,0.91,1.0,1.10,1.21];
%}

%------------------------------------48net-----------------
disp('48net');
tic;
count = 1;
s = size(win24);
ss = win24(:,4);
x1 = round(win24(:,1));
y1 = round(win24(:,2));
w = win24(:,3);
x2 = round(x1 + w);
y2 = round(y1 + w);
x2(x2>oh) = oh;
y2(y2>ow) = ow;
x1(x1<1) = 1;
y1(y1<1) = 1; 
process48_im = im2single(origin_im) ;
process48_im = 256 * (process48_im - net48.imageMean) ;
process48_cim = 256 * (process48_im - net48_c.imageMean) ;
for i=1:s(1)
    win = process48_im(x1(i):x2(i),y1(i):y2(i),:);
    winc = process48_cim(x1(i):x2(i),y1(i):y2(i),:);
    if( isempty(win) ) 
        continue; 
    end;
    imbatch48.im(:,:,:,i) = imresize(win,[48 48]);
    imbatch48.cim(:,:,:,i) = imresize(winc,[48 48]);
    imbatch48.local(:,i)=[x1(i),y1(i)];
end
%net48
res48 = vl_simplenn(net48, imbatch48.im) ;
res48_last = reshape(res48(end).x,2,[]);
chosen = res48_last(1,:)>(res48_last(2,:) + bias48);
value = res48_last(1,:)-res48_last(2,:);
value = value(chosen);
w = w(chosen);
w = w';
%48-net-c
posbatch48.im = imbatch48.cim(:,:,:,chosen);
posbatch48.local = imbatch48.local(:,chosen);
res48c = vl_simplenn(net48_c, posbatch48.im) ;
res48c_last = reshape(res48c(end).x,45,[]);
[~,index_c]=max(res48c_last,[],1);
index_c = reshape(index_c,1,[]);
xn = chang(1,index_c);
yn = chang(2,index_c);
sn = chang(3,index_c);
posbatch48.local(1,:) = posbatch48.local(1,:) - (xn.*w)./sn; 
posbatch48.local(2,:) = posbatch48.local(2,:) - (yn.*w)./sn;
posbatch48.width = w./sn;
posbatch48.value = value;
win48 = [posbatch48.local;posbatch48.width;posbatch48.value];
win48 = win48';
boxes48 = [posbatch48.local(1,:);posbatch48.local(2,:);posbatch48.local(1,:)+posbatch48.width;posbatch48.local(2,:)+posbatch48.width;posbatch48.value];
boxes48 = boxes48';
toc;
pick = nms(boxes48,0.2);
win48 = win48(pick(:),:);

%-----------------------------show
imshow(origin_im);
if isempty(win24)
   disp('noface');
else
 s = size(win24);
 x1 = round(win24(:,1));
 y1 = round(win24(:,2));
 w = win24(:,3);
 x2 = round(x1 + w);
 y2 = round(y1 + w);
 x2(x2>oh) = oh;
 y2(y2>ow) = ow;
 x1(x1<1) = 1;
 y1(y1<1) = 1; 
 for i=1:s(1)
    if ((x1(i)>x2(i)) || (y1(i)>y2(i)))
        continue;
    end
   rectangle('Position',[y1(i),x1(i),y2(i)-y1(i),x2(i)-x1(i)],'LineWidth',2,'EdgeColor','b');
 end
end