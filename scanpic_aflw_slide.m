clear;
setup;
% Load the CNN learned before
global net12;
global net12_c;
global net24;
global net24_c;
global net48;
global net48_c;
net12 = load('data/12net-aflw-experiment/f12net.mat') ;
net12_c = load('data/12netc-experiment/f12net_c.mat') ;
net24 = load('data/24net-aflw-experiment/f24net.mat') ;
net24_c = load('data/24netc-experiment/f24net_c.mat') ;
net48 = load('data/48net-aflw-experiment-gpu/f48net.mat') ;
net48_c = load('data/48netc-aflw-experiment-gpu/f48net_c.mat') ;
% Load the sentence
origin_im = imread('data/find3.jpg');
[oh,ow,oc] = size(origin_im);
bias12 = 1;
bias24 = 3;
bias48 = 30;
%calibration 打表
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
for k=1:10
    ss = 12/(oh/10+oh*(k-1)/10);% oh/10<f<oh   
    im = imresize(origin_im,ss); 
    cim = imresize(origin_im,ss); 
    [h, w ,c] = size(im);
    im = im2single(im) ;
    im = 256 * (im - net12.imageMean) ;
    cim = 256 * (im - net12_c.imageMean) ;%%
    stride = ceil(4*ss);
    imbatch.im = single(zeros(12,12,3,(h-12+1)*(w-12+1)));
    imbatch.cim = single(zeros(12,12,3,(h-12+1)*(w-12+1)));
    %im2col是竖过来扫的
    imbatch.im(:,:,1,:) = reshape(im2col(im(:,:,1),[12 12],'sliding'),12,12,1,[]);
    imbatch.im(:,:,2,:) = reshape(im2col(im(:,:,2),[12 12],'sliding'),12,12,1,[]);
    imbatch.im(:,:,3,:) = reshape(im2col(im(:,:,3),[12 12],'sliding'),12,12,1,[]);
    imbatch.cim(:,:,1,:) = reshape(im2col(cim(:,:,1),[12 12],'sliding'),12,12,1,[]);
    imbatch.cim(:,:,2,:) = reshape(im2col(cim(:,:,2),[12 12],'sliding'),12,12,1,[]);
    imbatch.cim(:,:,3,:) = reshape(im2col(cim(:,:,3),[12 12],'sliding'),12,12,1,[]);
    if stride>1
       line_index = mod(1:(h-12+1),stride)==1;
       index = [line_index,zeros(1,(stride-1)*(h-12+1))];
       index = repmat(index,1,floor((w-12)/stride)+1);
       index = index(:,1:(h-12+1)*(w-12+1));
       index = index>0;
    else index = 1:(h-12+1)*(w-12+1);
    end
    imbatch.im = imbatch.im(:,:,:,index); 
    imbatch.cim = imbatch.cim(:,:,:,index);
    %location
    l2 = 1:stride:(h-12+1);
    local2 = repmat(l2,1,floor((w-12)/stride)+1)';
    l1 = 1:stride:(w-12+1);
    local1 = sort(repmat(l1,1,floor((h-12)/stride)+1))';
    imbatch.local = [local2,local1]';
    if( isempty(imbatch.im) ) 
        continue; 
    end;
    %12net
    res12 = vl_simplenn(net12,imbatch.im) ;
    res12_last = reshape(res12(end).x,2,[]);
    chosen = res12_last(1,:)>(res12_last(2,:) + bias12);
    value = res12_last(1,:)-res12_last(2,:);
    value = value(chosen);
    %12-net-c
    posbatch.im = imbatch.cim(:,:,:,chosen);%im for calibration
    posbatch.local = imbatch.local(:,chosen);
    if isempty(posbatch.im)
       continue;
    end
    res12c = vl_simplenn(net12_c, posbatch.im) ;
    res12c_last = reshape(res12c(end).x,45,[]);
    [~,index_c]=max(res12c_last,[],1);
    index_c = reshape(index_c,1,[]);
     xn = chang(1,index_c);
     yn = chang(2,index_c);
     sn = chang(3,index_c);
     posbatch.local(1,:) = (posbatch.local(1,:) - (xn.*12)./sn) ./ss; % 在压缩比是ss的图片上移动12*12的窗口
     posbatch.local(2,:) = (posbatch.local(2,:) - (yn.*12)./sn) ./ss;
     posbatch.width = (12./sn)./ss;
     posbatch.value = value;
     win12_tmp= [posbatch.local;posbatch.width;posbatch.value];
     win12 = [win12;win12_tmp'];
     boxes12_tmp = [posbatch.local(1,:);posbatch.local(2,:);posbatch.local(1,:)+posbatch.width;posbatch.local(2,:)+posbatch.width;posbatch.value];
     boxes12 = [boxes12;boxes12_tmp'];
end

%nms
pick = nms(boxes12,0.8);
win12 = win12(pick(:),:);
toc;
%-------------------------------24net-----------------------
disp('24net');
tic;
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

pick = nms(boxes24,0.5);
win24 = win24(pick(:),:);
toc;

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

%show
imshow(origin_im);
if isempty(win48)
   disp('noface');
else
 s = size(win48);
 x1 = round(win48(:,1));
 y1 = round(win48(:,2));
 w = win48(:,3);
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