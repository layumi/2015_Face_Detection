clear;
setup;
% Load the CNN learned before
net12 = load('data/12net-experiment/f12net.mat') ;
net12_c = load('data/12netc-experiment/f12net_c.mat') ;
net24 = load('data/24net-experiment/f24net.mat') ;
net24_c = load('data/24netc-experiment/f24net_c.mat') ;
% Load the sentence
origin_im = imread('data/find.jpg');
win_count = 0;
[oh,ow,oc] = size(origin_im);
count=1;
%calibration 打表
xn = [-0.17,0,0.17];
yn = [-0.17,0,0.17];
sn = [0.83,0.91,1.0,1.10,1.21];
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
for k=1:39
    ss = 12/(oh/20+oh*(k-1)/40);% oh/20<f<oh   
    im = imresize(origin_im,ss); 
    [h, w ,c] = size(im);
    i=1; j=1;
   while (i+11<=h)
      while (j+11<=w)
         win_count = win_count+1;% count windows totally
         owin = im(i:i+11,j:j+11,:);
         win = im2single(owin) ;
         win = 256 * (win - net12.imageMean) ;
         res12 = vl_simplenn(net12, win) ;
         [value,index]=max(res12(8).x);
         if(index==1&&value>0)
             %12-net-c
             %imshow(owin)
             win = im2single(owin) ;
             win = 256 * (win - net12_c.imageMean) ;
            res_c = vl_simplenn(net12_c, win) ;
            [value,index]=max(res_c(8).x);
            xn = chang(1,index);
            yn = chang(2,index);
            sn = chang(3,index);
            ci = i - xn*12/sn; % 在压缩比是ss的图片上移动12*12的窗口
            cj = j - yn*12/sn;
            cw = 12/sn;
            win12(:,count)= [ci/ss,cj/ss,cw/ss,ss,value,1];
            count = count + 1;
         end
         j = j+4;
      end
      i = i+4;
      j = 1;
   end
end


%-------------------nms---------------------   
tic
win12 = sortrows(win12',-5); %%530 429 100430
win12 = win12';
toc
ttimes=1;%52853 46600
tic
s = size(win12);
for i=1:s(2)
    if(win12(6,i)==0 ) continue;end
    win1_true = zeros(oh,ow);
    x1 = round(win12(1,i));
    y1 = round(win12(2,i));
    w = win12(3,i);
    x2 = round(x1+w);
    y2 = round(y1+w);
    if(x2>oh) x2=oh;end
    if(y2>ow) y2=ow;end
    if(x1<1) x1=1;end
    if(y1<1) y1=1;end
    win1_true(x1:x2,y1:y2)= 1;
    length = i + s(2)/5;
    if(length>s(2)) length=s(2); end
    for j=i+1:length
       if(win12(6,j)==0 ) continue;end
       win2_true = zeros(oh,ow);
       x1 = round(win12(1,j));
       y1 = round(win12(2,j));
       w = win12(3,j);
       x2 = round(x1+w);
       y2 = round(y1+w);
       if(x2>oh) x2=oh;end
       if(y2>ow) y2=ow;end
       if(x1<1) x1=1;end
       if(y1<1) y1=1;end
       win2_true(x1:x2,y1:y2)= 1;
       overlap = win1_true&win2_true;
       maxwin = max(sum(sum(win1_true)),sum(sum(win2_true)));
       proportion = sum(sum(overlap))/maxwin;
       %if(proportion==0) % due to spacial cluster
        %  break; 
       %end
       ttimes=ttimes+1;
       if(proportion>0.8) 
          % if(win12(5,i)>win12(5,j))  %compare value
               win12(6,j)=0;
           %else win12(6,i)=0;
           %end
       end
    end
end
toc


%24net
count = 1;
imshow(origin_im);
for i=1:s(2)
    if(win12(6,i)==1)%nms剩下的有效窗口
        ss=win12(4,i);
        x1 = round(win12(1,i));
        y1 = round(win12(2,i));
        w = win12(3,i);
        x2 = round(x1 + w);
        y2 = round(y1 + w);
        if(x2>oh) x2 = oh;end;
        if(y2>ow) y2 = ow;end;
        if(x1<1) x1 = 1;end;
        if(y1<1) y1 = 1; end;
        win = origin_im(x1:x2,y1:y2,:);
        if( isempty(win) ) 
            continue; 
        end;
        % here we get window on original image x1 y1 win~
        save_win = win;
        win = imresize(win,[24 24]);
        win = im2single(win) ;
        win = 256 * (win - net24.imageMean) ;
        res24 = vl_simplenn(net24, win) ;
        [value,index]=max(res24(8).x);
        if(index==1&&value>0)%24net
            win = save_win;
            win = imresize(win,[24 24]);
            win = im2single(win) ;
            win = 256 * (win - net24_c.imageMean) ;
            res_c = vl_simplenn(net24_c, win) ;
            [value,index]=max(res_c(8).x);
            % here we get window on 24*24 prediction on(ss*2) image
            xn = chang(1,index);%change parameter
            yn = chang(2,index);%
            sn = chang(3,index);%
            ss = ss*2;%%%%this is important
            ci = x1*ss - xn*24/sn;
            cj = y1*ss - yn*24/sn;
            cw = 24/sn;
            win24(:,count)= [ci/ss,cj/ss,cw/ss,ss,value,1];
            count = count + 1;
       end
    end
end


%nms  before 530  after 504
s = size(win24);
for i=1:s(2)
    if(win24(6,i)==0 ) continue;end
    win1_true = zeros(oh,ow);
    ss=win24(4,i);
    x1 = round(win24(1,i));
    y1 = round(win24(2,i));
    w = win24(3,i);
    x2 = round(x1+w);
    y2 = round(y1+w);
    if(x2>oh) x2=oh;end
    if(y2>ow) y2=ow;end
    if(x1<1) x1=1;end
    if(y1<1) y1=1;end
    win1_true(x1:x2,y1:y2)= 1;
    for j=i+1:s(2)
       if(win24(6,j)==0 ) continue;end
       win2_true = zeros(oh,ow);
       ss=win24(4,j);
       x1 = round(win24(1,j));
       y1 = round(win24(2,j));
       w = win24(3,j);
       x2 = round(x1+w);
       y2 = round(y1+w);
       if(x2>oh) x2=oh;end
       if(y2>ow) y2=ow;end
       if(x1<1) x1=1;end
       if(y1<1) y1=1;end
       win2_true(x1:x2,y1:y2)= 1;
       overlap = win1_true&win2_true;
       maxwin = max(sum(sum(win1_true)),sum(sum(win2_true)));
       proportion = sum(sum(overlap))/maxwin;
       %if(proportion==0) % due to spacial cluster
        %  break; 
       %end
       if(proportion>0.8) 
           if(win24(5,i)>win24(5,j))  %compare value
               win24(6,j)=0;
           else win24(6,i)=0;
           end
       end
    end
end

%show
s = size(win24);
for i=1:s(2)
    if(win24(6,i)==1 )
        x1 = win24(1,i);
        y1 = win24(2,i);
        w = win24(3,i);
        x2 = round(x1 + w);
        y2 = round(y1 + w);
        if(x2>oh) x2 = oh;end;
        if(y2>ow) y2 = ow;end;
        if(x1<1) x1 = 1;end;
        if(y1<1) y1 = 1; end;
       rectangle('Position',[y1,x1,y2-y1,x2-x1],'LineWidth',2,'EdgeColor','b');
    end
end