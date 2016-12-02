clear;
addpath('../mksqlite');
dbpath = '../data/';
dbfile = 'aflw.sqlite';
imdbpath = '../data/flickr/';
%sql
mksqlite('open',fullfile(dbpath,dbfile));
fidQuery = 'SELECT face_id FROM Faces';
res = mksqlite(fidQuery);
face_num = size(res,1);
mksqlite('close');

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
        
imdb.images.data12=single([]);
imdb.images.data24=single([]);
imdb.images.data48=single([]);
imdb.images.label=[];
% get face
counter = 1;
pic_counter = 1;
for i=1:face_num
    face_id = res(i).face_id;
    facedata = getFaceDataFromSQLite([dbpath dbfile],face_id);
    rect = facedata.rect;
    try
      img = imread(strcat(imdbpath,facedata.image.filepath));
    catch
      continue;
    end
    img_color = size(img,3);
    if img_color==1
        img = repmat(img,[1,1,3]);
    end
    [oh,ow,~] = size(img);
    %3 scale
    for k = 1:45
        xn = chang(1,k);
        yn = chang(2,k);
        sn = chang(3,k);
        center_x = rect.x+rect.w/2;
        center_y = rect.y+rect.h/2;
        center_x = center_x-xn*rect.w/sn;
        center_y = center_y-yn*rect.h/sn;
        w = rect.w/sn;
        h = rect.h/sn;
        y1 = center_y - h/2;
        y2 = center_y + h/2;
        x1 = center_x - w/2;
        x2 = center_x + w/2;
        y1(y1<1)=1;
        y2(y2>oh)=oh;
        x1(x1<1)=1;
        x2(x2>ow)=ow;
        croppedimg = img(ceil(y1):floor(y2),ceil(x1):floor(x2),:);
        %if(k>3) 
         %   croppedimg = fliplr(croppedimg);
        %end
        imdb.images.data12(:,:,:,counter) = im2single(imresize(croppedimg,[12,12]));
        imdb.images.data24(:,:,:,counter) = im2single(imresize(croppedimg,[24,24]));
        imdb.images.data48(:,:,:,counter) = im2single(imresize(croppedimg,[48,48]));
        imdb.images.label(:,counter) = k;
        counter = counter+1;
        fprintf('selected_win:%d\n',counter);
        imshow(croppedimg);
    end
    fprintf('selected_pic:%d\n',pic_counter);
    pic_counter = pic_counter+1;
end

save('./positive_c.mat','imdb','-v7.3');