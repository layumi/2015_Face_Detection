function cascadeface_12c(varargin)
setup;
% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------

% Load character dataset
imdb = load('C:/Users/zhedong/Desktop/face-database/fddb&background_c.mat') ;
imdb = imdb.imdb;
imdb.images.data = imdb.images.data12;
s = size(imdb.images.data12);
imdb.images.set=ones(s(4),1);%set train&& valiation
imdb.images.set(round(rand(1,60000)*s(4)))=2;
imdb.meta.sets=['train','val'];
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------

net = f12net_c() ;

% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------

trainOpts.batchSize = 128 ;
trainOpts.numEpochs = 20 ;
trainOpts.continue = true ;
trainOpts.useGpu = false ;
trainOpts.learningRate = 0.003 ;
trainOpts.expDir = 'data/12netc-experiment' ;
trainOpts = vl_argparse(trainOpts, varargin);

% Take the average image out
imageMean = mean(imdb.images.data(:)) ;
imdb.images.data = imdb.images.data - imageMean ;

% Convert to a GPU array if needed
if trainOpts.useGpu
  imdb.images.data = gpuArray(imdb.images.data) ;
end

% Call training function in MatConvNet
[net,info] = cnn_train(net, imdb, @getBatch, trainOpts) ;

% Move the CNN back to the CPU if it was trained on the GPU
if trainOpts.useGpu
  net = vl_simplenn_move(net, 'cpu') ;
end

% Save the result for later use
net.layers(end) = [] ;
net.imageMean = imageMean ;
save('data/12netc-experiment/f12net_c.mat', '-struct', 'net') ;

% -------------------------------------------------------------------------
% Part 4.4: visualize the learned filters
% -------------------------------------------------------------------------

figure(2) ; clf ; colormap gray ;
vl_imarraysc(squeeze(net.layers{1}.filters),'spacing',2)
axis equal ; title('filters in the first layer') ;

% -------------------------------------------------------------------------
% Part 4.5: apply the model
% -------------------------------------------------------------------------

% Load the CNN learned before
net = load('data/12netc-experiment/f12net_c.mat') ;

% Load the sentence
im = imread('data/test2.png');
im = imresize(im,[12 12]);
im = im2single(im) ;
im = 256 * (im - net.imageMean) ;

% Apply the CNN to the larger image
res = vl_simplenn(net, im) ;
[value,index]=max(res(8).x);
disp(index);


% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
im = 256 * reshape(im, 12, 12, 3, []) ;
labels = imdb.images.label(1,batch) ;



