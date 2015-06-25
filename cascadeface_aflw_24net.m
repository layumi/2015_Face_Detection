function cascadeface_aflw_24net(varargin)

 setup;
% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------

% Load character dataset
imdb = load('C:/Users/zhedong/Desktop/face-database/train24net.mat') ;
imdb = imdb.imdb;
imdb.meta.sets=['train','val'];
ss = size(imdb.images.label);
imdb.images.set = ones(1,ss(2));
imdb.images.set(round(rand(1,60000)*ss(2))) = 2;
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------

net = f24net() ;

% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------

trainOpts.batchSize = 128 ;
trainOpts.numEpochs = 19 ;
trainOpts.continue = true ;
trainOpts.useGpu = false ;
trainOpts.errorType = 'binary';
trainOpts.learningRate = 0.003 ;
trainOpts.expDir = 'data/24net-experiment' ;
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
save('data/24net-experiment/f24net.mat', '-struct', 'net') ;

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
net = load('data/24net-experiment/f24net.mat') ;

% Load the sentence
im = imread('data/test3.png');
im = imresize(im,[24 24]);
im = im2single(im) ;
im = 256 * (im - net.imageMean) ;

% Apply the CNN to the larger image
res = vl_simplenn(net, im) ;
[value,index]=max(res(8).x);
if(index==2)
   disp('no face');
else
   disp('is face');
end

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
im = 256 * reshape(im, 24, 24, 3, []) ;
labels = imdb.images.label(1,batch) ;



