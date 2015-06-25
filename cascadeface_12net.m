function cascadeface_12net(varargin)
% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------

% Load character dataset
imdb = load('../../aflw/matlab/imdb12.mat') ;
imdb = imdb.imdb;
imdb.meta.sets=['train','val'];
ss = size(imdb.images.label);
imdb.images.set = ones(1,ss(2));
imdb.images.set(ceil(rand(1,90000)*ss(2))) = 2;
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------

net = f12net() ;

% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------

opts.train.batchSize = 128 ;
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.train.gpus = 3 ;
%opts.train.prefetch = true ;
opts.train.sync = false ;
opts.train.continue = false ;
opts.train.errorFunction = 'binary' ;
opts.train.expDir = 'data/12net-v1.0/' ;
opts.train.learningRate = [0.001*ones(1,50),0.0001*ones(1,20),0.00001*ones(1,5)] ;
opts.train.numEpochs = numel(opts.train.learningRate) ;
[opts, varargin] = vl_argparse(opts.train, varargin) ;

% Take the average image out
imageMean = mean(imdb.images.data(:)) ;
imdb.images.data = imdb.images.data - imageMean ;
% Call training function in MatConvNet
[net,info] = cnn_train(net, imdb, @getBatch,opts) ;

% Save the result for later use
net.layers(end) = [] ;
net.imageMean = imageMean ;
save(strcat(opts.train.expDir,'f12net.mat'), '-struct', 'net') ;

% -------------------------------------------------------------------------
% Part 4.4: visualize the learned filters
% -------------------------------------------------------------------------

figure(2) ; clf ; colormap gray ;
vl_imarraysc(squeeze(net.layers{1}.filters),'spacing',2)
axis equal ; title('filters in the first layer') ;

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
im = 256 * reshape(im, 12, 12, 3, []) ;
labels = imdb.images.label(1,batch) ;
im = gpuArray(im) ;