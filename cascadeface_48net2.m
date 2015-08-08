function cascadeface_48net(varargin)
% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------

% Load character dataset
imdb = load('../../aflw/matlab/imdb48_v1.mat') ;
imdb = imdb.imdb;
imdb.meta.sets=['train','val'];
ss = size(imdb.images.label);
imdb.images.set = ones(1,ss(2));
imdb.images.set(ceil(rand(1,ceil(ss(2)/5))*ss(2))) = 2;
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------

net = f48net() ;

% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------

opts.train.batchSize = 256 ;
%opts.train.numSubBatches = 1 ;
opts.train.continue = false;
opts.train.gpus = 3 ;
%opts.train.prefetch = true ;
opts.train.sync = false ;
opts.train.errorFunction = 'binary' ;
opts.train.expDir = 'data/48net-v1.0-4-dropout/' ;
opts.train.learningRate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,5)]  ;
opts.train.weightDecay = 0.0001 ;
opts.train.numEpochs = numel(opts.train.learningRate) ;
[opts, ~] = vl_argparse(opts.train, varargin) ;

% Take the average image out
imageMean = mean(imdb.images.data(:)) ;
imdb.images.data = imdb.images.data - imageMean ;
net.imageMean = imageMean ;

global net48;
net48 = net;
global net24;
net24 = load('./data/24net-v2.0/f24net.mat');
global net12;
net12 = load('./data/12netv2-v2.0/f12net.mat');
%net12 = vl_simplenn_move(net12,'gpu');
%net24 = vl_simplenn_move(net24,'gpu');
% Call training function in MatConvNet
[net,info] = cnn_train(net, imdb, @getBatch,opts) ;

% Save the result for later use
net.layers(end) = [] ;
net = vl_simplenn_move(net,'cpu');
save(strcat(opts.expDir,'f48net.mat'), '-struct', 'net') ;

% -------------------------------------------------------------------------
% Part 4.4: visualize the learned filters
% -------------------------------------------------------------------------

figure(2) ; clf ; colormap gray ;
vl_imarraysc(squeeze(net.layers{1}.weights{1}),'spacing',2)
axis equal ; title('filters in the first layer') ;

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
im = 256 * reshape(im, 48, 48, 3, []) ;
labels = imdb.images.label(1,batch) ;
%im = gpuArray(im) ;