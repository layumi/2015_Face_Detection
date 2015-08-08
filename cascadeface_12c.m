function cascadeface_12c(varargin)
setup;
% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------

% Load character dataset
imdb = load('../../aflw/positive_c.mat') ;
imdb = imdb.imdb;
imdb.meta.sets=['train','val'];
ss = size(imdb.images.label);
imdb.images.set = ones(1,ss(2));
imdb.images.set(ceil(rand(1,ceil(ss(2)/5))*ss(2))) = 2;
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------

net = f12net_c() ;

% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------

opts.train.batchSize = 256 ;
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.train.gpus = 3 ;
%opts.train.prefetch = true ;
opts.train.sync = false ;
opts.train.errorFunction = 'multiclass' ;
opts.train.expDir = 'data/12net-cc-v1-v1.0-dropout0.5-pad2/' ;
opts.train.learningRate = [0.05*ones(1,25),0.005*ones(1,15),0.0005*ones(1,7)] ;
opts.train.numEpochs = numel(opts.train.learningRate) ;
[opts, varargin] = vl_argparse(opts.train, varargin) ;

% Take the average image out
imageMean = mean(imdb.images.data12(:)) ;
imdb.images.data12 = imdb.images.data12 - imageMean ;

% Call training function in MatConvNet
[net,info] = cnn_train(net, imdb, @getBatch, opts) ;

% Save the result for later use
net.layers(end) = [] ;
net.imageMean = imageMean ;
save(strcat(opts.expDir,'f12netc.mat'), '-struct', 'net') ;

% -------------------------------------------------------------------------
% Part 4.4: visualize the learned filters
% -------------------------------------------------------------------------

figure(2) ; clf ; colormap gray ;
vl_imarraysc(squeeze(net.layers{1}.weights{1}),'spacing',2)
axis equal ; title('filters in the first layer') ;


% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data12(:,:,:,batch) ;
im = 256 * reshape(im, 12, 12, 3, []) ;
labels = imdb.images.label(1,batch) ;



