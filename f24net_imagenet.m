function net = f24net_imagenet(varargin)
% CNN_IMAGENET_INIT  Baseline CNN model

opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts = vl_argparse(opts, varargin) ;

net.layers = {} ;

% Define layers
net.layers = {} ;

% Block 1
net = add_block(net, opts, 1, 5, 5, 3, 64, 1, 0) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

% Block 2
net.layers{end+1} = struct('type', 'normalize', 'name', 'norm2', ...
                           'param', [9 1 0.0001/5 0.75]) ;
net = add_block(net, opts, 2, 5, 5, 64, 64, 1, 0) ;
net.layers{end+1} = struct('type', 'normalize', 'name', 'norm3', ...
                           'param', [9 1 0.0001/5 0.75]) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

% Block 3
net = add_block(net, opts, 3, 8, 8, 64,256, 1, 1) ;

% Block 4
net = add_block(net, opts, 4, 3, 3, 192, 384, 1, 1) ;

% Block 5
net = add_block(net, opts, 5, 3, 3, 192, 256, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

% Block 6
net = add_block(net, opts, 6, 6, 6, 256, 4096, 1, 0) ;
net.layers{end+1} = struct('type', 'dropout', 'name', 'dropout6', 'rate', 0.5) ;


% Block 7
net = add_block(net, opts, 7, 1, 1, 4096, 4096, 1, 0) ;
net.layers{end+1} = struct('type', 'dropout', 'name', 'dropout7', 'rate', 0.5) ;

% Block 8
net = add_block(net, opts, 8, 1, 1, 4096, 1000, 1, 0) ;
net.layers(end) = [] ;

% Block 9
net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;

function net = add_block(net, opts, id, h, w, in, out, stride, pad)
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
  name = 'fc' ;
else
  name = 'conv' ;
end
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%d', name, id), ...
                           'weights', {{0.01/opts.scale * randn(h, w, in, out, 'single'), []}}, ...
                           'stride', stride, ...
                           'pad', pad, ...
                           'learningRate', [1 2], ...
                           'weightDecay', [opts.weightDecay 0]) ;
net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%d',id)) ;
