data = require 'data'
dir = '/home/arun/simple/'
X,Y = data.loadXY(dir)
Xt,Yt,Xv,Yv = data.split(X,Y,1000)

rutil = require 'rutil'
net = rutil.model()
rnn,ct = rutil.getNetCt(net)
rnn = rnn:cuda()
ct = ct:cuda()
tnet = nn.SplitTable(2,2):cuda()
batchSize = 16
rutil.train(rnn,ct,Xt,Yt,Xv,Yv,3,batchSize,tnet,0.001)
rutil.train(rnn,ct,Xt,Yt,Xv,Yv,15,batchSize,tnet,0.01)
rutil.train(rnn,ct,Xt,Yt,Xv,Yv,15,batchSize,tnet,0.001)
rutil.train(rnn,ct,Xt,Yt,Xv,Yv,15,batchSize,tnet,0.0001)
