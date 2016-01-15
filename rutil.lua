require 'rnn';
require 'nn';
require 'cunn';
require 'MultiCrossEntropyCriterion'

local rutil = {}


function rutil.getNetCt(net,bs,rho,hs,cls)

    local batchSize = bs or 16
    local rho = rho or 1
    local hiddenSize = hs or 36 
    local classes = cls or 36

    local mlp = nn.Sequential()
       :add(nn.Recurrent(
           hiddenSize, nn.Identity(), 
           nn:Sequential():add(nn.Linear(hiddenSize, hiddenSize)):add(nn.ReLU()), nn.ReLU(), 
           rho
        ))
       :add(nn.Linear(hiddenSize, classes))
       :add(nn.LogSoftMax())


    local rnn = nn.Sequential():add(net):add(nn.Sequencer(mlp))
    local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
    return rnn,criterion
end

function rutil.kfacc(outputs,targets)
    local Y,y = nil,nil;
    local N = outputs[1]:size(1)
    local C = outputs[1]:size(2)
    for k=1,#outputs do 
        Y = Y and torch.cat(Y,outputs[k]:reshape(N,1,C),2) or outputs[k]:reshape(N,1,C)
        y = y and torch.cat(y,targets[k]:reshape(N,1),2) or targets[k]:reshape(N,1)
    end
    local t,idx = Y:max(3)
    return idx:squeeze():eq(y):sum(2):eq(#outputs):sum()
end


function rutil.facc(outputs,targets)
    local acc = 0
    local acci = {}
    for k=1,#outputs do 
        local t,idx = outputs[k]:max(2)
        local ai = targets[k]:eq(idx:squeeze()):sum()
        acc = acc + ai
        table.insert(acci,ai)
    end
    return acc,acci
end

function rutil.valid(rnn,Xv,Yv,batchSize,tnet,f)
    local batchSize = bs or 16
    local acc = 0
    local acci = {}
    local Nv = Xv:size(1)
    rnn:evaluate()
    for i=1,Nv,batchSize do 
        xlua.progress(i/batchSize, Nv/batchSize)
        local j = math.min(Nv,i+batchSize-1)
        local Xb = Xv[{{i,j}}]:cuda()            
        local Yb = Yv[{{i,j}}]:cuda()
        local inputs = Xb
        local targets = tnet:forward(Yb)
        local outputs = rnn:forward(inputs)
        local aa,ai = f(outputs,targets) or rutil.facc(outputs,targets)
        for k=1,#ai do
            acci[k] = (acci[k] or 0) + ai[k]
        end
        acc = acc + aa/#ai
        rnn:forget()
    end
    for k=1,#acci do
            acci[k] = (acci[k] or 0) * 100/(Nv)
    end
    return (acc*100)/Nv,acci
end


function rutil.train(rnn,criterion,Xt,Yt,Xv,Yv,T,batchSize,tnet,lr)
    local batchSize = batchSize or 16
    local maxv = 0 or maxv
    local T = T or 2
    for t = 1,T do
        print(os.date("%X", os.time()))
        print(t,T)
        rnn:forget()
        local loss = 0
        local acc = 0
        local Nt = Xt:size(1)
        rnn:training()
        for i=1,Nt,batchSize do 
            xlua.progress(i/batchSize, Nt/batchSize)
            local j = math.min(Nt,i+batchSize-1)
            local Xb = Xt[{{i,j}}]:cuda()            
            local Yb = Yt[{{i,j}}]:cuda()
            local inputs = Xb
            local targets = tnet:forward(Yb)
            local outputs = rnn:forward(inputs)
            loss = loss + criterion:forward(outputs, targets)/#targets
            local gradOutputs = criterion:backward(outputs,targets)
            acc = acc + rutil.facc(outputs,targets)/#targets
            rnn:backward(inputs,gradOutputs)
            rnn:backwardThroughTime()
            rnn:updateParameters(lr or 0.001)
            rnn:zeroGradParameters()
            rnn:forget()
        end
        print('loss',loss)
        print('train',100*acc/Nt)
        local v,acc = rutil.valid(rnn,Xv,Yv,batchSize,tnet)
        if(v>maxv) then maxv = v end
        print('v',v,'maxv',maxv)    
        print(acc)
        print(os.date("%X", os.time()))
    end
    print(maxv)
end



function rutil.model()

    local k = k or 5
    local c = c or 36
    vgg = nn.Sequential()
    vgg:add(nn.Reshape(1,50,200))
    local function ConvBNReLU(nInputPlane, nOutputPlane)
      vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
      vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
      vgg:add(nn.ReLU(true))
      return vgg
    end
    ConvBNReLU(1,64)--:add(nn.Dropout(0.3,nil,true))
    ConvBNReLU(64,64)
    ConvBNReLU(64,64)
    vgg:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
    ConvBNReLU(64,128)--:add(nn.Dropout(0.4,nil,true))
    ConvBNReLU(128,128)--:add(nn.Dropout(0.4,nil,true))
    vgg:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
    ConvBNReLU(128,256)--:add(nn.Dropout(0.4,nil,true))
    ConvBNReLU(256,256)--:add(nn.Dropout(0.4,nil,true))
    vgg:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
    ConvBNReLU(256,256)--:add(nn.Dropout(0.4,nil,true))
    ConvBNReLU(256,256)--:add(nn.Dropout(0.4,nil,true))
    vgg:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
    ConvBNReLU(256,256)--:add(nn.Dropout(0.4,nil,true))
    ConvBNReLU(256,256)--:add(nn.Dropout(0.4,nil,true))
    ConvBNReLU(256,256)--:add(nn.Dropout(0.4,nil,true))
    vgg:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
    vgg:add(nn.View(256*2*7))

    local classifier = nn.Sequential()
    --classifier:add(nn.Dropout(0.5,nil,true))
    classifier:add(nn.Linear(256*2*7,256))
    classifier:add(nn.BatchNormalization(256))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Linear(256,k*c))
    vgg:add(classifier)
    vgg:add(nn.Reshape(5,36))
    vgg:add(nn.SplitTable(2,3))
    return vgg
end


return rutil