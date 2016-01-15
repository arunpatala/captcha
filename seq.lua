local seq = {}

local function repl(N)
    local net = nn.ConcatTable()
    for i=1,N do 
        net:add(nn.Identity())
    end
    return net
end

function seq.encoder(net,hsize)
    local enc = nn.Sequential()
    enc:add(net)
    enc:add(nn.Sequencer(nn.LSTM(hsize, hsize)))
    enc:add(nn.SelectTable(-1))
    return enc
end

function seq.decoder(hsize,csize)
    local dec = nn.Sequential()
           :add(nn.LSTM(hsize, hsize))
           :add(nn.Linear(hsize, csize))
           :add(nn.LogSoftMax())
    return dec
end

function seq.decoder(hsize,csize)
    local dec = nn.Sequential()
           :add(nn.LSTM(hsize, hsize))
           :add(nn.Linear(hsize, csize))
           :add(nn.LogSoftMax())
    return dec
end

function seq.seq(net,hsize,csize,osize)
    local enc = seq.enc(net,hsize)
    local dec = seq.dec(hsize,csize)
    local encdec = nn.Sequential()
                    :add(enc)
                    :add(repl(osize))
                    :add(dec)
    local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
    return encdec,criterion
end

return seq