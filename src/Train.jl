module Train

export unetTrain, prepareData, readData

using UNet
using Flux
using GeoArrays
using CSV
using DataFrames
using CUDA
using BSON: @save

include("Tiles.jl")

#TODO Not sure why I have to define it toplevel and not in unetTrain.
u = Unet(3, 1)
u = gpu(u)
opt = Momentum()

const tileSize = 512
const overlap = 32

function loss(x, y)
    op = clamp.(u(x), 0.001f0, 1.f0)
    mean(bce(op, y))
end

const TrainData = Vector{Tuple{Array{Float32, 4}, Array{Float32, 4}}}

function prepareData(outFp::String, gaVFp::String, gaHFp::String, gaBatFp::String, csvFp::String)::TrainData
  gaV = GeoArrays.read(gaVFp)
  gaH = GeoArrays.read(gaHFp)
  gaBat = GeoArrays.read(gaBatFp)
  gaBatPrim = Tiles.generateBatimetryTile(gaV, gaBat)
  csv = CSV.read(csvFp, DataFrame)
  tilesV = Tiles.generateTiles(tileSize, overlap, gaV)
  tilesH = Tiles.generateTiles(tileSize, overlap, gaH)
  tilesBat = Tiles.generateTiles(tileSize, overlap, gaBat)
  trainData = zip(tilesV, tilesH, tilesBat) .|>
    ((tV,tH, tBat),) -> begin
        img = Tiles.generateImageTile(tV, csv)
        ret = zeros(tileSize, tileSize, 3, 1)
        ret[:,:,1,:] = tV.A[:,:,1]
        ret[:,:,2,:] = tH.A[:,:,1]
        ret[:,:,3,:] = tBat.A[:,:,1]
        return (ret, reshape(img, tileSize, tileSize, 1,1))
      end
  bson(outFp, trainData)
end

function readData(fp::String)::TrainData
  BSON.load(fp)
end

function accuracy(data::TrainData)
  mean(data .|> ((x, y),) -> loss(x, y))
end

evalCallback(data) = Flux.throttle(30) do
  @show accuracy(data)
  @save "model-checkpoint.bson" model
end

function unetTrain(data::TrainData)
    gpu(data)
    Flux.train!(loss, Flux.params(u), data, opt, cb = evalCallback(data))
    return u
end

end
