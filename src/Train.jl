module Train

export unetTrain, loadFile

using UNet
using Flux
using GeoArrays
using CSV
using DataFrames

include("Tiles.jl")

function loss(x, y)
    op = clamp.(u(x), 0.001f0, 1.f0)
    mean(bce(op, y))
end

const TrainData = Vector{Tuple{Matrix{Union{Missing,Float32}},Matrix{Float32},Int, Int}}

function loadFile(gaFp::String, csvFp)::TrainData
  ga = GeoArrays.read(gaFp)
  csv = CSV.read(csvFp, DataFrame)
  tiles = Tiles.generateTiles(2048, 64, ga)
  trainData = tiles .|> t -> begin
                            img = Tiles.generateImageTile(t, csv)
                            return (t.A[:,:,1], img, 1, 1)
                          end
  return trainData
end

function unetTrain(data::TrainData)
    u = Unet()
    u = gpu(u)
    opt = Momentum()
    Flux.train!(loss, Flux.params(u), data, opt)
end

end
