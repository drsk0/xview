module Train

export unetTrain, loadFile

using UNet
using Flux
using GeoArrays
using CSV
using DataFrames

include("Tiles.jl")

u = Unet()
u = gpu(u)
opt = Momentum()

const tileSize = 512
const overlap = 32

function loss(x, y)
    op = clamp.(u(x), 0.001f0, 1.f0)
    mean(bce(op, y))
end

const TrainData = Vector{Tuple{Array{Float32, 4}, Array{Float32, 4}}}

function loadFile(gaFp::String, csvFp::String)::TrainData
  ga = GeoArrays.read(gaFp)
  csv = CSV.read(csvFp, DataFrame)
  tiles = Tiles.generateTiles(tileSize, overlap, ga)
  trainData = tiles .|> t -> begin
                              img = Tiles.generateImageTile(t, csv)
                              # TODO replace missing values with min(Float32). Should we do something
                              # different?
                              t.A[t.A .|> ismissing] .= typemin(Float32)
                              return (reshape(t.A[:,:,1], tileSize, tileSize, 1, 1), reshape(img, tileSize, tileSize, 1,1))
                            end
  return trainData
end

evalcb = throttle(30) do
  @show(accuracy())
  @save "model-checkpoint.bson" model
end

function unetTrain(data::TrainData)
    Flux.train!(loss, Flux.params(u), data, opt, cb = evalcb)
end

end
