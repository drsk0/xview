module TrainUnet

using Flux
using CSV
using JLD2: save_object, load_object
using Base.Iterators
using Rasters
import Flux.Losses as L
using LoopVectorization
using Statistics: mean
using ..UNet
using ..Utils
import MLUtils as MLU

function loss(u, x, y)
    y_head = sigmoid.(u(x))
    (xdim, ydim) = size(y)
    muS = xdim * ydim
    int_falpha = sum(y)
    int_fbeta = muS - int_falpha
    alpha = int_falpha > 0.0 ? muS / (2 * int_falpha) : 1.0
    beta = int_fbeta > 0.0 ? (alpha * int_falpha) / int_fbeta : 1.0
    ws = alpha * y .+ beta * (1.0 .- y)
    return L.binarycrossentropy(y_head, y; agg = xs -> mean(ws .* xs))
end

# we keep track of the evolution of the accuracy
accuracy_history = Vector{Float32}(undef, 0)
function accuracy(u::Unet, data::MLU.DataLoader)
    x = mean(reduce(vcat, [first(data) for i = 1:20]) .|> x -> loss(u, x...))
    push!(accuracy_history, x)
    return x
end

function evalCallback(u::Unet, data::MLU.DataLoader)
    Flux.throttle(60) do
        acc = accuracy(u, data)
        @info "Mean loss: $(acc)"
        save_object("model-checkpoint.jld2", u)
    end
end

u = UNet.Unet(3, 1) |> cpu

struct SatelliteData
    rs::RasterStack
    tiles::Tiles
    tileSize::Int
end
export SatelliteData

function getSatelliteData(fp::String, tileSize::Int)::Tuple{SatelliteData,SatelliteData}
    ts = load_object(joinpath(fp, "tiles_$(tileSize).jld2"))
    return (
        SatelliteData(
            RasterStack(
                (
                    x -> joinpath(fp, x)
                ).(["VV_dB.tif", "VH_dB.tif", "bathymetry_processed.tif"]);
                lazy = true,
            ),
            ts,
            tileSize,
        ),
        SatelliteData(
            RasterStack([joinpath(fp, "image_01.tif")]; lazy = true),
            ts,
            tileSize,
        ),
    )
end

function MLU.numobs(sd::SatelliteData)
    return length(sd.tiles.nonempty)
end

function MLU.getobs(sd::SatelliteData, i::Int)
    t = sd.tiles.nonempty[i]
    ret = zeros(Float32, sd.tileSize, sd.tileSize, length(sd.rs))
    for (i, layer) in enumerate(propertynames(sd.rs.layermetadata))
        ret[:, :, i] = sd.rs[layer].data[t..., 1]
    end
    return ret
end

struct MultipleSatelliteData
    satData::Vector{SatelliteData}
    ixRanges::Vector{Int}
end
export MultipleSatelliteData

function MultipleSatelliteData(satData::Vector{SatelliteData})::MultipleSatelliteData
    return MultipleSatelliteData(satData, accumulate(+, satData .|> MLU.numobs))
end

function MLU.numobs(sds::MultipleSatelliteData)::Int
    return isempty(sds.ixRanges) ? 0 : last(sds.ixRanges)
end

function MLU.getobs(sds::MultipleSatelliteData, i::Int)
    ix = searchsortedfirst(sds.ixRanges, i)
    ix0 = ix == 1 ? 0 : sds.ixRanges[ix-1]
    return MLU.getobs(sds.satData[ix], i - ix0)
end


function trainUnet(
    u::Unet,
    opt = ADAM(),
    dataDir::String = "./data/train",
    batchSize::Int = 16,
    tileSize::Int = 128,
)

    dataDirs = getDataDirs(dataDir)
    @info "Training with data directories $(dataDirs)"
    @info "Batchsize $(batchSize)"
    @info "Tile size $(tileSize)"

    satData = dataDirs .|> fp -> getSatelliteData(fp, tileSize)
    xtrain = MultipleSatelliteData([first(sd) for sd in satData])
    ytrain = MultipleSatelliteData([last(sd) for sd in satData])
    dataLoader = MLU.DataLoader(
        (xtrain, ytrain);
        # parallel = true,
        collate = true,
        batchsize = batchSize,
        shuffle = true,
    )
    l(x, y) = loss(u, x, y)
    cb(dl) = evalCallback(u, dl)
    Flux.train!(l, Flux.params(u), dataLoader, opt, cb = cb(dataLoader))

end
export trainUnet

end #Train
