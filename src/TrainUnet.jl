module TrainUnet

using Flux
using CSV
using DataFrames
using JLD2: save_object, load_object
using CUDA
using Pipe
using Base.Iterators
using TiledIteration
using Rasters
using Plots
using Base.Iterators: zip, partition
import DimensionalData.Dimensions.LookupArrays as DD
import Flux.Losses as L
using LinearAlgebra
using ThreadsX
using LoopVectorization
using ProgressMeter
using Statistics
using ..UNet
using ..Utils
import MLUtils as MLU


function h(x)
    clamp(x, 0.001, 1.0)
end

function weightedMean(xs, ws)
    return mean(ws .* xs)
end
function loss(x, y)
    y_head = h.(u(x))
    (xdim, ydim) = size(y)
    muS = xdim * ydim
    int_falpha = sum(y)
    int_fbeta = muS - int_falpha
    alpha = int_falpha > 0.0 ? muS / (2 * int_falpha) : 1.0
    beta = int_fbeta > 0.0 ? (alpha * int_falpha) / int_fbeta : 1.0
    ws = alpha * y .+ beta * (1.0 .- y)
    return L.logitbinarycrossentropy(y_head, y; agg = xs -> weightedMean(xs, ws))
end

# we keep track of the evolution of the accuracy
accuracy_history = Vector{Float32}(undef, 0)
function accuracy(data::MLU.DataLoader)
    x = mean(reduce(vcat, [first(data) for i = 1:20]) .|> x -> loss(x...))
    push!(accuracy_history, x)
    return x
end

function evalCallback(data::MLU.DataLoader)
    Flux.throttle(60) do
        acc = accuracy(data)
        @info "Mean loss: $(acc)"
        save_object("model-checkpoint.jld2", u)
    end
end

function preprocessData(dataDir::String, tileSize::Int, alpha::Float32 = 0.1)
    csv = CSV.read(joinpath(dataDir, "train.csv"), DataFrame)
    dropmissing!(csv, [:is_vessel, :vessel_length_m])
    for subDir in getDataDirs(dataDir)
        preprocessDir(subDir, csv, tileSize)
    end
end

function preprocessDir(
    subDir::String,
    csv::DataFrame,
    tileSize::Int = 128,
    alpha::Float32 = 0.1,
)
    @info "Processing " * subDir
    id = last(splitpath(subDir))
    # TODO also filter for confidence level with in(["MEDIUM", "HIGH"]).(csv.confidence)
    vessels = @view csv[
        (csv.scene_id.==id).&(csv.is_vessel.==true),
        [:detect_scene_row, :detect_scene_column, :vessel_length_m],
    ]
    open(Raster(joinpath(subDir, "VV_dB.tif"))) do gaV
        img = generateImage(gaV, vessels, alpha)
        Rasters.write(joinpath(subDir, "image.tif"), img)

        open(Raster(joinpath(subDir, "bathymetry.tif"))) do gaBat
            gaBatPrim = resample(gaBat, to = gaV, method = :near)
            gaV = nothing
            Rasters.write(joinpath(subDir, "bathymetry_processed.tif"), gaBatPrim)
        end
    end
    ts = partitionTiles(subDir, tileSize)
    save_object("tiles_$(tileSize).jld2", ts)
end

function generateImage(
    ga::Raster{Float32,3},
    vessels::SubDataFrame,
    alpha::Float32,
)::Raster{Float32,3}
    result = Raster(zeros(Float32, dims(ga)))
    vs = [
        (CartesianIndex(v.detect_scene_column, v.detect_scene_row), v.vessel_length_m)
        for v in eachrow(vessels)
    ]
    result[:, :, 1] = ThreadsX.map(
        i -> maximum(
            v -> begin
                sigma = alpha * v[2]
                r = exp(-(norm((v[1] - i).I))^2 / (2 * sigma^2))
                return r
            end,
            vs;
            init = 0.0,
        ),
        CartesianIndices(result[:, :, 1]),
    )
    return result
end

# Plot a raster together with the marked vessel positions.
function drawBoxes(
    ga::Raster{Float32,3},
    vessels::Vector{CartesianIndex{2}},
    predicted_vessels::Vector{CartesianIndex{2}} = [],
)::Plots.Plot
    x_dim, y_dim = dims(ga)[1], dims(ga)[2]
    x_sign = DD.order(x_dim) isa DD.ForwardOrdered ? 1 : -1
    y_sign = DD.order(y_dim) isa DD.ForwardOrdered ? 1 : -1
    (xbounds, ybounds, zbounds) = bounds(ga)
    xbound = DD.order(x_dim) isa DD.ForwardOrdered ? xbounds[1] : xbounds[2]
    ybound = DD.order(y_dim) isa DD.ForwardOrdered ? ybounds[1] : ybounds[2]
    (xsize, ysize, zsize) = size(ga)
    xfactor = (xbounds[2] - xbounds[1]) / xsize
    yfactor = (ybounds[2] - ybounds[1]) / ysize
    vessels_coords = [
        (x_sign * xfactor, y_sign * yfactor) .* (v.I[1], v.I[2]) .+ (xbound, ybound) for
        v in vessels
    ]
    p = plot(ga)
    scatter!(
        p,
        vessels_coords;
        legend = :none,
        markercolor = :black,
        markerstrokewidth = 10,
        markerstrokealpha = 1.0,
        markershape = :rect,
        markeralpha = 0.6,
    )
    if !isempty(predicted_vessels)
        predicted_vessels_coords = [
            (x_sign * xfactor, y_sign * yfactor) .* (v.I[1], v.I[2]) .+ (xbound, ybound) for v in predicted_vessels
        ]
        scatter!(
            p,
            predicted_vessels_coords;
            legend = :none,
            markercolor = :red,
            markerstrokewidth = 10,
            markerstrokealpha = 1.0,
            markershape = :rect,
            markeralpha = 0.6,
        )
    end
    return p
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
    dataDir::String = "./data/train",
    batchSize::Int = 16,
    tileSize::Int = 128,
)
    opt = ADAM()

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
    @Flux.epochs 3 Flux.train!(loss, Flux.params(u), dataLoader, opt, cb = evalCallback(dataLoader))

end
export trainUnet

end #Train
