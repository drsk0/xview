module Train

export unetTrain, prepareData, readData

using UNet
using Flux
using CSV
using DataFrames
using JLD2: save_object, load_object
using CUDA
using Pipe
using Base.Iterators
using Base.Threads
using TiledIteration
using Rasters
using Plots
using Base.Iterators: zip, partition
import DimensionalData.Dimensions.LookupArrays as DD

const TrainData = Tuple{Array{Float32,4},Array{Float32,4}}

function loss(x, y)
    Flux.Losses.binarycrossentropy(clamp.(u(x), 0.05f0, 0.95f0), y, agg = sum)
end

# we keep track of the evolution of the accuracy
accuracy_history = Vector{Float32}(undef, 0)
function accuracy(data::Vector{TrainData})
    x = mean(data .|> x -> loss(x...))
    push!(accuracy_history, x)
    return x
end

function evalCallback(data)
    Flux.throttle(60) do
        @info "Mean loss: $(accuracy(data))"
        save_object("model-checkpoint.jld2", u)
    end
end

function preprocessData(dataDir::String, tileSize::Int)
    csv = CSV.read(joinpath(dataDir, "train.csv"), DataFrame)
    dropmissing!(csv, :is_vessel)
    for subDir in getDataDirs(dataDir)
        preprocessDir(subDir, csv, tileSize)
    end 
end

function preprocessDir(subDir::String, csv::DataFrame, tileSize::Int)
    @info "Processing " * subDir
    id = last(splitpath(subDir))
    # TODO also filter for confidence level with in(["MEDIUM", "HIGH"]).(csv.confidence)
    vessels = @view csv[
        (csv.scene_id.==id).&(csv.is_vessel.==true),
        [:detect_scene_row, :detect_scene_column],
    ]
    open(Raster(joinpath(subDir, "VV_dB.tif"))) do gaV
        img = generateImage(gaV, vessels)
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

"For a given tile, generate the image of the distribution given the ground truth data."
function generateImage(ga::Raster{Float32,3}, vessels::SubDataFrame)::Raster{Float32,3}
    result = Raster(zeros(Float32, dims(ga)))
    vs = [
        CartesianIndex(v.detect_scene_column, v.detect_scene_row) for v in eachrow(vessels)
    ]
    result[vs] .= 1.0
    return result
end

# Return all filepath pointing to a data directory.
function getDataDirs(dataDir::String)::Vector{String}
    @pipe readdir(dataDir, join = true) |>
          filter(fp -> isdir(fp) && !endswith(fp, "shoreline"), _)
end

# Plot a raster together with the marked vessel positions.
function drawBoxes(ga::Raster{Float32,3}, vessels::Vector{CartesianIndex{2}}, predicted_vessels::Vector{CartesianIndex{2}} = [])::Plots.Plot
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
            (x_sign * xfactor, y_sign * yfactor) .* (v.I[1], v.I[2]) .+ (xbound, ybound) for
            v in predicted_vessels
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

# Apply a Unet, returning all pixel coordinates above a given threshold.
function applyU(
    u::Unet,
    rs::RasterStack,
    tileSize::Int,
    threshold::Float32
)::Vector{CartesianIndex}
    tiles = TileIterator(axes(rs[:, :, 1]), RelaxStride((tileSize, tileSize)))
    (x, y, z) = size(rs[:VV_dB].data)
    img = Matrix{Float32}(undef, x, y)
    for t in tiles
        img[t...] = u(
            reshape(
                hcat(rs[:VV_dB].data[t..., 1], rs[:VH_dB].data[t..., 1], rs[:bathymetry_processed].data[t..., 1]),
                tileSize,
                tileSize,
                3,
                1,
            ),
        )
    end
    return findall(x -> x > threshold, img)
end

mutable struct Tiles
    empty::Vector{Tuple{UnitRange{Int}, UnitRange{Int}}}
    nonempty::Vector{Tuple{UnitRange{Int}, UnitRange{Int}}}
end

function Tiles()::Tiles
    Tiles([], [])
end

function partitionTiles(fp::String, tileSize::Int)::Tiles
    tilesPerThread = repeat([Tiles()], Threads.nthreads())
    rs = RasterStack(
        (
            x -> joinpath(fp, x)
        ).(["VV_dB.tif", "VH_dB.tif", "bathymetry_processed.tif", "image.tif"]);
        lazy = true,
    )
    tiles = TileIterator(axes(@view rs[:, :, 1]), RelaxStride((tileSize, tileSize)))
    @threads for j in tiles
        tId = Threads.threadid()
        ts = tilesPerThread[tId]
        t = @view rs[j..., 1]
        if all(t[:image] .== 0)
            @info "Empty tile $(j)"
            push!(ts.empty, j)
        else
            @info "Non-empty tile $(j)"
            push!(ts.nonempty, j)
        end
    end
    
    res = Tiles()
    for ts in tilesPerThread
        append!(res.empty, ts.empty)
        append!(res.nonempty, ts.nonempty)
    end
    
    return res
end

u = gpu(Unet(3, 1))
function unetTrain(dataDir::String, batchSize::Int, nonemptyNr::Int, tileSize::Int)
    @assert nonemptyNr < batchSize

    opt = Momentum()

    dataDirs = getDataDirs(dataDir)
    @info "Training with data directories $(dataDirs)"

    epochCounter = 0

    for fp in dataDirs
        rs = RasterStack(
            (
                x -> joinpath(fp, x)
            ).(["VV_dB.tif", "VH_dB.tif", "bathymetry_processed.tif", "image.tif"]);
            lazy = true,
        )
        tiles = load_object(joinpath(fp, "tiles_$(tileSize).jld2"))
        emptyTs = partition(tiles.empty, batchSize - nonemptyNr)
        nonemptyTs = partition(tiles.nonempty, nonemptyNr)
        tsZipped = zip(emptyTs, nonemptyTs)
        for (tsEmpty, tsNonEmpty) in tsZipped
            data = TrainData[]
            ts = append!(collect(tsNonEmpty), collect(tsEmpty))
            for (tx, ty) in ts
                ret = zeros(Float32, tileSize, tileSize, 3, 1)
                ret[:, :, 1, 1] = rs[:VV_dB].data[tx, ty, 1] 
                ret[:, :, 2, 1] = rs[:VH_dB].data[tx, ty, 1]
                ret[:, :, 3, 1] = rs[:bathymetry_processed].data[tx, ty, 1]
                img = reshape(rs[:image].data[tx, ty, 1], tileSize, tileSize, 1, 1)
                push!(data, (ret, img))
            end

            epochCounter = epochCounter + 1
            @info "starting training epoch $(epochCounter)"
            Flux.train!(loss, Flux.params(u), data, opt, cb = evalCallback(data))
            empty!(data)
        end
    end

end

end #Train
