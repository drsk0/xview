module Train

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
import Flux.Losses as L
using LinearAlgebra
using ThreadsX
using LoopVectorization

const TrainData = Tuple{Array{Float32,4},Array{Float32,4}}

function h(x)
    clamp(x, 0.001, 1.0)
end

function weightedMean(xs, ws)
    return mean(ws .* xs)
end
function loss(x, y)
    y_head = h.(u(x))
    @info size(y)
    (xdim, ydim) = size(y)
    muS = xdim * ydim
    int_falpha = sum(y)
    int_fbeta = muS - int_falpha
    alpha = int_falpha > 0.0 ? muS / ((1001/1000) * int_falpha) : 1.0
    beta = int_fbeta > 0.0 ? (alpha * int_falpha) / (1000 * int_fbeta) : 1.0
    ws = alpha * y .+ beta * (1.0 .- y)
    return L.logitbinarycrossentropy(y_head, y; agg = xs -> weightedMean(xs, ws))
end

# we keep track of the evolution of the accuracy
accuracy_history = Vector{Float32}(undef, 0)
function accuracy(data::Vector{TrainData})
    xs = data .|> x -> loss(x...)
    x = mean(xs)
    @info "Losses: $(xs)"
    push!(accuracy_history, x)
    return x
end

function evalCallback(data)
    Flux.throttle(60) do
        acc = accuracy(data)
        @info "Mean loss: $(acc)"
        save_object("model-checkpoint_$(acc).jld2", u)
    end
end

function preprocessData(dataDir::String, tileSize::Int, alpha::Float32 = 0.1)
    csv = CSV.read(joinpath(dataDir, "train.csv"), DataFrame)
    dropmissing!(csv, [:is_vessel, :vessel_length_m])
    for subDir in getDataDirs(dataDir)
        preprocessDir(subDir, csv, tileSize)
    end
end

function preprocessDir(subDir::String, csv::DataFrame, tileSize::Int = 128, alpha::Float32 = 0.1)
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

# Return all filepath pointing to a data directory.
function getDataDirs(dataDir::String)::Vector{String}
    @pipe readdir(dataDir, join = true) |>
          filter(fp -> isdir(fp) && !endswith(fp, "shoreline"), _)
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

# Apply a Unet, returning all pixel coordinates above a given threshold.
function applyU(u::Unet, rs::RasterStack, tileSize::Int)::Matrix{Float32}
    tiles = TileIterator(axes(rs[:, :, 1]), RelaxStride((tileSize, tileSize)))
    x, y = size(rs[:VV_dB])[1], size(rs[:VV_dB])[2]
    img = zeros(Float32, x, y)
    for t in tiles
        img[t...] = applyU(u, rs, t)
    end
    return img
end

const Tile = Tuple{UnitRange{Int},UnitRange{Int}}

function applyU(u, rs::RasterStack, t::Tile)::Matrix{Float32}
    tileSize = length(t[1])
    u(
        reshape(
            hcat(
                rs[:VV_dB].data[t..., 1],
                rs[:VH_dB].data[t..., 1],
                rs[:bathymetry_processed].data[t..., 1],
            ),
            tileSize,
            tileSize,
            3,
            1,
        ),
    )[
        :,
        :,
        1,
        1,
    ]
end

mutable struct Tiles
    empty::Vector{Tile}
    nonempty::Vector{Tile}
end

function Tiles()::Tiles
    Tiles([], [])
end

function checkTiles(ts::Tiles, u::Unet, rs::RasterStack, tileSize::Int, threshold::Float64)
    function check(t::Tile)::Bool
        return any(x -> x > threshold, h.(applyU(u, rs, t)))
    end

    # ThreadsX.findfirst seems buggy and returns something even when the condition is not met.
    i = findfirst(check, ts.nonempty)
    if isnothing(i)
        return nothing
    else
        t = ts.nonempty[i]
        return (h.(
            u(
                reshape(
                    hcat(
                        rs[:VV_dB].data[t..., 1],
                        rs[:VH_dB].data[t..., 1],
                        rs[:bathymetry_processed].data[t..., 1],
                    ),
                    tileSize,
                    tileSize,
                    3,
                    1,
                ),
            )[
                :,
                :,
                1,
                1,
            ]
        ), i)
    end
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
        if any(t[:image] .== 1.0)
            push!(ts.nonempty, j)
        else
            push!(ts.empty, j)
        end
    end

    res = Tiles()
    for ts in tilesPerThread
        append!(res.empty, ts.empty)
        append!(res.nonempty, ts.nonempty)
    end

    return res
end

u = Unet(3, 1) |> cpu
function train(
    dataDir::String = "./data/train",
    batchSize::Int = 16,
    tileSize::Int = 128,
    alpha::String = "025",
)
    opt = Adam()

    imgfilename = "image_" * alpha

    dataDirs = getDataDirs(dataDir)
    @info "Training with data directories $(dataDirs)"
    @info "Batchsize $(batchSize)"
    @info "Tile size $(tileSize)"
    @info "alpha $(alpha)"

    epochCounter = 0

    for fp in dataDirs
        rs = RasterStack(
            (
                x -> joinpath(fp, x)
            ).([
                "VV_dB.tif",
                "VH_dB.tif",
                "bathymetry_processed.tif",
                imgfilename * ".tif",
            ]);
            lazy = true,
        )
        tiles = load_object(joinpath(fp, "tiles_$(tileSize).jld2"))
        nonemptyTs = partition(tiles.nonempty, batchSize)
        for ts in nonemptyTs
            data = TrainData[]
            for t in ts
                ret = zeros(Float32, tileSize, tileSize, 3, 1)
                ret[:, :, 1, 1] = rs[:VV_dB].data[t..., 1]
                ret[:, :, 2, 1] = rs[:VH_dB].data[t..., 1]
                ret[:, :, 3, 1] = rs[:bathymetry_processed].data[t..., 1]
                img = reshape(
                    rs[Symbol(imgfilename)].data[t..., 1],
                    tileSize,
                    tileSize,
                    1,
                    1,
                )
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
