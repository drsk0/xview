module TrainCnn

using UNet
using Flux
using CSV
import DataFrames as DF
using JLD2: save_object, load_object
using CUDA
using Pipe
using Base.Threads
using TiledIteration
using Rasters
using Plots
using Base.Iterators: zip, partition
import DimensionalData.Dimensions.LookupArrays as DD
import Flux.Losses as L
using LinearAlgebra
using ProgressMeter
import MLUtils as MLU

include("Utils.jl")

function loss(x::Array{Float32,4}, y::Array{Float32, 2})::Float64
    y_head = cnn(x)
    return norm(reshape(y .- y_head, :, 1), 1)
end

accuracy_history = Vector{Float64}(undef, 0)
function accuracy(data)
    x = mean(data .|> x -> loss(x...))
    push!(accuracy_history, x)
    return x
end

function evalCallback(data)
    Flux.throttle(60) do
        acc = accuracy(data)
        @info "Mean loss: $(acc)"
        save_object("model-checkpoint-cnn.jld2", cnn)
    end
end

cnn = Chain(
    # First convolution, operating upon a 128x128 image
    Conv((3, 3), 1 => 16, pad = (1, 1), relu),
    MaxPool((2, 2)),

    # Second convolution, operating upon a 64x64 image
    Conv((3, 3), 16 => 32, pad = (1, 1), relu),
    MaxPool((2, 2)),

    # Third convolution, operating upon a 32x32 image
    Conv((3, 3), 32 => 32, pad = (1, 1), relu),
    MaxPool((2, 2)),

    # Reshape 3d array into a 2d one using `Flux.flatten`, at this point it should be (16, 16, 32, N)
    flatten,
    Dense(16 * 16 * 32, 3),
)

function train(
    u::Unet;
    dataDir::String = "./data/train",
    tileSize::Int = 128,
    batchSize::Int = 8,
)
    dataDirs = Utils.getDataDirs(dataDir)
    csv = CSV.read(joinpath(dataDir, "train.csv"), DF.DataFrame)
    opt = ADAM()
    for fp in dataDirs
        @info "Processing $(fp)"
        id = last(splitpath(fp))
        DF.dropmissing!(csv, [:is_vessel, :vessel_length_m])
        vessels = @view csv[
            (csv.scene_id.==id).&(csv.is_vessel.==true),
            [:detect_scene_row, :detect_scene_column, :vessel_length_m],
        ]
        rs = RasterStack(
            (x -> joinpath(fp, x)).(["VV_dB.tif", "VH_dB.tif", "bathymetry_processed.tif"]);
            lazy = true,
        )
        (rx, ry) = size(rs)
        dataX = Vector{Array{Float32}}()
        dataY = Vector{Vector{Float32}}()
        # TODO store tiles together with contained vessel coordinates
        ts = load_object(joinpath(fp, "tiles_$(tileSize).jld2"))
        for vessel in eachrow(vessels)
            x = vessel[:detect_scene_column]
            y = vessel[:detect_scene_row]
            i = findfirst(((rx, ry),) -> (x ∈ rx) && (y ∈ ry), ts.nonempty)
            t = ts.nonempty[i]
            (tx, ty) = t
            heatm = Utils.applyU(u, rs, t)
            img = [x - tx.start, y - ty.start, vessel[:vessel_length_m]]
            push!(dataX, heatm)
            push!(dataY, img)
        end
        data = MLU.DataLoader((
            reshape(mapreduce(permutedims, vcat, dataX), tileSize, tileSize, 1, :)
            , reshape(mapreduce(permutedims, vcat, dataY), 3, :)
            );
            batchsize = batchSize,
        )
        Flux.@epochs 3 Flux.train!(loss, Flux.params(cnn), data, opt, cb = evalCallback(data))
    end
end

end #TrainCnn
