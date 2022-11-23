module TrainCnn

using Flux
using CSV
import DataFrames as DF
using JLD2: save_object, load_object
using CUDA
using Pipe
using Base.Threads
using TiledIteration
using Rasters
using Base.Iterators: zip, partition
import Flux.Losses as L
using LinearAlgebra
using ProgressMeter
import MLUtils as MLU
using Metalhead

using ..UNet
using ..Utils

function loss(x::Array{Float32,4}, y::Array{Float32,2})::Float64
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

cnn = GoogLeNet(
    Chain(
        Chain(
            Conv((7, 7), 1 => 64, pad = 3, stride = 2),  # 9_472 parameters
            MaxPool((3, 3), pad = 1, stride = 2),
            Conv((1, 1), 64 => 64),           # 4_160 parameters
            Conv((3, 3), 64 => 192, pad = 1),   # 110_784 parameters
            MaxPool((3, 3), pad = 1, stride = 2),
            Parallel(
                Metalhead.cat_channels,
                Chain(
                    Conv((1, 1), 192 => 64),      # 12_352 parameters
                ),
                Chain(
                    Conv((1, 1), 192 => 96),      # 18_528 parameters
                    Conv((3, 3), 96 => 128, pad = 1),  # 110_720 parameters
                ),
                Chain(
                    Conv((1, 1), 192 => 16),      # 3_088 parameters
                    Conv((5, 5), 16 => 32, pad = 2),  # 12_832 parameters
                ),
                Chain(
                    MaxPool((3, 3), pad = 1, stride = 1),
                    Conv((1, 1), 192 => 32),      # 6_176 parameters
                ),
            ),
            Parallel(
                Metalhead.cat_channels,
                Chain(
                    Conv((1, 1), 256 => 128),     # 32_896 parameters
                ),
                Chain(
                    Conv((1, 1), 256 => 128),     # 32_896 parameters
                    Conv((3, 3), 128 => 192, pad = 1),  # 221_376 parameters
                ),
                Chain(
                    Conv((1, 1), 256 => 32),      # 8_224 parameters
                    Conv((5, 5), 32 => 96, pad = 2),  # 76_896 parameters
                ),
                Chain(
                    MaxPool((3, 3), pad = 1, stride = 1),
                    Conv((1, 1), 256 => 64),      # 16_448 parameters
                ),
            ),
            MaxPool((3, 3), pad = 1, stride = 2),
            Parallel(
                Metalhead.cat_channels,
                Chain(
                    Conv((1, 1), 480 => 192),     # 92_352 parameters
                ),
                Chain(
                    Conv((1, 1), 480 => 96),      # 46_176 parameters
                    Conv((3, 3), 96 => 208, pad = 1),  # 179_920 parameters
                ),
                Chain(
                    Conv((1, 1), 480 => 16),      # 7_696 parameters
                    Conv((5, 5), 16 => 48, pad = 2),  # 19_248 parameters
                ),
                Chain(
                    MaxPool((3, 3), pad = 1, stride = 1),
                    Conv((1, 1), 480 => 64),      # 30_784 parameters
                ),
            ),
            Parallel(
                Metalhead.cat_channels,
                Chain(
                    Conv((1, 1), 512 => 160),     # 82_080 parameters
                ),
                Chain(
                    Conv((1, 1), 512 => 112),     # 57_456 parameters
                    Conv((3, 3), 112 => 224, pad = 1),  # 226_016 parameters
                ),
                Chain(
                    Conv((1, 1), 512 => 24),      # 12_312 parameters
                    Conv((5, 5), 24 => 64, pad = 2),  # 38_464 parameters
                ),
                Chain(
                    MaxPool((3, 3), pad = 1, stride = 1),
                    Conv((1, 1), 512 => 64),      # 32_832 parameters
                ),
            ),
            Parallel(
                Metalhead.cat_channels,
                Chain(
                    Conv((1, 1), 512 => 128),     # 65_664 parameters
                ),
                Chain(
                    Conv((1, 1), 512 => 128),     # 65_664 parameters
                    Conv((3, 3), 128 => 256, pad = 1),  # 295_168 parameters
                ),
                Chain(
                    Conv((1, 1), 512 => 24),      # 12_312 parameters
                    Conv((5, 5), 24 => 64, pad = 2),  # 38_464 parameters
                ),
                Chain(
                    MaxPool((3, 3), pad = 1, stride = 1),
                    Conv((1, 1), 512 => 64),      # 32_832 parameters
                ),
            ),
            Parallel(
                Metalhead.cat_channels,
                Chain(
                    Conv((1, 1), 512 => 112),     # 57_456 parameters
                ),
                Chain(
                    Conv((1, 1), 512 => 144),     # 73_872 parameters
                    Conv((3, 3), 144 => 288, pad = 1),  # 373_536 parameters
                ),
                Chain(
                    Conv((1, 1), 512 => 32),      # 16_416 parameters
                    Conv((5, 5), 32 => 64, pad = 2),  # 51_264 parameters
                ),
                Chain(
                    MaxPool((3, 3), pad = 1, stride = 1),
                    Conv((1, 1), 512 => 64),      # 32_832 parameters
                ),
            ),
            Parallel(
                Metalhead.cat_channels,
                Chain(
                    Conv((1, 1), 528 => 256),     # 135_424 parameters
                ),
                Chain(
                    Conv((1, 1), 528 => 160),     # 84_640 parameters
                    Conv((3, 3), 160 => 320, pad = 1),  # 461_120 parameters
                ),
                Chain(
                    Conv((1, 1), 528 => 32),      # 16_928 parameters
                    Conv((5, 5), 32 => 128, pad = 2),  # 102_528 parameters
                ),
                Chain(
                    MaxPool((3, 3), pad = 1, stride = 1),
                    Conv((1, 1), 528 => 128),     # 67_712 parameters
                ),
            ),
            MaxPool((3, 3), pad = 1, stride = 2),
            Parallel(
                Metalhead.cat_channels,
                Chain(
                    Conv((1, 1), 832 => 256),     # 213_248 parameters
                ),
                Chain(
                    Conv((1, 1), 832 => 160),     # 133_280 parameters
                    Conv((3, 3), 160 => 320, pad = 1),  # 461_120 parameters
                ),
                Chain(
                    Conv((1, 1), 832 => 32),      # 26_656 parameters
                    Conv((5, 5), 32 => 128, pad = 2),  # 102_528 parameters
                ),
                Chain(
                    MaxPool((3, 3), pad = 1, stride = 1),
                    Conv((1, 1), 832 => 128),     # 106_624 parameters
                ),
            ),
            Parallel(
                Metalhead.cat_channels,
                Chain(
                    Conv((1, 1), 832 => 384),     # 319_872 parameters
                ),
                Chain(
                    Conv((1, 1), 832 => 192),     # 159_936 parameters
                    Conv((3, 3), 192 => 384, pad = 1),  # 663_936 parameters
                ),
                Chain(
                    Conv((1, 1), 832 => 48),      # 39_984 parameters
                    Conv((5, 5), 48 => 128, pad = 2),  # 153_728 parameters
                ),
                Chain(
                    MaxPool((3, 3), pad = 1, stride = 1),
                    Conv((1, 1), 832 => 128),     # 106_624 parameters
                ),
            ),
        ),
        Chain(
            AdaptiveMeanPool((1, 1)),
            Flux.flatten,
            Dropout(0.4),
            Dense(1024, 3),                   # 3_075 parameters
        ),
    ),
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
        data = MLU.DataLoader(
            (
                reshape(mapreduce(permutedims, vcat, dataX), tileSize, tileSize, 1, :),
                reshape(mapreduce(permutedims, vcat, dataY), 3, :),
            );
            batchsize = batchSize,
        )
        Flux.@epochs 100 Flux.train!(
            loss,
            Flux.params(cnn),
            data,
            opt,
            cb = evalCallback(data),
        )
    end
end

end #TrainCnn
