module Train

export unetTrain, prepareData, readData

using UNet
using Flux
using CSV
using DataFrames
using CUDA
using BSON: @save, @load
using Pipe
using Base.Iterators
using Base.Threads
using ImageDraw
using TiledIteration
using ColorTypes
using Rasters

const TrainData = Tuple{Array{Float32,4},Array{Float32,4}}

function loss(x, y)
    Flux.Losses.binarycrossentropy(clamp.(u(x),0.05f0, 0.95f0), y, agg = sum)
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
        @save "model-checkpoint.bson" u
    end
end

function processData(dataDir::String)
    csv = CSV.read(joinpath(dataDir, "train.csv"), DataFrame)
    dropmissing!(csv, :is_vessel)
    getDataDirs(dataDir) .|> subDir -> processGAs(csv, subDir)
end

function processGAs(csv::DataFrame, subDir::String)
    gaV = Raster(joinpath(subDir, "VV_dB.tif"))
    gaH = Raster(joinpath(subDir, "VH_dB.tif"))

    gaBat = Raster(joinpath(subDir, "bathymetry.tif"))
    gaBatPrim = resample(gaBat, to=gaV, method=:near)
    # TODO also filter for confidence level with in(["MEDIUM", "HIGH"]).(csv.confidence)
    vessels = @view csv[(csv.scene_id .== subDir) .& (csv.is_vessel .== true), [:detect_scene_row, :detect_scene_column]]
    img = generateImage(gaV, csv)
    
    Rasters.write(joinpath(subDir, "bathymetry_processed.tif"), gaBatPrim)
    @save imgTileName(subDir, "image.bson") img
end

"For a given tile, generate the image of the distribution given the ground truth data."
function generateImage(ga::Raster{Float32, 3}, vessels::SubDataFrame)::Raster{Float32,3}
    result = copy(ga)
    xs = result.data
    xs .= 0.0
    for r in eachrow(vessels)
        xs[r.detect_scene_row, r.detect_scene_column, 1] = 1.0
    end
    return result
end

# Return all filepath pointing to a data directory.
function getDataDirs(dataDir::String)::Vector{String}
    @pipe readdir(dataDir, join = true) |>
          filter(fp -> isdir(fp) && !endswith(fp, "shoreline"), _)
end

function drawBoxes(ga::Raster{Float32, 3}, img::Raster{Float32, 3})::Raster{Float32, 3}
    tmp = copy(ga)
    xs = ga.data
    ys = img.data
    c = maximum(xs) + 20.0f0
    ret = Gray.(xs[:, :, 1])
    for (x, y) in [(i[1], i[2]) for i = CartesianIndices(ret) if ys[Tuple(i)...] > 0.9]
        draw!(
            ret,
            Ellipse(CirclePointRadius(x, y, 20; fill = true)),
            Gray(c),
        )
    end
    tmp.data[:,:,1] = gray.(ret)

    return tmp
end

function applyU(
    u::Unet,
    tileSize::Int,
    gaV::Raster{Float32, 3},
    gaH::Raster{Float32, 3},
    bat::Raster{Float32, 3},
)::Matrix{Float32}
    tiles = TileIterator(axes(gaV.A[:, :, 1]), RelaxStride((tileSize, tileSize)))
    (x, y, z) = size(gaV)
    ret = Matrix{Float32}(undef, x, y)
    for t in tiles
        ret[t...] = u(
            reshape(
                hcat(gaV.A[t..., 1], gaH.A[t..., 1], bat.A[t..., 1]),
                tileSize,
                tileSize,
                3,
                1,
            ),
        )
    end
    return ret
end

u = Unet(3, 1)
function unetTrain(dataDir::String, batchSize::Int, tileSize::Int)
    opt = Momentum()

    dataDirs = getDataDirs(dataDir)
    @info dataDirs

    data = Vector{TrainData}(undef, 0)
    epochCounter = 0

    for fp in dataDirs
        for i = 1:16

            gaV = GeoArrays.read(gaTileName(fp, "VV_dB", i))
            gaH = GeoArrays.read(gaTileName(fp, "VH_dB", i))
            gaBat = GeoArrays.read(gaTileName(fp, "bathymetry", i))
            @load imgTileName(fp, "image", i) im
            img = im
            tiles = TileIterator(axes(gaV.A[:, :, 1]), RelaxStride((tileSize, tileSize)))
            for j in tiles
                tV = gaV[j..., 1]
                tH = gaH[j..., 1]
                tBat = gaBat[j..., 1]
                im = reshape(img[j...], tileSize, tileSize, 1, 1)
                if count(x -> x == 1.0, im) == 0
                    continue;
                end
                ret = zeros(Float32, tileSize, tileSize, 3, 1)
                ret[:, :, 1, 1] = tV[:, :]
                ret[:, :, 2, 1] = tH[:, :]
                ret[:, :, 3, 1] = tBat[:, :]

                if (length(data) < batchSize)
                    push!(data, (ret, im))
                else
                    epochCounter = epochCounter + 1
                    @info "starting training epoch $(epochCounter)"
                    Flux.train!(loss, Flux.params(u), data, opt, cb = evalCallback(data))
                    empty!(data)
                end
            end


        end

    end
end
end
