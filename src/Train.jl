module Train

export unetTrain, prepareData, readData

using UNet
using Flux
using GeoArrays
using CSV
using DataFrames
using CUDA
using BSON: @save, @load
using Pipe
using Base.Iterators
using Base.Threads

include("Tiles.jl")

const TrainData = Tuple{Array{Float32,4},Array{Float32,4}}

function loss(x, y)
    op = clamp.(u(x), 0.001f0, 1.0f0)
    mean(bce(op, y))
end

function accuracy(data::Vector{TrainData})
    return mean(data .|> x -> loss(x...))
end

function evalCallback(data)
    Flux.throttle(30) do
        @show accuracy(data)
        @save "model-checkpoint.bson" u
    end
end

function processData(dataDir::String)
    csv = CSV.read(joinpath(dataDir, "train.csv"), DataFrame)
    getDataDirs(dataDir) .|> subDir -> processGAs(csv, subDir)
end

function processGAs(csv::DataFrame, subDir::String)
    gaV = GeoArrays.read(joinpath(subDir, "VV_dB.tif"))
    gaH = GeoArrays.read(joinpath(subDir, "VH_dB.tif"))

    # TODO get rid of this
    @threads for ga in [gaV, gaH]
        ga.A[ismissing.(ga.A)] .= 0
    end

    gaBat = GeoArrays.read(joinpath(subDir, "bathymetry.tif"))
    gaBatPrim = Tiles.generateBathymetryTile(gaV, gaBat)
    img = Tiles.generateImageTile(gaV, csv)

    @threads for (ga, relPath) in
                 [(gaV, "VV_dB"), (gaH, "VH_dB"), (gaBatPrim, "bathymetry")]
        gas = reshape(partitionGA(ga), (1, 16))
        for (i, gaA) in zip(1:16, gas)
            ga.A = gaA
            mkpath(joinpath(subDir, "processed"))
            GeoArrays.write!(gaTileName(subDir, relPath, i), ga)
        end
    end
    
    imgs = reshape(partition(img), (1, 16))
    for (i, im) in zip(1:16, imgs)
        mkpath(joinpath(subDir, "processed"))
        @save imgTileName(subDir, "image", i) im
    end


end

# Compute the name of the i-th tile of a GeoArray
function gaTileName(fp::String, relPath::String, i::Int)::String
    return joinpath(fp, "processed", relPath * "_$(i).tiff")
end

function imgTileName(fp::String, relPath::String, i::Int)::String
    return joinpath(fp, "processed", relPath * "_$(i).bson")
end

# Paritition a GeoArray in 16 tiles.
function partitionGA(ga::GeoArray)::Matrix{Array{Union{Missing,Float32},3}}
    ret = Matrix{Array{Union{Missing, Float32}, 3}}(undef, (4, 4))
    (x, y, z) = size(ga)
    for i = 1:4
        for j = 1:4
            ret[i, j] = ga[
                begin+(i-1)*div(x, 4):begin+i*div(x, 4),
                begin+(j-1)*div(y, 4):begin+j*div(y, 4),
                begin:end
            ]
        end
    end
    return ret
end

function partition(xs::Matrix{T})::Matrix{Matrix{T}} where T
    ret = Matrix{Matrix{T}}(undef, (4, 4))
    (x, y) = size(xs)
    for i = 1:4
        for j = 1:4
            ret[i, j] = xs[
                begin+(i-1)*div(x, 4):begin+i*div(x, 4),
                begin+(j-1)*div(y, 4):begin+j*div(y, 4),
            ]
        end
    end
    return ret
end

# Return all filepath pointing to a data directory.
function getDataDirs(dataDir::String)::Vector{String}
    @pipe readdir(dataDir, join = true) |>
          filter(fp -> isdir(fp) && !endswith(fp, "shoreline"), _)
end

u = Unet(3, 1)
function unetTrain(dataDir::String, batchSize::Int, tileSize::Int)
    opt = Momentum()

    dataDirs = getDataDirs(dataDir)
    @show dataDirs

    data = Vector{TrainData}(undef, 0)

    for fp in dataDirs
        for i = 1:16

            gaV = GeoArrays.read(gaTileName(fp, "VV_dB", i))
            gaH = GeoArrays.read(gaTileName(fp, "VH_dB", i))
            gaBat = GeoArrays.read(gaTileName(fp, "bathymetry", i))
            @load imgTileName(fp, "image", i) im
            img = im

            (x, y) = size(gaV.A)

            function selectTile(ga::GeoArray, i::Int, j::Int)
                return @view ga.A[
                    1+i*tileSize:(i+1)*tileSize,
                    1+j*tileSize:(j+1)*tileSize,
                    begin:end,
                ]
            end
            
            function selectImg(img::Matrix{Float32}, i::Int, j::Int)
                return @view img[
                    1+i*tileSize:(i+1)*tileSize,
                    1+j*tileSize:(j+1)*tileSize,
                    ]
            end

            for i in countfrom(0)
                if ((i + 1) * tileSize > x)
                    break
                else
                    for j in countfrom(0)
                        if ((j + 1) * tileSize > y)
                            break
                        else

                            tV = selectTile(gaV, i, j)
                            tH = selectTile(gaH, i, j)
                            tBat = selectTile(gaBat, i, j)
                            im = reshape(selectImg(img, i, j), tileSize, tileSize, 1, 1)
                            ret = zeros(Float32, tileSize, tileSize, 3, 1)
                            ret[:, :, 1, 1] = tV[:, :]
                            ret[:, :, 2, 1] = tH[:, :]
                            ret[:, :, 3, 1] = tBat[:, :]

                            if (length(data) < batchSize)
                                push!(data, (ret, im))
                            else
                                println("training epoch!")
                                Flux.train!(
                                    loss,
                                    Flux.params(u),
                                    data,
                                    opt,
                                    cb = evalCallback(data),
                                )
                                empty!(data)
                            end
                        end
                    end
                end

            end

        end

    end
end
end
