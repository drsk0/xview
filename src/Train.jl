module Train

export unetTrain, prepareData, readData

using UNet
using Flux
using GeoArrays
using CSV
using DataFrames
using CUDA
using BSON: @save
using Pipe
using Base.Iterators

include("Tiles.jl")

const TrainData = Tuple{Array{Float32,4},Array{Float32,4}}

#TODO Not sure why I have to define it toplevel and not in unetTrain.
u = Unet(3, 1)
u = gpu(u)
opt = Momentum()

function loss(x, y)
    op = clamp.(u(x), 0.001f0, 1.0f0)
    mean(bce(op, y))
end

function accuracy(data::TrainData)
    ((x, y),) = data
    return loss(x, y)
end

function evalCallback(data)
    Flux.throttle(30) do
        @show accuracy(data)
        @save "model-checkpoint.bson" model
    end
end

function unetTrain(dataDir::String, tileSize::Int)
    tifDirs = @pipe readdir(dataDir, join = true) |>
          filter(fp -> isdir(fp) && !endswith(fp, "shoreline"), _)
    csv = CSV.read(joinpath(dataDir, "train.csv"), DataFrame)

    for tifDir in tifDirs
        gaV = GeoArrays.read(joinpath(tifDir, "VV_dB.tif"))
        gaH = GeoArrays.read(joinpath(tifDir, "VH_dB.tif"))
        for ga in [gaV, gaH]
            ga.A[ismissing.(ga.A)] .= 0
        end
        gaBat = GeoArrays.read(joinpath(tifDir, "bathymetry.tif"))
        gaBatPrim = Tiles.generateBatimetryTile(gaV, gaBat)

        (x, y) = size(gaV.A)

        function selectTile(ga::GeoArray, i::Int, j::Int)
            return @view ga[1+i*tileSize:1+(i+1)*tileSize, 1+j*tileSize:1+(j+1)*tileSize]
        end

        for i in countfrom(0)
            if (1 + (i + 1) * tileSize > x)
                break
                for j in countfrom(0)
                    if (1 + (j + 1) * tileSize > y)
                        break

                        tV = selectTile(gaV, i, j)
                        tH = selectTile(gaH, i, j)
                        tBat = selectTile(gaBatPrim, i, j)

                        img = Tiles.generateImageTile(tV, csv)
                        ret = zeros(tileSize, tileSize, 3, 1)
                        ret[:, :, 1, :] = tV.A[:, :, 1]
                        ret[:, :, 2, :] = tH.A[:, :, 1]
                        ret[:, :, 3, :] = tBat.A[:, :, 1]
                        data = (ret, reshape(img, tileSize, tileSize, 1, 1))

                        gpu(data)
                        Flux.train!(
                            loss,
                            Flux.params(u),
                            data,
                            opt,
                            cb = evalCallback(data),
                        )
                    end
                end
            end


        end

    end
end
end
