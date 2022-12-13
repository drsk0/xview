module Preprocess
using DataFrames
using Rasters
using LinearAlgebra
using Plots
using ThreadsX
using Base.Threads
using TiledIteration
using JLD2
import DimensionalData.Dimensions.LookupArrays as DD
using ..Utils

function preprocessData(dataDir::String, tileSize::Int)
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
    alpha::Float32 = 0.1f32,
)
    @info "Processing " * subDir
    id = last(splitpath(subDir))
    # TODO also filter for confidence level with in(["MEDIUM", "HIGH"]).(csv.confidence)
    vessels = @view csv[
        (csv.scene_id.==id).&(csv.is_vessel.!==missing) .& (csv.is_vessel.==true),
        [:detect_scene_row, :detect_scene_column, :vessel_length_m],
    ]
    open(Raster(joinpath(subDir, "VV_dB.tif"))) do gaV
        img = generateImage(gaV, vessels, alpha)
        Rasters.write(joinpath(subDir, "image_"*alpha *".tif"), img)

        open(Raster(joinpath(subDir, "bathymetry.tif"))) do gaBat
            gaBatPrim = resample(gaBat, to = gaV, method = :near)
            gaV = nothing
            Rasters.write(joinpath(subDir, "bathymetry_processed.tif"), gaBatPrim)
        end
    end
    ts = partitionTiles(subDir, tileSize)
    save_object(joinpath(subDir, "tiles_$(tileSize).jld2"), ts)
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
    result[:, :, 1] = map(
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

function partitionTiles(fp::String, tileSize::Int)::Utils.Tiles
    tilesPerThread = repeat([Utils.Tiles()], Threads.nthreads())
    rs = RasterStack(
        (
            x -> joinpath(fp, x)
        ).(["VV_dB.tif", "VH_dB.tif", "bathymetry_processed.tif", "image_01.tif"]);
        lazy = true,
    )
    tiles = TileIterator(axes(@view rs[:, :, 1]), RelaxStride((tileSize, tileSize)))
    @threads for j in tiles
        tId = Threads.threadid()
        ts = tilesPerThread[tId]
        t = @view rs[j..., 1]
        if any(t[:image_01] .== 1.0)
            push!(ts.nonempty, j)
        else
            push!(ts.empty, j)
        end
    end

    res = Utils.Tiles()
    for ts in tilesPerThread
        append!(res.empty, ts.empty)
        append!(res.nonempty, ts.nonempty)
    end

    return res
end
export partitionTiles

end #Preprocess