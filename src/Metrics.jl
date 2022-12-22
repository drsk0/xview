module Metrics

using MLJBase
using DataFrames
using CSV
using Flux
using Rasters
using ..Utils
using ..UNet
using ..Preprocess

function f1_V(fp::String, csv::DataFrame, u::Unet, treshold::Float32)::Float32
    id = last(splitpath(fp))
    dropmissing!(csv, :is_vessel)
    objects = @view csv[
        (csv.scene_id.==id).&(csv.confidence.∈[["HIGH", "MEDIUM"]]),
        [:is_vessel, :detect_scene_row, :detect_scene_column],
    ]

    tiles = partitionTiles(fp, objects, 128)

    r = RasterStack([joinpath(fp, f) for f ∈ ["VV_dB.tif", "VH_dB.tif", "bathymetry_processed.tif"]]; lazy=true)

    objectToTile = Dict()
    for (t, os) ∈ tiles.nonempty
        for o ∈ os
            push!(objectToTile, o => t)
        end
    end

    y = [o[:is_vessel] ? 1.0 : 0.0 for o in eachrow(objects)]

    y_head = [
        begin
            coord = CartesianIndex(o.detect_scene_column, o.detect_scene_row)
            t = objectToTile[coord]
            (applyU(u, r, t).|>sigmoid)[coord - CartesianIndex(t[1].start - 1, t[2].start - 1)] > treshold ? 1.0 : 0.0
        end
        for o ∈ eachrow(objects) 
    ]

    return f1score(y_head, y)
end
end
