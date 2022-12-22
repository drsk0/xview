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
        (csv.scene_id.==id).&(csv.confidence ∈ ["HIGH", "MEDIUM"]),
        [:is_vessel, :detect_scene_row, :detect_scene_column],
    ]

    tiles = partitionTiles(fp, objects, 128)

    r = RasterStack([joinpath(fp, f) for f ∈ ["VV_dB.tif", "VH_dB.tif", "bathymetry_processed.tif"]]; lazy = true)

    objectToTile = Dict()
    for (t, os) ∈ tiles.nonempty
        for o ∈ os 
            # TODO deal with multiple objects within the same tile here
            push!(objectToTile, o => t)
        end
    end

    y = [o[:is_vessel] ? 1.0 : 0.0 for o in eachrow(objects)]

    y_head = [
        (applyU(u, r, objectToTile[o]) .|> sigmoid)[o...] > treshold ? 1.0 : 0.0
         for o in eachrow(objects)
    ]

    return f1score(y, y_head)

end

function pe_L(l_head::Vector{Float32}, l::Vector{Float32})::Float32
    @assert length(l_head) == length(l)

    N = length(l)
    return 1.0 - min(1.0 / N * norm((@. (l_head - l) / l), 1.0), 1.0)
end

end
