module Metrics

using MLJBase
using DataFrames

function f1_V(csv::DataFrame, img::Array{Float32}, id::String, treshold::Float)::Float
    dropmissing!(csv, [:is_vessel])
    objects = @view csv[
        (csv.scene_id.==id).&(csv.confidence ∈ ["HIGH", "MEDIUM"]),
        [:is_vessel, :detect_scene_row, :detect_scene_column, :vessel_length_m],
    ]
    y = [o[:is_vessel] ? 1.0 : 0.0 for o in eachrow(objects)]

    y_head = [
        img[o[:detect_scene_row], o[:detect_scene_column]] > treshold ? 1.0 : 0.0 for
        o in eachrow(objects)
    ]

    return f1score(y, y_head)

end

function pe_L(l_head::Vector{Float32}, l::Vector{Float32})::Float
    @assert length(l_head) == length(l)

    N = length(l)
    return 1.0 - min(1.0 / N * norm((@. (l_head - l) / l), 1.0), 1.0)
end

end
