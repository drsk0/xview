module Tiles

export generateTiles
export generateImageTile
export allVessels

using GeoArrays
using DataFrames
# using Geodesy
using Pipe
# using ImageTransformations
using CoordinateTransformations
using GeoEstimation
using GeoStatsBase
using Meshes

"Take the low resolution batymetry tif and rescale to a given GeoArray via coordinates"
function generateBathymetryTile(ga::GeoArray, gaBat::GeoArray)::GeoArray
    (x, y, bands) = size(ga.A)
    (xBat, yBat) = size(gaBat.A)
    interpolate!(gaBat)
    ret = GeoArray(Array{Union{Missing,Float32},3}(missing, x, y, bands))
    ret.crs = ga.crs
    ret.f = ga.f
    CartesianIndices(ga.A[:, :, 1]) .|> begin
        pixel -> begin
            coords(ga, [pixel[1], pixel[2]]) |>
            c ->
                GeoArrays.indices(gaBat, [c[1], c[2]]) |>
                #checkbounds for GeoArray seems broken"
                pixelPrim -> if (1 <= pixelPrim[1] <= xBat && 1 <= pixelPrim[2] <= yBat)
                    ret.A[Tuple(pixel)..., 1] = gaBat.A[Tuple(pixelPrim)..., 1]
                end
        end
    end
    return ret
end

"Find pixels of vessels in given GeoArray"
function allVessels(ga::GeoArray, df::DataFrame)::Array{Tuple{Int,Int},1}
    (x_max, y_max, _bands) = size(ga.A)
    vessels = @pipe [
              (d.detect_lat, d.detect_lon, 0.0) for
              d in eachrow(filter(x -> x.is_vessel !== missing && x.is_vessel == true, df))
          ] .|>
          LLA(_...) .|>
          UTMZfromLLA(wgs84) .|>
          GeoArrays.indices(ga, (_.x, _.y)) .|>
          (xy -> return (xy[1], xy[2])) |>
          filter(((x, y),) -> 1 <= x <= x_max && 1 <= y <= y_max, _)
    return vessels
end

"For a given tile, generate the image of the distribution given the ground truth data."
function generateImageTile(ga::GeoArray, df::DataFrame)::Array{Float32,2}
    (x_max, y_max, _bands) = size(ga.A)
    vessels = allVessels(ga, df)
    result = zeros(Float32, x_max, y_max)
    for xy in vessels
        result[xy...] = 1.0
    end
    return result
end

"""Interpolate missing values in GeoArray."""
function interpolate!(ga::GeoArray, band = 1)
    data = @view ga.A[:, :, band]
    problemdata = georef(
        (; band = data),
        origin = Tuple(ga.f.translation),
        spacing = (abs(ga.f.linear[1]), abs(ga.f.linear[4])),
    )
    dom = LinearIndices(size(data))[ismissing.(data)]
    problem =
        EstimationProblem(problemdata, view(GeoStatsBase.domain(problemdata), dom), :band)
    solution = solve(problem, IDW(:band => (neighbors = 3,)))

    data[dom] .= solution[:band][:mean]
    return ga
end
end
