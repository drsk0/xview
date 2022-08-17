module Tiles

export generateTiles
export generateImageTile
export allVessels

using GeoArrays
using DataFrames
using Geodesy
using Pipe
using Rotations
using ImageTransformations
using CoordinateTransformations
using GeoEstimation
using GeoStatsBase
using Meshes

const TileCollection = Array{GeoArray,1}

"Subdivide TIFF into overlapping tiles of a given width and minimal overlap."
function generateTiles(width::Int, overlap::Int, ga::GeoArray)::TileCollection
    function nrTiles(sideLength::Int)::Tuple{Int,Int}
        x_nr = 1
        x_actual_overlap = 0
        while (x_actual_overlap < overlap)
            x_nr = x_nr + 1
            x_actual_overlap = cld(x_nr * width - sideLength, x_nr - 1)
        end
        return (x_nr, x_actual_overlap)
    end

    # TODO replace missing values with 0. Should we do something different?
    ga.A[ismissing.(ga.A)] .= 0

    (x, y) = size(ga.A)
    (x_nr, x_actual_overlap) = nrTiles(x)
    (y_nr, y_actual_overlap) = nrTiles(y)

    tiles = [
        ga[
            1+i*(width-x_actual_overlap):i*(width-x_actual_overlap)+width,
            1+j*(width-y_actual_overlap):j*(width-y_actual_overlap)+width,
            begin:end,
        ] for i = 0:x_nr-1 for j = 0:y_nr-1
    ]

    return tiles
end

"Take the low resolution batymetry tif and rescale to a given GeoArray via coordinates"
function generateBathymetryTile(ga::GeoArray, gaBat::GeoArray)::GeoArray
    (x, y, bands) = size(ga.A)
    (xBat, yBat) = size(gaBat.A)
    interpolate!(gaBat)
    ret = GeoArray(Array{Union{Missing,Float32}}(missing, x, y))
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
              d in eachrow(df[(df.is_vessel.!==missing).&(df.is_vessel.==true), :])
          ] .|>
          LLA(_...) .|>
          UTMZfromLLA(wgs84) .|>
          indices(ga, (_.x, _.y)) .|>
          (xy -> (xy[1], xy[2])) |>
          filter(((x, y),) -> 1 <= x && x <= x_max && 1 <= y && y <= y_max, _)
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

"Sample tile by applying symmetries."
function sampleTileWithSymmetry(
    ga::GeoArray,
    df::DataFrame,
    n::Int,
    r::Int,
)::(TyleCollection, Array{Float32,2})
    image = generateImageTile(ga, df)
    transformations = [identity rotations(ga, df, n, r)...]
    tiles = [t(ga.A) for t in transformations]
    return (tiles, image)
end

struct LocalRotation
    center::Tuple{Int,Int}
    r::Int
    rot::Angle2d

end

"Apply a local rotation. TODO check for out of bounds radii."
function (rot::LocalRotation)(x::Array{Float32,2})::Array{Float32,2}
    (c_x, c_y) = rot.center
    r = rot.r
    x_window = x[c_x-r:c_x+r, c_y-r:c_y+r]
    rotation = recenter(rot.rot, center(x_window))
    warped = no_offset_view(warp(x_window, rotation))
    (w_x, w_y) = center(warped)
    y = copy(x)
    y[c_x-r:c_x+r, c_y-r:c_y+r] = warped[w_x-r:w_x+r, w_y-r:w_y+r]
    nans = findall(isnan, y)
    y[nans] = x[nans]
    return y
end

"Compute rotations around vessels"
function rotations(
    vessels::Array{Tuple{Int,Int},1},
    n::Int,
    r::Float32,
)::Array{LocalRotation,1}
    phis = [0:2.0*pi/n:2*pi]
    [LocalRotation(v, r, Angle2d(phi)) for v in vessels for phi in phis]
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
