module Tiles

export generateTiles
export generateImageTile

using GeoArrays
using DataFrames
using Geodesy
using Pipe
using Rotations
using ImageTransformations
using CoordinateTransformations

const TileCollection = Array{GeoArray,1}

"Subdivide TIFF into overlapping tiles of a given width and minimal overlap."
function generateTiles(width::Int, overlap::Int, ga::GeoArray)::TileCollection
  (x, y) = size(ga.A)
  x_nr = 1
  x_actual_overlap = 0
  while (x_actual_overlap < overlap)
    x_nr = x_nr + 1
    x_actual_overlap = cld(x_nr * width - x, x_nr - 1)
  end

  y_nr = 1
  y_actual_overlap = 0
  while (y_actual_overlap < overlap)
    y_nr = y_nr + 1
    y_actual_overlap = cld(y_nr * width - y, y_nr - 1)
  end

  tiles = [ga[1 + i * (width - x_actual_overlap):i * (width - x_actual_overlap) + width,
              1 + j * (width - y_actual_overlap):j * (width - y_actual_overlap) + width,
              begin:end] for i = 0:x_nr - 1 for j = 0:y_nr - 1]

  return tiles
end

"Find pixels of vessels in given GeoArray"
function allVessels(ga::GeoArray, df::DataFrame)::Array{(Int, Int),1}
  (x_max, y_max, _bands) = size(ga.A)
  vessels = @pipe [(d.detect_lat, d.detect_lon, 0.0) for d in eachrow(df[(df.is_vessel .!== missing) .& (df.is_vessel .== true), :])] .|>
              LLA(_ ...) .|>
              UTMZfromLLA(wgs84) .|>
              indices(ga, (_.x, _.y)) .|>
              (xy -> (xy[1], xy[2])) |>
              filter((x, y) -> 1 <= x && x <= x_max && 1 <= y && y <= y_max, _)
  return vessels
end

"For a given tile, generate the image of the distribution given the ground truth data."
function generateImageTile(ga::GeoArray, df::DataFrame)::Array{Float32,2}
  vessels = allVessels(ga, df)
  result = zeros(Float32, x_max, y_max)
  for xy in vessels
    result[xy ...] = 1.0
  end
end

"Sample tile by applying symmetries."
function sampleTileWithSymmetry(ga::GeoArray, df::DataFrame, n::Int, r::Int)::(TyleCollection, Array{Float32,2})
  image = generateImageTile(ga, df)
  transformations = [identity rotations(ga, df, n, r) ...]
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
  x_window = x[c_x - r:c_x + r, c_y - r:c_y + r]
  rotation = recenter(rot.rot, center(x_window))
  warped = no_offset_view(warp(x_window, rotation))
  (w_x,w_y) = center(warped)
  y = copy(x)
  y[c_x - r:c_x + r, c_y - r:c_y + r] = warped[w_x - r:w_x + r, w_y - r:w_y + r]
  nans = findall(isnan, y)
  y[nans] = x[nans]
  return y
end

"Compute rotations around vessels"
function rotations(vessels::Array{Tuple{Int,Int},1}, n::Int, r::Float32)::Array{LocalRotation,1}
  phis = [0:2.0 * pi / n:2 * pi]
  [LocalRotation(v, r, Angle2d(phi)) for v in vessels for phi in phis]
end
end
