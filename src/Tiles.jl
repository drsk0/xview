module Tiles

export generateTiles
export generateImageTile

using GeoArrays
using DataFrames
using Geodesy
using Pipe

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

"For a given tile, generate the image of the distribution given the ground truth data."
function generateImageTile(ga::GeoArray, df::DataFrame) ::Array{Float32, 2}
  "get pixels for fishing vessels in tile and set them to one."
  (x_max, y_max, _bands) = size(ga.A)
  vessels = @pipe [(d.detect_lat, d.detect_lon, 0.0) for d in eachrow(df[(df.is_vessel .!== missing) .& (df.is_vessel .== true), :])] .|>
              LLA(_ ...) .|>
              UTMZfromLLA(wgs84) .|>
              indices(ga, (_.x, _.y)) |>
              filter(xy -> 1 <= xy[1] && xy[1] <= x_max && 1 <= xy[2] && xy[2] <= y_max, _)
  result = zeros(Float32, x_max, y_max)
  for xy in vessels
    result[xy ...] = 1.0
  end

  return result
end

"TODO: Sample tile by applying given symmetry."
function sampleTileWithSymmetry(ga::GeoArray)::TyleCollection
end
end


