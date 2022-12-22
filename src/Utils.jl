module Utils

using Pipe
using Rasters
using TiledIteration
using Base.Threads
using DataFrames
using ..UNet

const Tile = Tuple{UnitRange{Int},UnitRange{Int}}
export Tile

mutable struct Tiles
    empty::Vector{Tile}
    nonempty::Vector{Tuple{Tile, Vector{CartesianIndex}}}
end
export Tiles

function Tiles()::Tiles
    Tiles([], [])
end

# Return all filepath pointing to a data directory.
function getDataDirs(dataDir::String)::Vector{String}
    @pipe readdir(dataDir, join = true) |>
          filter(fp -> isdir(fp) && !endswith(fp, "shoreline"), _)
end
export getDataDirs

# Apply a Unet, returning all pixel coordinates above a given threshold.
function applyU(u::Unet, rs::RasterStack, tileSize::Int)::Matrix{Float32}
    tiles = TileIterator(axes(rs[:, :, 1]), RelaxStride((tileSize, tileSize)))
    x, y = size(rs[:VV_dB])[1], size(rs[:VV_dB])[2]
    img = zeros(Float32, x, y)
    for t in tiles
        img[t...] = applyU(u, rs, t)
    end
    return img
end
export applyU

function applyU(u, rs::RasterStack, t::Tile)::Matrix{Float32}
    tileSize = length(t[1])
    u(
        reshape(
            hcat(
                rs[:VV_dB].data[t..., 1],
                rs[:VH_dB].data[t..., 1],
                rs[:bathymetry_processed].data[t..., 1],
            ),
            tileSize,
            tileSize,
            3,
            1,
        ),
    )[
        :,
        :,
        1,
        1,
    ]
end
export applyU

function partitionTiles(fp::String, objects::SubDataFrame, tileSize::Int)::Utils.Tiles
    rs = Raster(joinpath(fp, "VV_dB.tif"), lazy=true)
    vs = [CartesianIndex(v.detect_scene_column, v.detect_scene_row) for v in eachrow(objects)]
    tiles = TileIterator(axes(@view rs[:, :, 1]), RelaxStride((tileSize, tileSize)))

    nonempty = []
    empty = []
    for t ∈ tiles
        os = [v for v ∈ vs if (v.I[1] ∈ t[1] && v.I[2] ∈ t[2])]
        if !isempty(os)
            push!(nonempty, (t, os))
        else
            push!(empty, t)
        end
    end

    return Utils.Tiles(empty, nonempty)
end
export partitionTiles

end #Utils
