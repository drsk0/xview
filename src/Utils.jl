module Utils

using Pipe
using Rasters

include("UNet.jl")

const Tile = Tuple{UnitRange{Int},UnitRange{Int}}

mutable struct Tiles
    empty::Vector{Tile}
    nonempty::Vector{Tile}
end

function Tiles()::Tiles
    Tiles([], [])
end

# Return all filepath pointing to a data directory.
function getDataDirs(dataDir::String)::Vector{String}
    @pipe readdir(dataDir, join = true) |>
          filter(fp -> isdir(fp) && !endswith(fp, "shoreline"), _)
end


# Apply a Unet, returning all pixel coordinates above a given threshold.
function applyU(u::UNet.Unet, rs::RasterStack, tileSize::Int)::Matrix{Float32}
    tiles = TileIterator(axes(rs[:, :, 1]), RelaxStride((tileSize, tileSize)))
    x, y = size(rs[:VV_dB])[1], size(rs[:VV_dB])[2]
    img = zeros(Float32, x, y)
    for t in tiles
        img[t...] = applyU(u, rs, t)
    end
    return img
end


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

end #Utils
