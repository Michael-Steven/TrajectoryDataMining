using Serialization
include("utils.jl")


cellsize = 100.0
cityname = "geolife"
region = SpatialRegion(cityname,
                       116.25, 39.8,
                       116.5, 40.1,
                       cellsize, cellsize,
                       100, # minfreq
                       40_000, # maxvocab_size
                       10, # k
                       4) # vocab_start
paramfile = "../../data/$(region.name)-param-cell$(Int(cellsize))"
region = deserialize(paramfile)
#loadregion!(region, joinpath("../data", paramfile))

rate = 0.5
do_split = true
querydbfile = "querydb.h5"
createQueryDB("../../data/geolife.h5", 70_000, 1000, 100_000,
              (x, y)->(x, y),
              (x, y)->(x, y);
              do_split=do_split,
              querydbfile=querydbfile)
# (x,y)->downsampling(x, y, rate),(x,y)->downsampling(x, y, rate);
createTLabel(region, querydbfile; tfile="exp1-trj.t",labelfile="exp1-trj.label")
