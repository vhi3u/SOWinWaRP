# sowinwarp_nested_bsose_closure.jl -- examples/sowinwarp_nested.jl with B-SOSE's closure instead of CATKE,
# for two years, as 2Y_nested_bsose_conv_hj:
#
#     julia --project=. examples/sowinwarp_nested_bsose_closure.jl
#
# Everything else is sowinwarp_nested.jl's own switches (B-SOSE parent, B-SOSE bathymetry, open-boundary
# sea ice, penetrating shortwave, grounding off), with Hammond & Jones as the coastal freshwater
# (tools/hammond_jones_freshwater.jl writes coastal_freshwater_hj.nc) in place of the B-SOSE residual,
# whose polynya hotspots track the ice term rather than melt.
#
# CLOSURE "bsose_conv" is described where sowinwarp_nested.jl builds it ("closure"): B-SOSE's Laplacian
# horizontal and constant vertical backgrounds plus convective adjustment, a stand-in for MITgcm's KPP.
#
# This file only sets the ARM_ values below and includes sowinwarp_nested.jl, so the two cannot drift
# apart. RUN_NAME and END_DATE given on the command line still win, as in sowinwarp_nested.jl; pass them
# per command, never exported.

const ARM_CLOSURE                 = "bsose_conv"
const ARM_COASTAL_FRESHWATER_FILE = "coastal_freshwater_hj.nc"
const ARM_RUN_NAME                = "2Y_nested_bsose_conv_hj"
const ARM_END_DATE                = "2016-06-01T12:00"

include(joinpath(@__DIR__, "sowinwarp_nested.jl"))
