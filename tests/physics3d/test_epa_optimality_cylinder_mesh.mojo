"""Is our cylinder-mesh EPA answer the OPTIMUM? An optimality certificate.

Task #57 filed our cylinder-mesh contact NORMAL as a defect because it sits
9.4e-3 away from MuJoCo's on Jaco `reach_site_features` pose t=38 — the
cleanest case in the sweep, ONE contact on each side, `dist` agreeing to
5.7e-7 so it could not be a different feature pair. **That premise was wrong,
and this file is the gate that settles it.**

⚠⚠ MUJOCO IS NOT CONVERGED HERE, AND TIGHTENING ITS TOLERANCE MAKES IT WORSE.
Run against itself at this pose with only `ccd_tolerance` moving:

    ccd_tolerance   dist                    normal
    1e-6 (default)  -2.879689378564938e-03  (0.14009430, 0.46353334, 0.87493453)
    1e-8            -2.886381635873179e-03  (0.19142066, 0.48217779, 0.85490509)
    1e-12           -2.886904966246419e-03  (0.19390578, 0.48304316, 0.85385588)
    libccd          -2.879689378564938e-03  identical to the 1e-6 native answer

Its "converged" answer is 6.1e-2 from its own default one, and — measured
below — 4.01 degrees from the sampled optimum, against 0.42 degrees for the
default. So "agree with MuJoCo" is not a well-posed target for this quantity,
and neither is "agree with MuJoCo once it has converged".

⚠ THE TEST PRINTS AN ANGLE FROM *OUR* NORMAL, NOT FROM THE OPTIMUM — they are
close here but they are not the same quantity, and reading one as the other is
how the 0.51/0.42 discrepancy between drafts of this docstring arose.

WHAT IS WELL-POSED is the definition. For convex A and B the penetration depth
is

    depth = min over unit n of  h(n),   h(n) = h_A(n) + h_B(-n)

with `h_X` the support extent of X along a direction. That makes any candidate
(normal, depth) pair CHECKABLE without a second implementation: the depth must
equal `h` along the normal, and no direction may beat it. This file asserts
both, against a dense sweep plus deterministic local refinement.

Measured, at the frozen pose below, with BOTH engines now running the same
stopping rule (smaller `h` is better; it is what the penetration depth
minimises, and the sampled optimum is 2.880402e-03):

    ours                            h = 2.880567e-03   1.64e-07 above the optimum
    MuJoCo @ ccd_tolerance 1e-6     h = 2.883992e-03   3.43e-06 above, 0.42 deg away
    MuJoCo @ ccd_tolerance 1e-12    h = 2.964017e-03   8.36e-05 above, 4.01 deg away

i.e. at MuJoCo's own tolerance we land 21x closer to the optimum than MuJoCo
does. This is a fact about the two EPA implementations, not about the
tolerance, which is now shared.

⚠ THE NORMAL HERE IS NOT BIT-IDENTICAL TO THE ONE THE MODEL PATH PRODUCES, and
that is the fixture's doing, not a defect: the vertices below are MuJoCo's
`mesh_vert`, i.e. float32-quantised, while the engine builds its hull in
float64 from the STL. Same pose through `ReachSiteFeaturesModel` gives
(0.13339824, 0.46098917, 0.87732200) against this file's
(0.13364544, 0.46108063, 0.87723631) — 2.6e-05 apart, the size of the 7.5e-9
vertex difference propagated through. Running on MuJoCo's own numbers is the
deliberate choice: it removes "you two disagree about the shape" as an escape
route for a failure.

⚠ THE NUMBERS ABOVE MOVED ONCE ALREADY, when `ccd_tolerance` stopped being
hardcoded at 1e-8. Cylinder-vs-mesh is a SMOOTH pair, so it takes
`opt.ccd_tolerance`; a mesh-vs-mesh or mesh-vs-box pair would take `mjMINVAL`
instead via `discreteGeoms`, and this fixture would then be gating almost
nothing. If this file is ever re-pointed at a different pair, check which
branch of `_epa_tolerance` it lands in first.

⚠ OUR EPA WAS ALREADY CONVERGED — the caps are not the story either. Traced at
this pose under the old hardcoded 1e-8 it bracketed the depth to 6.6e-9 in 18
iterations, and raising `EPA_V_CAP`/`EPA_F_CAP` (36/64 -> 72/140) changed the
answer by ZERO bits. Both were checked before this conclusion was written,
because "we ran out of budget" is the explanation a capped solver invites.

⚠ THE GEOMETRY WAS RULED OUT FIRST, and had to be: two correct EPAs cannot
disagree on the same inputs. At this pose every geom's world position matches
MuJoCo's `geom_xpos` to 4e-16, every orientation matches `geom_xmat` to 3e-15,
every size to 1e-16, and mesh 2's 130 hull vertices match `mesh_vert` to
7.5e-9 — which is float32 quantisation, since MuJoCo stores `mesh_vert` as
`float`. The vertices below are MuJoCo's own, so this gate runs on ITS
geometry and cannot be passed by disagreeing about the shape.

⚠ THIS IS NOT A LICENCE TO DIVERGE FROM MUJOCO. It is a narrow finding about
one quantity that MuJoCo itself computes to ~0.5 degrees at its defaults.
Anywhere the reference is exact, parity remains the gate.

Frozen data: Jaco `reach_site_features`, `np.random.default_rng(4)` pose 38 of
the 40-pose sweep, cylinder geom 9 (r 0.035, half-length 0.009) against mesh
geom 4 (`jaco_arm/link_2`), both already in WORLD coordinates so no model has
to be built to run this.

Run with:
    pixi run mojo run -I . tests/physics3d/test_epa_optimality_cylinder_mesh.mojo
"""

from std.math import abs, sqrt, acos, cos, sin, pi
from noeira.core.tiles import RowMajor
from std.testing import assert_true, TestSuite

from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.collision.ccd_workspace import L_CCD_WS1
from noeira.physics3d.collision.ccd_workspace_host import ccd_ws_alloc
from noeira.physics3d.collision.gjk import gjk_epa
from noeira.physics3d.collision.gjk_support import (
    support_cylinder,
    support_mesh,
)
from noeira.physics3d.gpu.constants import mesh_max_edge
from noeira.physics3d.constants import GEOM_CYLINDER, GEOM_MESH

comptime DT = DType.float64
comptime NV_HULL: Int = 130
comptime L_MV = RowMajor[NV_HULL, 3]

# ⚠ NO EDGE GRAPH. `_support_mesh` hill-climbs the hull adjacency when it is
# present and falls back to the exhaustive scan when `edgeadr < 0`, which is
# how `fields/model.mojo` marks an unbuilt graph. These fixtures only need the
# tensors present and marked absent; the two paths are compared against each
# other in `test_gjk_hillclimb_support.mojo`.
comptime L_VEADR_H = RowMajor[NV_HULL]
comptime L_EDGES_H = RowMajor[mesh_max_edge(NV_HULL)]


def _no_graph_H() raises -> TensorImpl[DType.float64]:
    var t = TensorImpl[DType.float64].alloc(NV_HULL)
    for i in range(NV_HULL):
        t.data[i] = -1.0
    return t^


def _no_edges_H() raises -> TensorImpl[DType.float64]:
    var t = TensorImpl[DType.float64].alloc(mesh_max_edge(NV_HULL))
    for i in range(mesh_max_edge(NV_HULL)):
        t.data[i] = -1.0
    return t^


comptime CYL_R: Float64 = 0.035
comptime CYL_HL: Float64 = 0.009
comptime CYL_PX: Float64 = -0.09117972936009013
comptime CYL_PY: Float64 = 0.5491810220592841
comptime CYL_PZ: Float64 = 0.12953081577053305
# (x, y, z, w) — the order this engine stores quaternions in.
comptime CYL_QX: Float64 = 0.6577032646339427
comptime CYL_QY: Float64 = -0.2748479486015801
comptime CYL_QZ: Float64 = -0.5442743864180514
comptime CYL_QW: Float64 = -0.44232387809003715

# MuJoCo's own answer at its default `ccd_tolerance`, for the report below.
# NOT asserted — see the module docstring.
comptime MJ_DIST: Float64 = -0.002879689378564938
comptime MJ_NX: Float64 = 0.14009430423374564
comptime MJ_NY: Float64 = 0.4635333423892305
comptime MJ_NZ: Float64 = 0.8749345269302906

# ⚠ BOTH TOLERANCES ARE SET BY `ccd_tolerance`, NOT BY TASTE. EPA stops when
# its bound gap falls below `mjModel.opt.ccd_tolerance` = 1e-6, so `dist` — the
# closest FACE PLANE's distance, a lower bound — sits up to that far below `h`
# along the same normal, and the normal itself is only optimal to the same
# order. 2e-6 is one doubling of the stopping rule.
#
# ⚠ THIS FILE HAS BEEN WRONG ABOUT THESE TWICE, both times by pinning them to
# something other than the engine's actual stopping rule. First 1e-9, which is
# below any bracket EPA has ever had. Then 1e-7, which matched the hardcoded
# 1e-8 tolerance that `ccd_tolerance` replaced — correct for a day, then stale
# the moment the engine adopted MuJoCo's looser rule. A tolerance that is not
# derived from the thing it is measuring will drift out from under the gate.
#
# It still SEPARATES rather than merely admits: MuJoCo's own default answer is
# 3.43e-6 above the optimum here, so it fails this budget by 1.7x, and its
# 1e-12 answer fails by 42x.
comptime SELF_CONSISTENCY_TOL: Float64 = 2e-6
# No sampled direction may beat ours by more than this.
comptime OPTIMALITY_TOL: Float64 = 2e-6


def _hull() raises -> TensorImpl[DT]:
    """Mesh 2's 130 hull vertices at this pose, in WORLD, from MuJoCo."""
    var t = TensorImpl[DT].alloc(NV_HULL * 3)
    t.data[0] = -0.00569464122650451
    t.data[1] = 0.4139951581457198
    t.data[2] = 0.31391652219574295
    t.data[3] = -0.1504134671114586
    t.data[4] = 0.7502856097846216
    t.data[5] = 0.17053996050775416
    t.data[6] = 0.017530922402486687
    t.data[7] = 0.44258489223489006
    t.data[8] = 0.3054603740316222
    t.data[9] = -0.160194491312041
    t.data[10] = 0.7631237538637111
    t.data[11] = 0.1649164465382894
    t.data[12] = -0.1569327559080849
    t.data[13] = 0.7699202136274415
    t.data[14] = 0.1551159921713226
    t.data[15] = -0.15806687931349683
    t.data[16] = 0.7776748368718338
    t.data[17] = 0.13906744448682667
    t.data[18] = -0.15560094961680832
    t.data[19] = 0.7765323129592934
    t.data[20] = 0.1189338938544024
    t.data[21] = -0.1492577016486604
    t.data[22] = 0.7681888576819892
    t.data[23] = 0.1399830249838303
    t.data[24] = -0.13878155635643497
    t.data[25] = 0.7518638261690678
    t.data[26] = 0.13841046787061087
    t.data[27] = 0.03132011632700486
    t.data[28] = 0.4242444113877073
    t.data[29] = 0.2383625736201072
    t.data[30] = -0.13244944525291624
    t.data[31] = 0.7329090662349577
    t.data[32] = 0.10638383175838337
    t.data[33] = -0.1349343334278637
    t.data[34] = 0.7242953979750204
    t.data[35] = 0.09626168266329993
    t.data[36] = -0.13858370458607533
    t.data[37] = 0.7386255824997803
    t.data[38] = 0.09597987088064595
    t.data[39] = 0.027784810398025564
    t.data[40] = 0.4184034806237542
    t.data[41] = 0.23081042982274896
    t.data[42] = 0.018065819326697735
    t.data[43] = 0.36971948709242136
    t.data[44] = 0.29696712092760247
    t.data[45] = 0.019575483878018496
    t.data[46] = 0.36690635221314716
    t.data[47] = 0.29052442188826355
    t.data[48] = -0.15034868557144765
    t.data[49] = 0.7578101796142588
    t.data[50] = 0.16440949222629905
    t.data[51] = 0.015963876121987616
    t.data[52] = 0.43955009741797646
    t.data[53] = 0.30865680577428634
    t.data[54] = 0.03451707902516843
    t.data[55] = 0.38902793392243307
    t.data[56] = 0.3076499999169318
    t.data[57] = 0.03798773588808947
    t.data[58] = 0.3804175948697339
    t.data[59] = 0.2971768478074596
    t.data[60] = 0.016050086652556764
    t.data[61] = 0.3734756406043507
    t.data[62] = 0.3027575259176025
    t.data[63] = 0.013589493768425873
    t.data[64] = 0.37806075354509455
    t.data[65] = 0.3077197740878555
    t.data[66] = 0.010758852300260219
    t.data[67] = 0.38333542036661394
    t.data[68] = 0.31170303960599177
    t.data[69] = 0.04330226021666124
    t.data[70] = 0.3781979224110942
    t.data[71] = 0.28641754156395927
    t.data[72] = 0.04100922186438666
    t.data[73] = 0.392916009483332
    t.data[74] = 0.30367210411540396
    t.data[75] = -0.18929685716946965
    t.data[76] = 0.7561228121232602
    t.data[77] = 0.1613902418517001
    t.data[78] = -0.1862503425007135
    t.data[79] = 0.7504458835111694
    t.data[80] = 0.164582872702364
    t.data[81] = -0.17340183476935622
    t.data[82] = 0.7712245353979407
    t.data[83] = 0.15712204827383253
    t.data[84] = -0.1693318478001814
    t.data[85] = 0.766876068073472
    t.data[86] = 0.16136255327574878
    t.data[87] = -0.17174386175661388
    t.data[88] = 0.7770526854520273
    t.data[89] = 0.15244435493359648
    t.data[90] = -0.1920347712939872
    t.data[91] = 0.7612246910756848
    t.data[92] = 0.15712751495170932
    t.data[93] = -0.1943813515116667
    t.data[94] = 0.7655973514709215
    t.data[95] = 0.15192377344610053
    t.data[96] = -0.19864574934667684
    t.data[97] = 0.7735437066941377
    t.data[98] = 0.12518731959426876
    t.data[99] = -0.19827253913627563
    t.data[100] = 0.7728489188590761
    t.data[101] = 0.11802533720481963
    t.data[102] = -0.19626488488675853
    t.data[103] = 0.7691071603937439
    t.data[104] = 0.14593756231560898
    t.data[105] = -0.17859155779770758
    t.data[106] = 0.7837142807570389
    t.data[107] = 0.13260285833001065
    t.data[108] = -0.19843025003601827
    t.data[109] = 0.773142141536853
    t.data[110] = 0.13236290558882474
    t.data[111] = -0.17670707227603705
    t.data[112] = 0.7852594234335755
    t.data[113] = 0.1256930832841291
    t.data[114] = -0.1680920470267272
    t.data[115] = 0.7835001226071294
    t.data[116] = 0.137312679743665
    t.data[117] = -0.17857372515927783
    t.data[118] = 0.779748495082544
    t.data[119] = 0.14415260809730618
    t.data[120] = -0.1976281140039075
    t.data[121] = 0.7716474239732661
    t.data[122] = 0.1393507604818856
    t.data[123] = 0.031813499654013214
    t.data[124] = 0.4333879312881593
    t.data[125] = 0.2825983102764692
    t.data[126] = 0.03512375534553269
    t.data[127] = 0.42871572412015135
    t.data[128] = 0.2721001972694796
    t.data[129] = -0.1404028968056017
    t.data[130] = 0.75220308331491
    t.data[131] = 0.1484121501039956
    t.data[132] = -0.14907613414875415
    t.data[133] = 0.7638358554434448
    t.data[134] = 0.15218427231845036
    t.data[135] = 0.039504682443814707
    t.data[136] = 0.37665152466606105
    t.data[137] = 0.2628199409124001
    t.data[138] = 0.03593827433096025
    t.data[139] = 0.3837478467619661
    t.data[140] = 0.248594560070073
    t.data[141] = 0.011560781159810102
    t.data[142] = 0.38184109272907163
    t.data[143] = 0.24027609726986884
    t.data[144] = 0.008514281256707446
    t.data[145] = 0.3875179977685125
    t.data[146] = 0.23708347727984572
    t.data[147] = 0.014298833092508123
    t.data[148] = 0.376738957012181
    t.data[149] = 0.2445389381987382
    t.data[150] = 0.03355378857567856
    t.data[151] = 0.39450791076047775
    t.data[152] = 0.23978930311538887
    t.data[153] = 0.020909819509400573
    t.data[154] = 0.3644199297491221
    t.data[155] = 0.27647913811958497
    t.data[156] = 0.0430473803311552
    t.data[157] = 0.3763562550325672
    t.data[158] = 0.27582142583121455
    t.data[159] = 0.02069395911171143
    t.data[160] = 0.36482216768360554
    t.data[161] = 0.269303254345254
    t.data[162] = 0.0205331775003062
    t.data[163] = 0.3651217685817537
    t.data[164] = 0.28362520045593076
    t.data[165] = -0.1587824840559097
    t.data[166] = 0.738914050699341
    t.data[167] = 0.08572450436067362
    t.data[168] = -0.14543293764925969
    t.data[169] = 0.7362100397519072
    t.data[170] = 0.0886241550657059
    t.data[171] = -0.17204142657979588
    t.data[172] = 0.7239687353608414
    t.data[173] = 0.08774982235652018
    t.data[174] = -0.1973115466122985
    t.data[175] = 0.7710575279052678
    t.data[176] = 0.11114192728011604
    t.data[177] = -0.1786827562329253
    t.data[178] = 0.7363443063099624
    t.data[179] = 0.08492810789613639
    t.data[180] = -0.17530380917466043
    t.data[181] = 0.7300479166606233
    t.data[182] = 0.08572450255284648
    t.data[183] = 0.004340017319375167
    t.data[184] = 0.395296395446212
    t.data[185] = 0.31628185000400644
    t.data[186] = 0.007644162010078254
    t.data[187] = 0.3891393920618026
    t.data[188] = 0.3145862501731158
    t.data[189] = 0.026063418903712984
    t.data[190] = 0.39807292534090244
    t.data[191] = 0.3144174049666698
    t.data[192] = 0.020674493073945283
    t.data[193] = 0.4127131396337233
    t.data[194] = 0.3167417798470158
    t.data[195] = 0.03105601262475821
    t.data[196] = 0.41212552272072067
    t.data[197] = 0.31290266131017586
    t.data[198] = 0.017084135376045348
    t.data[199] = 0.426663081444473
    t.data[200] = 0.3138126169051638
    t.data[201] = 0.022257462988943766
    t.data[202] = 0.40615654193635403
    t.data[203] = 0.3164406934771364
    t.data[204] = -0.002432266834999086
    t.data[205] = 0.4079159881917085
    t.data[206] = 0.31594183656872243
    t.data[207] = 0.0009468245937405417
    t.data[208] = 0.4016193295510241
    t.data[209] = 0.3167383506842577
    t.data[210] = 0.0450656803348663
    t.data[211] = 0.4068689201799758
    t.data[212] = 0.283436986102112
    t.data[213] = 0.04232012637751682
    t.data[214] = 0.4146499855346576
    t.data[215] = 0.2760247976041769
    t.data[216] = 0.037435742205047806
    t.data[217] = 0.4208955249861893
    t.data[218] = 0.29329444629420953
    t.data[219] = 0.04174910887786287
    t.data[220] = 0.40686764923863517
    t.data[221] = 0.2985323152370675
    t.data[222] = 0.047622442744757326
    t.data[223] = 0.39038286992805304
    t.data[224] = 0.28576409090265464
    t.data[225] = -0.13721003933972475
    t.data[226] = 0.7498455922856113
    t.data[227] = 0.12956668712372116
    t.data[228] = 0.044457646279481455
    t.data[229] = 0.39871442891480724
    t.data[230] = 0.25461944290245103
    t.data[231] = 0.047448373412798645
    t.data[232] = 0.3982559486639114
    t.data[233] = 0.27390913447022225
    t.data[234] = 0.046893412759346054
    t.data[235] = 0.38586483633604285
    t.data[236] = 0.26662252632277983
    t.data[237] = 0.042992741010081784
    t.data[238] = 0.3846940882557185
    t.data[239] = 0.25611861059533697
    t.data[240] = 0.038666972176037026
    t.data[241] = 0.3974793260640861
    t.data[242] = 0.24305396909576488
    t.data[243] = 0.035985803372557745
    t.data[244] = 0.4118274198442897
    t.data[245] = 0.2394606329783811
    t.data[246] = 0.03532330789483197
    t.data[247] = 0.42475172885744933
    t.data[248] = 0.2532543198960984
    t.data[249] = 0.035936862256134725
    t.data[250] = 0.42629039685957465
    t.data[251] = 0.2632561477434835
    t.data[252] = 0.043749729279085006
    t.data[253] = 0.4085355510348021
    t.data[254] = 0.26168435538789675
    t.data[255] = 0.033719780024351706
    t.data[256] = 0.42421547607915755
    t.data[257] = 0.24517602192771096
    t.data[258] = 0.040222507722172576
    t.data[259] = 0.4110935322122683
    t.data[260] = 0.2494825969175704
    t.data[261] = 0.023641858377530384
    t.data[262] = 0.4420457393923604
    t.data[263] = 0.29528242365545715
    t.data[264] = -0.14303846188580804
    t.data[265] = 0.7491473061483893
    t.data[266] = 0.1633040910428671
    t.data[267] = 0.026697656015082824
    t.data[268] = 0.43066592716810964
    t.data[269] = 0.30676588661430937
    t.data[270] = -0.1417361667247942
    t.data[271] = 0.7511632291512889
    t.data[272] = 0.15649047602803787
    t.data[273] = 0.03265585640893606
    t.data[274] = 0.42530927924202555
    t.data[275] = 0.30145500231579675
    t.data[276] = 0.027582721616109862
    t.data[277] = 0.4382195783946221
    t.data[278] = 0.29044051564225126
    t.data[279] = -0.1424489153771256
    t.data[280] = 0.7560958085443802
    t.data[281] = 0.10837199100397392
    t.data[282] = -0.14075380012448058
    t.data[283] = 0.7485408232917938
    t.data[284] = 0.1006630672951299
    t.data[285] = -0.13345898472032522
    t.data[286] = 0.7383075784017161
    t.data[287] = 0.1112259967281182
    t.data[288] = -0.1351465994554641
    t.data[289] = 0.7445043679842691
    t.data[290] = 0.119068161896311
    t.data[291] = -0.1483256172434267
    t.data[292] = 0.768218465262008
    t.data[293] = 0.11925491916218803
    t.data[294] = -0.1514282723642015
    t.data[295] = 0.7663748902729681
    t.data[296] = 0.10149993946032615
    t.data[297] = -0.14495318773548718
    t.data[298] = 0.7636184306269381
    t.data[299] = 0.12564163230314943
    t.data[300] = 0.01852881543939329
    t.data[301] = 0.36885673223170856
    t.data[302] = 0.25572878223590656
    t.data[303] = 0.03644230423102339
    t.data[304] = 0.37847819977023367
    t.data[305] = 0.25571101562359944
    t.data[306] = 0.016645290911035185
    t.data[307] = 0.3723665286134189
    t.data[308] = 0.24974257306695719
    t.data[309] = 0.019892176124115687
    t.data[310] = 0.3663162235169289
    t.data[311] = 0.2623156909339859
    t.data[312] = -0.15968160593502878
    t.data[313] = 0.7009371393733612
    t.data[314] = 0.13040424990723798
    t.data[315] = -0.19580175744775496
    t.data[316] = 0.7682441608460268
    t.data[317] = 0.10469933178920107
    t.data[318] = -0.19378602469319217
    t.data[319] = 0.7644880071846797
    t.data[320] = 0.09890892636555702
    t.data[321] = -0.19132578601211028
    t.data[322] = 0.7599035541965846
    t.data[323] = 0.09394638411089962
    t.data[324] = -0.1664241067803261
    t.data[325] = 0.7566422617902115
    t.data[326] = 0.0863370142863123
    t.data[327] = -0.15874933440768765
    t.data[328] = 0.7467941951184015
    t.data[329] = 0.08494334158822395
    t.data[330] = -0.15146423283503052
    t.data[331] = 0.7529638964429656
    t.data[332] = 0.08860686471016681
    t.data[333] = -0.18207607856338123
    t.data[334] = 0.7426674858334699
    t.data[335] = 0.0853844999782033
    t.data[336] = -0.18538023145744348
    t.data[337] = 0.7488245005636499
    t.data[338] = 0.0870800943783997
    t.data[339] = -0.18849479050173928
    t.data[340] = 0.7546282277212518
    t.data[341] = 0.08996341354445576
    t.data[342] = -0.16880960242117687
    t.data[343] = 0.7783062376492189
    t.data[344] = 0.10185996478783123
    t.data[345] = -0.17726469942758544
    t.data[346] = 0.7733541416874407
    t.data[347] = 0.09890892904081232
    t.data[348] = -0.17632072640460555
    t.data[349] = 0.7822597036018728
    t.data[350] = 0.11107211907358754
    t.data[351] = -0.16559252170421185
    t.data[352] = 0.7829623424632357
    t.data[353] = 0.11414405267610456
    t.data[354] = -0.16118554089944548
    t.data[355] = 0.7692570047742051
    t.data[356] = 0.09376532338245877
    t.data[357] = -0.17070791190267726
    t.data[358] = 0.7661109438094469
    t.data[359] = 0.09097345878004691
    t.data[360] = 0.020110469635116604
    t.data[361] = 0.45641030029044505
    t.data[362] = 0.26687517962698515
    t.data[363] = -0.12345776377181197
    t.data[364] = 0.7243739886598837
    t.data[365] = 0.14088759412746144
    t.data[366] = -0.12470748244015731
    t.data[367] = 0.7262666645129173
    t.data[368] = 0.14709943506340206
    t.data[369] = 0.021500283638754072
    t.data[370] = 0.4542565670831645
    t.data[371] = 0.2607792084639461
    t.data[372] = -0.12619970621502374
    t.data[373] = 0.7277135065348223
    t.data[374] = 0.15340431305926466
    t.data[375] = 0.018220244103982555
    t.data[376] = 0.4585987864514607
    t.data[377] = 0.2728508777234535
    t.data[378] = -0.12243132191359431
    t.data[379] = 0.7220252257195447
    t.data[380] = 0.1347913683912116
    t.data[381] = 0.022386630242101427
    t.data[382] = 0.45216886134765477
    t.data[383] = 0.2545671125211507
    t.data[384] = -0.12165245794206142
    t.data[385] = 0.719240081302336
    t.data[386] = 0.12881559469376067
    t.data[387] = 0.022767492216101234
    t.data[388] = 0.4501253615178098
    t.data[389] = 0.24826216022523753
    return t^


def _h(
    verts: List[Float64],
    nx: Float64,
    ny: Float64,
    nz: Float64,
) -> Float64:
    """Support extent of the Minkowski difference `cylinder (-) mesh` along n.

    `h(n) = h_cyl(n) + h_mesh(-n)`, so `min over n of h(n)` IS the penetration
    depth. Built from the same support functions the narrow phase uses, which
    is deliberate: a support bug would then move the certificate too, and the
    gate would go quiet. That is covered separately — `test_mesh_support` and
    `test_narrow_phase_pairs` anchor the support functions themselves against
    MuJoCo — so this file is free to assume them and check only the SEARCH.
    """
    var sc = support_cylinder[DT](
        Scalar[DT](nx), Scalar[DT](ny), Scalar[DT](nz),
        Scalar[DT](CYL_PX), Scalar[DT](CYL_PY), Scalar[DT](CYL_PZ),
        Scalar[DT](CYL_QX), Scalar[DT](CYL_QY), Scalar[DT](CYL_QZ),
        Scalar[DT](CYL_QW),
        Scalar[DT](CYL_R), Scalar[DT](CYL_HL),
    )
    var sm = support_mesh[DT](
        Scalar[DT](-nx), Scalar[DT](-ny), Scalar[DT](-nz),
        Scalar[DT](0), Scalar[DT](0), Scalar[DT](0),
        Scalar[DT](0), Scalar[DT](0), Scalar[DT](0), Scalar[DT](1),
        verts, 0, NV_HULL,
    )
    return (
        Float64(sc[0]) * nx + Float64(sc[1]) * ny + Float64(sc[2]) * nz
        - (Float64(sm[0]) * nx + Float64(sm[1]) * ny + Float64(sm[2]) * nz)
    )


def _epa() raises -> Tuple[Float64, Float64, Float64, Float64]:
    """Our narrow phase on the frozen pair: (dist, nx, ny, nz)."""
    var mv = _hull()
    var _ng = _no_graph_H()
    var _ne = _no_edges_H()
    var ws = ccd_ws_alloc[DT]()
    var r = gjk_epa[DT](
        GEOM_CYLINDER,
        Scalar[DT](CYL_PX), Scalar[DT](CYL_PY), Scalar[DT](CYL_PZ),
        Scalar[DT](CYL_QX), Scalar[DT](CYL_QY), Scalar[DT](CYL_QZ),
        Scalar[DT](CYL_QW),
        Scalar[DT](CYL_R), Scalar[DT](CYL_HL),
        Scalar[DT](0), Scalar[DT](0), Scalar[DT](0),
        mv.tt["cpu", NV_HULL, 3](), _ng.tt["cpu", NV_HULL](), _ne.tt["cpu", mesh_max_edge(NV_HULL)](), 0, 0,
        GEOM_MESH,
        Scalar[DT](0), Scalar[DT](0), Scalar[DT](0),
        Scalar[DT](0), Scalar[DT](0), Scalar[DT](0), Scalar[DT](1),
        Scalar[DT](0), Scalar[DT](0),
        Scalar[DT](0), Scalar[DT](0), Scalar[DT](0),
        0, NV_HULL,
        ws.tt_dyn["cpu", L_CCD_WS1](L_CCD_WS1()), 0,
    )
    return (Float64(r[0]), Float64(r[4]), Float64(r[5]), Float64(r[6]))


def _verts() raises -> List[Float64]:
    var t = _hull()
    var v = List[Float64]()
    for i in range(NV_HULL * 3):
        v.append(Float64(t.data[i]))
    return v^


def test_reported_depth_is_the_extent_along_the_reported_normal() raises:
    """`dist` and `normal` must describe the SAME direction.

    A contact whose depth was measured along one direction and reported with
    another is wrong even if both are individually plausible, and nothing
    downstream can see it: the solver takes the pair on trust and builds
    `efc_J` from the normal and `aref` from the depth.
    """
    print("=== reported depth is the extent along the reported normal ===")
    var e = _epa()
    var dist = e[0]
    print("  our dist =", dist, " normal = (", e[1], e[2], e[3], ")")
    assert_true(
        dist < 0.0,
        "the frozen Jaco t=38 pair is PENETRATING (MuJoCo reports"
        " dist = -2.8797e-3); a non-negative dist means the pair was not"
        " detected at all and every assertion below is vacuous",
    )
    var v = _verts()
    var h = _h(v, e[1], e[2], e[3])
    print("  h(n) =", h, "  -dist =", -dist, "  |diff| =", abs(h + dist))
    assert_true(
        abs(h + dist) <= SELF_CONSISTENCY_TOL,
        "the reported penetration depth is not the support extent along the"
        " reported normal: h(n) = " + String(h) + " but -dist = "
        + String(-dist),
    )
    print("  PASS")


def test_our_normal_is_the_minimising_direction() raises:
    """No direction beats ours — the optimality half of the certificate.

    ⚠ THE SEARCH IS GLOBAL FIRST, THEN LOCAL, AND BOTH ARE NEEDED. A local
    probe around our own answer would only show we sit at the bottom of SOME
    basin, which a prematurely-terminated EPA also satisfies — MuJoCo does
    exactly that here at `ccd_tolerance` 1e-12, locally optimal and 4.01 deg
    away. A global sweep alone is too coarse to resolve a 0.5 deg error, which
    is the size of the difference this task was filed over. So: 40 000
    near-uniform directions to rule out another basin, then rings of shrinking
    angular radius about OUR normal to rule out a better answer next door.

    ⚠ THE COMPARISON IS ON `h`, NOT ON AN ANGLE TO A RECONSTRUCTED OPTIMUM.
    The first draft refined a "true optimum" and measured our angle to it, and
    the refinement converged WORSE than the answer it was judging (h 0.0028857
    against our 0.0028803) — so the gate failed us for being 0.98 deg from a
    point that was itself off. `h` is the quantity being minimised and needs no
    reconstruction.
    """
    print("=== our normal minimises the Minkowski support extent ===")
    var e = _epa()
    var v = _verts()
    var h_ours = _h(v, e[1], e[2], e[3])

    # ---- global: Fibonacci sphere, deterministic, no RNG ------------------
    var N = 40000
    var hb = 1e30
    var bx = 0.0
    var by = 0.0
    var bz = 1.0
    var ga = pi * (1.0 + sqrt(5.0))
    for i in range(N):
        var t = (Float64(i) + 0.5) / Float64(N)
        var phi = acos(1.0 - 2.0 * t)
        var th = ga * (Float64(i) + 0.5)
        var dx = cos(th) * sin(phi)
        var dy = sin(th) * sin(phi)
        var dz = cos(phi)
        var hv = _h(v, dx, dy, dz)
        if hv < hb:
            hb = hv
            bx = dx
            by = dy
            bz = dz
    print("  global sweep (", N, "dirs): best h =", hb)

    # ---- local: rings about OUR normal, shrinking angular radius ----------
    var nx = e[1]
    var ny = e[2]
    var nz = e[3]
    var ax = 0.0
    var ay = 0.0
    var az = 1.0
    if abs(nz) > 0.9:
        ax = 1.0
        az = 0.0
    var t1x = ay * nz - az * ny
    var t1y = az * nx - ax * nz
    var t1z = ax * ny - ay * nx
    var l1 = sqrt(t1x * t1x + t1y * t1y + t1z * t1z)
    t1x /= l1
    t1y /= l1
    t1z /= l1
    var t2x = ny * t1z - nz * t1y
    var t2y = nz * t1x - nx * t1z
    var t2z = nx * t1y - ny * t1x

    var h_local = 1e30
    var span = 0.08
    for _ring in range(9):
        for k in range(180):
            var a = 2.0 * pi * Float64(k) / 180.0
            var ca = cos(a) * span
            var sa = sin(a) * span
            var dx = nx + ca * t1x + sa * t2x
            var dy = ny + ca * t1y + sa * t2y
            var dz = nz + ca * t1z + sa * t2z
            var ln = sqrt(dx * dx + dy * dy + dz * dz)
            var hv = _h(v, dx / ln, dy / ln, dz / ln)
            if hv < h_local:
                h_local = hv
        span *= 0.4
    print("  local rings about ours: best h =", h_local)
    if h_local < hb:
        hb = h_local

    print("  ours:                        h =", h_ours)
    var mjn = sqrt(MJ_NX * MJ_NX + MJ_NY * MJ_NY + MJ_NZ * MJ_NZ)
    var h_mj = _h(v, MJ_NX / mjn, MJ_NY / mjn, MJ_NZ / mjn)
    var mdot = (MJ_NX * nx + MJ_NY * ny + MJ_NZ * nz) / mjn
    if mdot > 1.0:
        mdot = 1.0
    print("  MuJoCo @ its default:        h =", h_mj, " (",
          h_mj - h_ours, "above ours,", acos(mdot) * 180.0 / pi,
          "deg away) — reported, NOT asserted")

    assert_true(
        h_ours <= hb + OPTIMALITY_TOL,
        "a sampled direction beats our contact normal: h(ours) = "
        + String(h_ours) + " against h(best) = " + String(hb)
        + ". EPA settled on a face that is on the Minkowski boundary but is"
        " not the closest one",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
