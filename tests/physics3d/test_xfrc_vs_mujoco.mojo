"""`Data.xfrc_applied` (an Euler integrator built with `XFRC=True`) against
MuJoCo's `xfrc_applied`, float64, CPU.

    pixi run mojo run -I . tests/physics3d/test_xfrc_vs_mujoco.mojo

The pass is `mj_xfrcAccumulate` (`noeira/physics3d/dynamics/xfrc.mojo`): per
body a world force at its CoM and a world torque, added to the smooth force.
The reference numbers between the `xfrc-reference` markers are MuJoCo 3.12's,
written by `tests/physics3d/xfrc_reference.py`, which reads the fixtures from
THIS file, so the two sides cannot step different models.

WHAT IS CHECKED, AND THE BARS (written before the first run)

1. A rotated free-root chain in the air, gravity with three components, a
   wrench on each of its three bodies (a torque alone, a force and a torque, a
   force alone). After 1 and 10 steps, qpos and qvel within 1e-13 and 1e-11
   of MuJoCo. Its CONTROL is the same file with no wrench, same bars: if it
   fails, the gap is not the wrench. Vacuity: MuJoCo's two answers differ by
   more than 1e-5.
2. Zero is nothing: an `XFRC=True` build with an all-zero buffer gives the
   `XFRC=False` build's qpos and qvel bit for bit.
3. `xfrc_at_point` (a force at a world point of the body) against a second
   MuJoCo path, `mj_applyFT` into `qfrc_applied`: one step, 1e-13.
4. A box resting on a plane, pushed and twisted under its friction limit: the
   wrench must reach the constraint solver's smooth acceleration. 1 and 20
   steps.
5. The G1 (mjlab's model, the one SuperMimic trains on) with a force at each
   hand's capsule centre and a wrench on the torso: in the air, and with its
   feet in the floor. 1 and 5 steps.

   For 4 and 5 the engines differ by their contact solves before any wrench,
   so the bar is relative: the wrench leg's distance from MuJoCo is at most
   10 times the control leg's (floor 1e-12), AND at most a thousandth of what
   the wrench does (MuJoCo's two answers apart). A wrench that missed the
   solver, a wrong point or a wrong frame is an error of the size of the
   effect.
6. The CPU env reads the buffer (`Phyics3dEnv[..., XFRC=True]`, HalfCheetah,
   5 control steps): a wrench moves the state; `set_xfrc_at_point` equals the
   record it stands for within 1e-12; an env built without `XFRC` ignores
   `set_xfrc` and equals the `XFRC=True` env with no wrench, bit for bit.
   No MuJoCo here: the integrator is the one gated above.
"""

from std.math import abs
from std.testing import assert_true, TestSuite

from noeira.core.cont_action import ContAction
from noeira.envs.half_cheetah.half_cheetah_config import HalfCheetahConfig
from noeira.envs.half_cheetah.half_cheetah_xml import HalfCheetahModel
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.physics3d.parser.full_parser import parse_xml_full
from noeira.physics3d.parser.expander import expand_mjcf
from noeira.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
    read_model_source,
)
from noeira.physics3d.fields import Data, Model, DynDims
from noeira.physics3d.dynamics.actuation import apply_actions_fields
from noeira.physics3d.dynamics.xfrc import XFRC_STRIDE, xfrc_at_point
from noeira.physics3d.integrator.euler import EulerIntegrator
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.studio.stepping import (
    STUDIO_MAX_CONDIM, STUDIO_NOSLIP, studio_cone_of, studio_uses_implicit,
)
from noeira.physics3d.types import ConeType

comptime DT = DType.float64

comptime G1_XML = String("noeira/envs/robots/assets/unitree_g1_mjlab.xml")

comptime XML_CHAIN = String(
    """<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.004" gravity="1.2 -2.5 -9.81"/>
  <worldbody>
    <body name="root" pos="0.1 -0.2 1.3" euler="0.3 -0.4 0.5">
      <freejoint/>
      <geom type="box" size="0.12 0.08 0.05" mass="2.5"/>
      <body name="link" pos="0.2 0.05 -0.03" euler="0 0.7 0">
        <joint name="j1" type="hinge" axis="0 1 0" damping="0.02"/>
        <geom type="capsule" fromto="0 0 0 0.3 0.02 0.01" size="0.03" mass="1.1"/>
        <body name="tip" pos="0.3 0.02 0.01" euler="0.2 0 0">
          <joint name="j2" type="hinge" axis="1 0 0" damping="0.01"/>
          <geom type="sphere" size="0.04" mass="0.4"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""
)

comptime XML_BOX = String(
    """<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.004"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="box" pos="0.3 -0.1 0.099" euler="0 0 0.4">
      <freejoint/>
      <geom name="boxg" type="box" size="0.2 0.15 0.1" mass="2"/>
    </body>
  </worldbody>
</mujoco>"""
)

# Wrenches: one row of 8 per body: body id (MuJoCo's, world = 0), kind, six
# numbers. kind 0: the record itself (force, torque). kind 1: a force and the
# world point it acts at (`xfrc_at_point`, from the start state's kinematics).
# The same tables are read by `xfrc_reference.py`.
# <xfrc-wrenches>
def _w_chain() -> List[Float64]:
    return [
        1.0, 0.0, 0.0, 0.0, 0.0, 0.3, -0.2, 0.5,
        2.0, 0.0, 1.5, -2.0, 3.0, 0.1, 0.25, -0.15,
        3.0, 0.0, -0.7, 0.4, 1.1, 0.0, 0.0, 0.0,
    ]


def _w_chain_point() -> List[Float64]:
    return [3.0, 1.0, 0.9, -1.3, 2.1, 0.45, -0.05, 1.2]


def _w_box() -> List[Float64]:
    return [1.0, 0.0, 3.0, -2.0, 0.0, 0.2, 0.0, 0.4]


def _w_g1_names() -> List[String]:
    return ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]


def _w_g1_forces() -> List[Float64]:
    """Hands: the force (its point is the hand capsule's centre at the
    start state). Torso: the record (force, torque)."""
    return [
        5.0, -8.0, -20.0, 0.0, 0.0, 0.0,
        -3.0, 6.0, -15.0, 0.0, 0.0, 0.0,
        10.0, 0.0, 0.0, 0.0, 2.0, -1.0,
    ]
# </xfrc-wrenches>

comptime CHAIN_STEPS_A = 1
comptime CHAIN_STEPS_B = 10
comptime BOX_STEPS_A = 1
comptime BOX_STEPS_B = 20
comptime G1_STEPS_A = 1
comptime G1_STEPS_B = 5
comptime G1_AIR_DZ = 0.5

# <xfrc-reference>
# MuJoCo 3.12.0, written by tests/physics3d/xfrc_reference.py. Do not edit by hand.


def _ref_chain_zero() -> List[Float64]:
    """(qpos, qvel) after the first step count, then after the second."""
    return [
        0.1000192, -0.20004000000000002, 1.29984304, 0.9462808319656861,
        0.0933065937729005, -0.2265663068902134, 0.2109838268563661, -9.094947017729282e-19,
        -1.8280904931291798e-18, 0.004799999999999996, -0.010000000000000002, -0.039240000000000004,
        4.263256414560601e-17, 1.4210854715202004e-16, -2.1316282072803006e-17, -2.2737367544323206e-16,
        -4.570226232822949e-16, 0.101056, -0.20220000000000002, 1.2913672,
        0.9462808319656861, 0.0933065937729005, -0.2265663068902134, 0.2109838268563661,
        -7.958078640513122e-17, -1.0201913227075494e-16, 0.04799999999999995, -0.10000000000000006,
        -0.3924, 4.689582056016661e-16, 2.088995643134695e-15, -1.4921397450962102e-16,
        -3.723243935382925e-15, -3.731909964959176e-15,
    ]


def _ref_chain_wrench() -> List[Float64]:
    """(qpos, qvel) after the first step count, then after the second."""
    return [
        0.1000296768267432, -0.20004165345480918, 1.299842273015415, 0.9462573318030444,
        0.09365535052978474, -0.22657856310851984, 0.21092153058852062, 0.0003719353663667427,
        -0.0005643558935887585, 0.007419206685798982, -0.01041336370228985, -0.03943174614622012,
        0.15775723082224646, -0.04815842965490983, -0.06593227856036378, 0.09298384159168567,
        -0.14108897339718962, 0.10162861569051973, -0.20229003868152146, 1.2913285544151174,
        0.9448512788876386, 0.1122435839791186, -0.22712861073854615, 0.20753320896640456,
        0.01986279003755506, -0.020971731596623508, 0.07400593265460817, -0.10395396811587669,
        -0.39403415155912264, 1.5481057437028363, -0.46418055194733165, -0.6528773509228305,
        0.8880593161819564, -0.7770871177394132,
    ]


def _ref_chain_point() -> List[Float64]:
    """(qpos, qvel) after the first step count, then after the second."""
    return [
        0.10001962003228373, -0.20004037164711497, 1.2998444842215788, 0.9462797147096859,
        0.09328058244323253, -0.22659560067097753, 0.21096887975378115, -2.7493342053181436e-05,
        -0.021705431980515832, 0.004905008070932807, -0.010092911778736361, -0.0388789446053173,
        -0.017038399568530097, -0.011939984579878072, -0.002640919961285988, -0.006873335513295359,
        -5.426357995128958,
    ]


def _ref_box_zero() -> List[Float64]:
    """(qpos, qvel) after the first step count, then after the second."""
    return [
        0.3, -0.1, 0.09903952, 0.9800665778412416,
        1.4205631350463977e-20, 2.8796240354756637e-21, 0.19866933079506122, -3.308663212881093e-19,
        6.19826830246917e-19, 0.00988000000000001, -7.217668738113656e-18, -6.537077266771869e-19,
        -3.119868078765204e-20, 0.3, -0.1, 0.09981682977292133,
        0.9800665778412416, 4.190078711645376e-19, 8.493710044217671e-20, 0.19866933079506122,
        -1.9212284493791885e-19, 1.556873179162499e-19, 0.0027148203847771634, -9.99932713279103e-19,
        -9.650806655970885e-19, 2.6596035674836964e-19,
    ]


def _ref_box_wrench() -> List[Float64]:
    """(qpos, qvel) after the first step count, then after the second."""
    return [
        0.30000072514881526, -0.10000059550447465, 0.09903952, 0.980066445163857,
        1.0488133364061103e-06, 3.650028401040575e-07, 0.19866998530892205, 0.0001812872038237874,
        -0.0001488761186627083, 0.009880000000000009, 0.0005502108836939346, 7.468002036521102e-05,
        0.0003339130434782611, 0.30004718859192486, -0.10003585204680666, 0.09981691281960207,
        0.9800574565771796, 3.955591397411691e-05, 1.3903758106082443e-05, 0.19871431767611616,
        0.0007900785925927491, -0.0005709167242333496, 0.0027158859722852037, 0.0008133821455351326,
        0.00011430190091832008, 0.001594722791887863,
    ]


def _ref_g1air_zero() -> List[Float64]:
    """(qpos, qvel) after the first step count, then after the second."""
    return [
        -3.612753926542082e-20, 3.175552593100958e-21, 1.29275475, 1.0,
        1.425718500559863e-19, 0.0, 0.0, 2.220446049250313e-19,
        8.326672684688674e-20, -8.326672684688674e-20, 3.0531133177191807e-20, -4.4408920985006264e-20,
        -2.5110782469122783e-19, 2.3314683517128286e-19, 1.5508427875232654e-19, 9.36750677027476e-20,
        1.9428902930940242e-20, 0.0, 1.4670047871390066e-19, 1.170938346284345e-20,
        1.3895135042574226e-19, 4.996003610813205e-19, -3.108624468950439e-19, -3.3306690738754697e-19,
        3.5874081483200373e-19, 8.881784197001253e-19, 3.608224830031759e-19, -1.4210854715202004e-18,
        -9.151750537950143e-20, -2.1094237467877974e-19, 6.550315845288423e-19, -7.61196661258623e-19,
        1.7763568394002505e-19, 5.495603971894525e-19, -7.105427357601002e-19, 2.190251018746592e-19,
        -7.225507853084164e-18, 6.351105186201916e-19, -0.04905, -2.011845551264102e-17,
        -5.329070518200751e-17, 2.7614629333205797e-18, 4.4408920985006264e-17, 1.6653345369377347e-17,
        -1.6653345369377347e-17, 6.106226635438361e-18, -8.881784197001253e-18, -5.0221564938245566e-17,
        4.662936703425657e-17, 3.101685575046531e-17, 1.8735013540549518e-17, 3.885780586188048e-18,
        0.0, 2.934009574278013e-17, 2.3418766925686897e-18, 2.779027008514845e-17,
        9.992007221626409e-17, -6.217248937900877e-17, -6.661338147750939e-17, 7.174816296640074e-17,
        1.7763568394002506e-16, 7.216449660063518e-17, -2.842170943040401e-16, -1.8303501075900285e-17,
        -4.218847493575595e-17, 1.3100631690576846e-16, -1.522393322517246e-16, 3.552713678800501e-17,
        1.099120794378905e-16, -1.4210854715202004e-16, 4.380502037493184e-17, -4.92975135850025e-19,
        5.424326359496343e-21, 1.2893212500000002, 1.0, 1.462845812479021e-18,
        0.0, 0.0, 1.815214645262131e-18, 9.509754095304857e-19,
        -1.131733595727269e-18, 5.218048215738236e-19, -5.329070518200752e-19, -3.773806572823283e-18,
        1.9262369477246467e-18, 2.0348306373207948e-18, 1.5369649997154513e-18, 4.579669976578771e-19,
        -1.332267629550188e-19, 2.1896300854964027e-18, 7.042977312465838e-20, 1.3985340663325017e-18,
        5.906386491005833e-18, -4.157785227221212e-18, -4.762856775641922e-18, 5.363938460067886e-18,
        1.0302869668521452e-17, 5.4012350148013864e-18, -1.7053025658242406e-17, -1.363996806127976e-18,
        -3.1752378504279478e-18, 9.803269307440133e-18, -1.1061811189261307e-17, 3.3750779948604763e-18,
        8.310019339319297e-18, -9.947598300641403e-18, 3.228764911683557e-18, -3.2483251074520196e-17,
        1.322952150787178e-19, -0.24525000000000002, -7.573802696114739e-17, -1.687538997430238e-16,
        1.360456886034811e-17, 1.099120794378905e-16, 5.953570969552402e-17, -7.591149930874508e-17,
        3.608224830031759e-17, -3.552713678800501e-17, -2.5153024358275643e-16, 1.1990408665951691e-16,
        1.3190837311327642e-16, 1.0227929614359255e-16, 2.942091015256665e-17, -8.881784197001253e-18,
        1.4597057688091358e-16, 4.909267437014364e-18, 8.895661984809066e-17, 3.841371665203042e-16,
        -2.753353101070388e-16, -3.1308289294429413e-16, 3.511774204767448e-16, 6.750155989720952e-16,
        3.5915714846623813e-16, -1.1368683772161603e-15, -9.036283006580436e-17, -1.9872992140790302e-16,
        6.594724766273429e-16, -7.415942859800851e-16, 1.7763568394002506e-16, 5.534461777756406e-16,
        -6.039613253960851e-16, 2.1660841523218898e-16,
    ]


def _ref_g1air_wrench() -> List[Float64]:
    """(qpos, qvel) after the first step count, then after the second."""
    return [
        -1.7797630719748753e-05, 1.1930071892200706e-06, 1.2927647355400127, 0.9999999979356358,
        -6.943299484863824e-07, -6.179364471190379e-05, -1.76009052855104e-05, 0.0001192336217480261,
        2.836974744142844e-06, 1.3040861300430246e-05, 6.164433956604656e-06, 1.4724528275933447e-05,
        8.726991845864444e-07, 9.582028065737283e-05, 2.6772623855435254e-06, 1.3104922197124776e-05,
        2.376562282979322e-05, 1.9501049658885082e-05, 1.0204845087977694e-06, -0.00027769596768280834,
        1.7353405362757726e-05, 0.0004152485012942063, -6.56398676074912e-05, 0.0005204883554317837,
        -0.001283786352623468, 0.002038258144410463, 0.0010849600356063714, 0.006046872471055266,
        -0.0022621154706342008, -0.00011719083726460635, -0.00036949847967442876, 0.0014183488708355383,
        0.0015362125425255563, -0.000815479067539574, 0.004506479292819029, 0.002026413427363147,
        -0.0035595261439497505, 0.0002386014378440141, -0.0470528919974864, -0.00027773197958566625,
        -0.02471745790177013, -0.007040362119048784, 0.02384672434960522, 0.0005673949488285688,
        0.0026081722600860492, 0.0012328867913209312, 0.0029449056551866895, 0.00017453983691728887,
        0.019164056131474566, 0.000535452477108705, 0.002620984439424955, 0.004753124565958644,
        0.0039002099317770166, 0.0002040969017595539, -0.05553919353656167, 0.003470681072551545,
        0.08304970025884126, -0.013127973521498238, 0.10409767108635673, -0.2567572705246936,
        0.40765162888209255, 0.21699200712127428, 1.2093744942110531, -0.4524230941268401,
        -0.02343816745292127, -0.07389969593488575, 0.28366977416710765, 0.30724250850511126,
        -0.1630958135079148, 0.9012958585638058, 0.4052826854726294, -0.00026436253011588237,
        1.6704243311353505e-05, 1.2894689784814608, 0.9999995446434206, -8.585931698044097e-06,
        -0.0009169927838780035, -0.00026412774942104016, 0.0017706375836717405, 4.0669775407103584e-05,
        0.00019722285915541282, 9.051118979802953e-05, 0.00021845947321899278, 1.2439146200692675e-05,
        0.0014191153885178465, 3.8367474079470525e-05, 0.0001981354002979801, 0.00035489923828862445,
        0.0002887392402014113, 1.4630225520885057e-05, -0.004172335056423087, 0.00022226690823988973,
        0.006189209861932405, -0.0017273740322641892, 0.007839692399911325, -0.019618059355616606,
        0.031125459721171103, 0.016000003607300573, 0.09022115275148355, -0.03414136932348062,
        -0.0022151604176687567, -0.005591839880403607, 0.021525843134371933, 0.02340250727368148,
        -0.01213948274895968, 0.06771288801114281, 0.030650768005113592, -0.01749518343788141,
        0.0010539743421420094, -0.23551101568415025, -0.000962343704638382, -0.12128395341124322,
        -0.03523283424045052, 0.11715806310554898, 0.002617693123755814, 0.013229994268255984,
        0.005937904661630239, 0.014433138565117602, 0.0007966869977056925, 0.093704256436292,
        0.0024692057639062507, 0.01328909900517205, 0.023584591015320528, 0.01905055956464607,
        0.000941356206229699, -0.27853651669240687, 0.012889329219021325, 0.41066966475674366,
        -0.152826583144142, 0.5241422523122322, -1.326925555867684, 2.104267314626293,
        1.0524003907148836, 5.989530048756436, -2.286374177980778, -0.17105266065132804,
        -0.3750512554701841, 1.4480239848263774, 1.5789960505421066, -0.8043302308825552,
        4.5187295640261365, 2.0556761705203694,
    ]


def _ref_g1floor_zero() -> List[Float64]:
    """(qpos, qvel) after the first step count, then after the second."""
    return [
        -6.459261566731989e-06, -1.947510391582322e-07, 0.790586377209237, 0.9999999871749687,
        -3.7363933960931574e-10, -0.00016015631182034098, 1.3448364965341048e-07, 0.00045727216348719085,
        4.0113787675304985e-07, -4.452147476610843e-07, -0.00019908359641216528, 4.284164203048026e-05,
        -2.0103587509880562e-06, 0.00045745220311235366, 1.2636726750169645e-07, -1.9422821940957734e-07,
        -0.0001992554516141088, 4.282623013520722e-05, 8.568160903332191e-07, 1.47152262200437e-07,
        -2.0233762319476616e-06, 0.00047462264087081796, 1.2997995410911564e-05, -9.000816202816752e-05,
        8.479543987743074e-05, 0.0016618521858435104, 9.148994721766255e-06, -0.0006376840784252973,
        -2.888753366418791e-05, 1.3748907790311993e-05, 9.057837685660773e-05, -8.49881990020398e-05,
        0.0016644742134776828, -8.82666122748546e-06, -0.0006389092123459523, 2.8698403874808965e-05,
        -0.0012918523133463977, -3.8950207831646436e-05, 0.017275441847390684, -1.4945573648265113e-07,
        -0.06406252500200435, 5.37934600913318e-05, 0.09145443269743816, 8.022757535060997e-05,
        -8.904294953221686e-05, -0.039816719282433057, 0.008568328406096051, -0.0004020717501976113,
        0.09149044062247073, 2.5273453500339288e-05, -3.884564388191547e-05, -0.03985109032282176,
        0.008565246027041444, 0.00017136321806664382, 2.9430452440087397e-05, -0.00040467524638953227,
        0.0949245281741636, 0.002599599082182313, -0.018001632405633503, 0.016959087975486148,
        0.33237043716870207, 0.0018297989443532509, -0.12753681568505945, -0.005777506732837582,
        0.0027497815580623986, 0.018115675371321544, -0.01699763980040796, 0.33289484269553654,
        -0.001765332245497092, -0.12778184246919047, 0.005739680774961793, -8.699626524519552e-05,
        -2.5095328286572645e-06, 0.7910826724438829, 0.9999977989286924, -1.736997211060561e-08,
        -0.0020981264655333524, 1.7618946938077327e-06, 0.005977423756592805, 4.150144756309191e-06,
        -4.963551974927363e-06, -0.002614909198190911, 0.0007282568479156509, -1.3454564151840373e-05,
        0.005979803978649023, 2.485331170699054e-06, -3.3598630514224354e-06, -0.0026172166066578562,
        0.000728144156544848, 2.657563377890679e-06, 1.8412970172224263e-06, -2.602497370391457e-05,
        0.0062106368214717095, 6.061700873522754e-05, -0.0011487281854368888, 0.0011231824432834637,
        0.021413663480755976, 0.00011800874045728999, -0.008173710999012573, -0.0003659705093369304,
        6.986436620521847e-05, 0.0011562629323240625, -0.0011251905789014095, 0.021447813303617565,
        -0.00011386857636985179, -0.008189392897824747, 0.0003635803873458391, -0.005660734331685692,
        -0.00015862297466530335, 0.023234575691850886, -2.4745690939365177e-06, -0.2681568457128699,
        0.0002249099712806394, 0.3815128620246076, 0.00024031052833594634, -0.00029742841057727327,
        -0.16702321403428336, 0.050095953759164, -0.0005785958809861947, 0.3816657788478124,
        0.0001750736990188113, -0.00023196887391749226, -0.16717282510167863, 0.05009128717884125,
        -1.488886094776233e-05, 0.00011433699466324501, -0.001642335830387609, 0.39652001944993415,
        -0.001357023150979679, -0.07199343346147041, 0.07241802472705028, 1.3521036268670417,
        0.0074560753682852195, -0.513966982854597, -0.02279973036790954, -0.0007938948815846334,
        0.0724799224888729, -0.07252123650672337, 1.354278309836031, -0.007195374472345385,
        -0.5149522959112304, 0.022651105048088822,
    ]


def _ref_g1floor_wrench() -> List[Float64]:
    """(qpos, qvel) after the first step count, then after the second."""
    return [
        -2.4041574950106312e-05, 8.194030019089888e-07, 0.7905864775726926, 0.9999999762971529,
        -3.4476537957003927e-09, -0.00021719155410474274, -1.5281445469008745e-05, 0.0005620234034670424,
        5.350315009828756e-06, 2.9861992429274215e-05, -0.0001870035127141443, 4.0659818019480303e-05,
        -1.5515317866848208e-06, 0.0005415864127311629, 5.082522002217539e-06, 3.0107939738421518e-05,
        -0.00016738271552610803, 4.149102690596803e-05, 1.243208026601628e-06, -0.0002814091073650536,
        1.3243040202299973e-05, 0.0008756732811889183, -5.336709343127074e-05, 0.0004341664952203833,
        -0.0012018466288289205, 0.0036503756264645894, 0.0010938150720814395, 0.005428446223658171,
        -0.0022901951321985738, -0.00010335027355523481, -0.00028061271717948763, 0.001335602894930626,
        0.0031512778414103373, -0.0008240642331977573, 0.0038863642456235382, 0.002054208152672507,
        -0.004808314990021262, 0.00016388060038179775, 0.01729551453853977, -1.3790615291760517e-06,
        -0.08687662232830486, -0.006112578235898667, 0.11240468069340846, 0.0010700630019657511,
        0.005972398485854843, -0.03740070254282886, 0.00813196360389606, -0.00031030635733696416,
        0.10831728254623257, 0.0010165044004435079, 0.0060215879476843035, -0.03347654310522161,
        0.008298205381193606, 0.00024864160532032557, -0.05628182147301072, 0.0026486080404599945,
        0.17513465623778365, -0.010673418686254148, 0.08683329904407666, -0.2403693257657841,
        0.7300751252929178, 0.2187630144162879, 1.0856892447316342, -0.4580390264397148,
        -0.02067005471104696, -0.05612254343589752, 0.26712057898612523, 0.6302555682820674,
        -0.16481284663955145, 0.7772728491247076, 0.4108416305345014, -0.0003520635238708,
        1.1832859419240132e-05, 0.7910828804965654, 0.9999955880896881, -6.457127522451215e-07,
        -0.0029615671570184226, -0.00023004503099445314, 0.00756368934635775, 8.038303061250406e-05,
        0.00045133942218265714, -0.0024296292830966646, 0.0006874404356422303, -8.84642985485577e-06,
        0.007254654384747797, 7.883371012567519e-05, 0.0004529011761847784, -0.002131196509139558,
        0.0006981202398189416, 6.798743750009058e-06, -0.004228043771269012, 0.00014904936544770248,
        0.012252187424451411, -0.001990253134055898, 0.006706304197280427, -0.018495656690492204,
        0.052046226155056656, 0.01621079819833619, 0.08233326345609357, -0.034373397357293375,
        -0.002380673406529732, -0.004437597203008289, 0.020392921126879578, 0.04433122412062806,
        -0.012341404675682304, 0.059845835745534814, 0.030906601992791324, -0.023379362809427304,
        0.0007537524434160802, 0.02320610901417758, -0.00015245858745253975, -0.3839363718664262,
        -0.030738161641442635, 0.48790169248452614, 0.005422694438522412, 0.030179803411363827,
        -0.15453002181450945, 0.04714216606703883, -0.00037516141430353787, 0.46719367011955454,
        0.005367363970580301, 0.030243027081170444, -0.13445828662500012, 0.04777956152253259,
        0.00017211740101631163, -0.28225985492332656, 0.007322691695358547, 0.8003421841286184,
        -0.19069093467070702, 0.4511905544725705, -1.2528336436502878, 3.436274700851419,
        1.07055220876806, 5.493901304579565, -2.294330464452742, -0.19891703459801405,
        -0.30149138767938494, 1.3732027592490148, 2.909803715128801, -0.8216511809053845,
        4.026777302847043, 2.066565496315539,
    ]


def _g1_points() -> List[Float64]:
    """The hand capsules' centres at qpos0: left, right."""
    return [
        0.30977236668867597, 0.13864034471638517, 0.8882261522828662, 0.30977236668867597,
        -0.13863034471638516, 0.8882261522828662,
    ]


def _g1_floor_dz() -> Float64:
    """The root's drop from qpos0 at which a foot contact is 1 mm deep."""
    return -0.0025000000000000022
# </xfrc-reference>


def _roll[X: Bool](
    xml: String, base: String, dz: Float64, w: List[Float64], steps_a: Int, steps_b: Int, names: List[String],
) raises -> List[Float64]:
    """qpos0 with the root raised by `dz`, at rest, the wrenches `w` held;
    (qpos, qvel) after `steps_a` steps, then after `steps_b`. `names`: when
    given, the name each row's body must have in the parsed model."""
    var fmd = parse_xml_full(expand_mjcf(xml, base), base)
    assert_true(not studio_uses_implicit(fmd), "the fixture must ask for the Euler integrator")
    assert_true(studio_cone_of(fmd) == ConeType.PYRAMIDAL, "the fixture must use the pyramidal cone")
    var dims = dims_from_flat(fmd, max_contacts=128, nmesh_verts=65536)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var d = Data[DT, DynDims, 1](dims)
    for i in range(nq):
        d.qpos.data[i] = sf.qpos0.data[i]
    d.qpos.data[2] = d.qpos.data[2] + Scalar[DT](dz)
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](0)
        d.qfrc.data[i] = Scalar[DT](0)
    forward_kinematics["cpu", DT, BATCH=1](d, m, None)
    for r in range(len(w) // 8):
        var b = Int(w[r * 8])
        if len(names) > 0:
            assert_true(
                String(fmd.body_names[b]) == names[r],
                "body " + String(b) + " is '" + String(fmd.body_names[b]) + "', the table says '" + names[r] + "'",
            )
        if w[r * 8 + 1] == 0.0:
            for k in range(6):
                d.xfrc_applied.data[b * XFRC_STRIDE + k] = Scalar[DT](w[r * 8 + 2 + k])
        else:
            xfrc_at_point(d, 0, b, w[r * 8 + 2], w[r * 8 + 3], w[r * 8 + 4], w[r * 8 + 5], w[r * 8 + 6], w[r * 8 + 7])
    var nact = dims.get_nact()
    var ctrl = List[Float64](length=nact, fill=0.0)
    var act = List[Scalar[DT]](length=nact if nact > 0 else 1, fill=Scalar[DT](0))
    var integ = EulerIntegrator[
        DT, DynDims, ConeType.PYRAMIDAL, 1, "newton",
        MAX_CONDIM=STUDIO_MAX_CONDIM, NOSLIP_ITER=STUDIO_NOSLIP, CRBA_TREEWALK=True, XFRC=X,
    ](dims)
    var out = List[Float64]()
    for t in range(1, steps_b + 1):
        if nact > 0:
            apply_actions_fields[DT](sf, d, ctrl, act, fmd.timestep)
        integ.step["cpu"](d, m)
        if t == steps_a or t == steps_b:
            for i in range(nq):
                out.append(Float64(d.qpos.data[i]))
            for i in range(nv):
                out.append(Float64(d.qvel.data[i]))
    return out^


def _worst(got: List[Float64], want: List[Float64], lo: Int, hi: Int) -> Float64:
    var w = 0.0
    for i in range(lo, hi):
        var e = abs(got[i] - want[i])
        if e > w:
            w = e
    return w


def _g1_rows(fmd_names: List[String], dz: Float64) raises -> List[Float64]:
    """The G1's wrench table: body ids by name, each hand's force at its
    capsule centre (`_g1_points` is at qpos0; the root is raised by `dz`)."""
    var names = _w_g1_names()
    var f = _w_g1_forces()
    var p = _g1_points()
    var rows = List[Float64]()
    for r in range(len(names)):
        var hit = -1
        for b in range(len(fmd_names)):
            if fmd_names[b] == names[r]:
                hit = b
        if hit < 0:
            raise Error("no body " + names[r])
        rows.append(Float64(hit))
        if r < 2:
            rows.append(1.0)
            for k in range(3):
                rows.append(f[r * 6 + k])
            rows.append(p[r * 3 + 0])
            rows.append(p[r * 3 + 1])
            rows.append(p[r * 3 + 2] + dz)
        else:
            rows.append(0.0)
            for k in range(6):
                rows.append(f[r * 6 + k])
    return rows^


def test_chain_wrench_and_its_control() raises:
    print("=== a wrench on each body of a rotated free-root chain ===")
    var none = List[Float64]()
    var nn = List[String]()
    var n = len(_ref_chain_zero()) // 2
    var got0 = _roll[True](XML_CHAIN, String(""), 0.0, none, CHAIN_STEPS_A, CHAIN_STEPS_B, nn)
    var w0a = _worst(got0, _ref_chain_zero(), 0, n)
    var w0b = _worst(got0, _ref_chain_zero(), n, 2 * n)
    print("  no wrench: worst |d| after", CHAIN_STEPS_A, "step", w0a, " after", CHAIN_STEPS_B, "steps", w0b)
    assert_true(w0a < 1e-13 and w0b < 1e-11, "the CONTROL is off MuJoCo: the gap is not the wrench")
    var got = _roll[True](XML_CHAIN, String(""), 0.0, _w_chain(), CHAIN_STEPS_A, CHAIN_STEPS_B, nn)
    var wa = _worst(got, _ref_chain_wrench(), 0, n)
    var wb = _worst(got, _ref_chain_wrench(), n, 2 * n)
    print("  wrenches:  worst |d| after", CHAIN_STEPS_A, "step", wa, " after", CHAIN_STEPS_B, "steps", wb)
    var spread = _worst(_ref_chain_wrench(), _ref_chain_zero(), 0, n)
    print("  MuJoCo's two answers differ by", spread, "after", CHAIN_STEPS_A, "step")
    assert_true(spread > 1e-5, "the fixture cannot tell a wrench from none")
    assert_true(wa < 1e-13, "one step with the wrenches is " + String(wa) + " from MuJoCo")
    assert_true(wb < 1e-11, String(CHAIN_STEPS_B) + " steps with the wrenches are " + String(wb) + " from MuJoCo")
    print("  PASS")


def test_zero_wrench_is_the_plain_build() raises:
    print("=== XFRC=True with a zero buffer == XFRC=False, bit for bit ===")
    var none = List[Float64]()
    var nn = List[String]()
    var a = _roll[True](XML_CHAIN, String(""), 0.0, none, CHAIN_STEPS_A, CHAIN_STEPS_B, nn)
    var b = _roll[False](XML_CHAIN, String(""), 0.0, none, CHAIN_STEPS_A, CHAIN_STEPS_B, nn)
    var c = _roll[True](XML_BOX, String(""), 0.0, none, BOX_STEPS_A, BOX_STEPS_B, nn)
    var e = _roll[False](XML_BOX, String(""), 0.0, none, BOX_STEPS_A, BOX_STEPS_B, nn)
    var diff = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            diff += 1
    for i in range(len(c)):
        if c[i] != e[i]:
            diff += 1
    print("  values compared", len(a) + len(c), " differing", diff)
    assert_true(len(a) > 0 and len(c) > 0, "nothing compared")
    assert_true(diff == 0, String(diff) + " values differ between the two builds with no wrench")
    # and the XFRC=False build does not read the buffer
    var f = _roll[False](XML_CHAIN, String(""), 0.0, _w_chain(), CHAIN_STEPS_A, CHAIN_STEPS_B, nn)
    var diff2 = 0
    for i in range(len(f)):
        if f[i] != b[i]:
            diff2 += 1
    assert_true(diff2 == 0, "an XFRC=False build read the wrench buffer")
    print("  PASS")


def test_force_at_a_point() raises:
    print("=== xfrc_at_point against mj_applyFT into qfrc_applied ===")
    var nn = List[String]()
    var got = _roll[True](XML_CHAIN, String(""), 0.0, _w_chain_point(), 1, 1, nn)
    var want = _ref_chain_point()
    var w = _worst(got, want, 0, len(want))
    var spread = _worst(want, _ref_chain_zero(), 0, len(want))
    print("  worst |d| after one step", w, " (the force moves the answer by", spread, ")")
    assert_true(spread > 1e-5, "the point force does nothing readable")
    assert_true(w < 1e-13, "a force at a point is " + String(w) + " from MuJoCo's mj_applyFT")
    print("  PASS")


def _relative_leg(label: String, got0: List[Float64], got: List[Float64], want0: List[Float64], want: List[Float64]) raises:
    var n = len(want) // 2
    for part in range(2):
        var lo = part * n
        var e0 = _worst(got0, want0, lo, lo + n)
        var e = _worst(got, want, lo, lo + n)
        var spread = _worst(want, want0, lo, lo + n)
        var bar = max(10.0 * e0, 1e-12)
        print("  ", label, "part", part, ": control", e0, " wrench", e, " bar", bar, " the wrench's effect", spread)
        assert_true(spread > 1e-6, label + ": the wrench does nothing readable")
        assert_true(e <= bar, label + ": the wrench leg is " + String(e) + " from MuJoCo, the control " + String(e0))
        assert_true(e * 1000.0 <= spread, label + ": the error " + String(e) + " is not small beside the effect " + String(spread))


def test_box_in_contact() raises:
    print("=== a box on a plane, pushed and twisted: the wrench reaches the solver ===")
    var none = List[Float64]()
    var nn = List[String]()
    var got0 = _roll[True](XML_BOX, String(""), 0.0, none, BOX_STEPS_A, BOX_STEPS_B, nn)
    var got = _roll[True](XML_BOX, String(""), 0.0, _w_box(), BOX_STEPS_A, BOX_STEPS_B, nn)
    _relative_leg(String("box"), got0, got, _ref_box_zero(), _ref_box_wrench())
    print("  PASS")


def test_g1_hands_and_torso() raises:
    print("=== the G1: a force at each hand's capsule centre, a wrench on the torso ===")
    var src = read_model_source(G1_XML)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var names = List[String]()
    for b in range(len(fmd.body_names)):
        names.append(String(fmd.body_names[b]))
    var none = List[Float64]()
    var nn = List[String]()
    var air_rows = _g1_rows(names, G1_AIR_DZ)
    var air0 = _roll[True](src[0], src[1], G1_AIR_DZ, none, G1_STEPS_A, G1_STEPS_B, nn)
    var air = _roll[True](src[0], src[1], G1_AIR_DZ, air_rows, G1_STEPS_A, G1_STEPS_B, nn)
    _relative_leg(String("G1 in the air"), air0, air, _ref_g1air_zero(), _ref_g1air_wrench())
    # the floor leg: the same forces, the points lowered with the robot
    var floor_rows = _g1_rows(names, _g1_floor_dz())
    var fl0 = _roll[True](src[0], src[1], _g1_floor_dz(), none, G1_STEPS_A, G1_STEPS_B, nn)
    var fl = _roll[True](src[0], src[1], _g1_floor_dz(), floor_rows, G1_STEPS_A, G1_STEPS_B, nn)
    _relative_leg(String("G1 on the floor"), fl0, fl, _ref_g1floor_zero(), _ref_g1floor_wrench())
    print("  PASS")


comptime HCEnv[X: Bool] = Phyics3dEnv[HalfCheetahModel, HalfCheetahConfig, DType.float64, False, XFRC=X]
comptime HC_STEPS = 5
comptime HC_NONE = 0
comptime HC_RECORD = 1       # force (30, 0, 20) at the torso's CoM, torque (0, 3, 0)
comptime HC_POINT = 2        # the same force 0.1 m above the CoM: r x f = (0, 3, 0)


def _hc[X: Bool](kind: Int) raises -> List[Float64]:
    """The CPU HalfCheetah from qpos = qvel = 0, HC_STEPS control steps at a
    zero action; (qpos, qvel) at the end."""
    comptime E = HCEnv[X]
    var env = E()
    _ = env.reset()
    env.set_state(List[Float64](length=E.NQ, fill=0.0), List[Float64](length=E.NV, fill=0.0))
    if kind == HC_RECORD:
        env.set_xfrc(1, 30.0, 0.0, 20.0, 0.0, 3.0, 0.0)
    elif kind == HC_POINT:
        env.set_xfrc_at_point(
            1, 30.0, 0.0, 20.0,
            Float64(env.d.xipos.data[3]), Float64(env.d.xipos.data[4]), Float64(env.d.xipos.data[5]) + 0.1,
        )
    var act = ContAction[E.ACTION_DIM]()
    for j in range(E.ACTION_DIM):
        act.data[j] = 0.0
    for _ in range(HC_STEPS):
        _ = env.step(act)
    var out = List[Float64]()
    for i in range(E.NQ):
        out.append(Float64(env.d.qpos.data[i]))
    for i in range(E.NV):
        out.append(Float64(env.d.qvel.data[i]))
    return out^


def test_cpu_env_reads_the_buffer() raises:
    print("=== Phyics3dEnv[..., XFRC=True]: set_xfrc and set_xfrc_at_point ===")
    var none = _hc[True](HC_NONE)
    var rec = _hc[True](HC_RECORD)
    var pt = _hc[True](HC_POINT)
    var off_none = _hc[False](HC_NONE)
    var off_rec = _hc[False](HC_RECORD)
    var n = len(none)
    var moved = _worst(rec, none, 0, n)
    var same = _worst(pt, rec, 0, n)
    var diff = 0
    for i in range(n):
        if off_rec[i] != off_none[i] or off_none[i] != none[i]:
            diff += 1
    print("  values", n, " the wrench moves the state by", moved, " point form vs record", same, " XFRC=False vs no wrench: differing", diff)
    assert_true(n > 0 and moved > 1e-3, "the CPU env did not read the wrench")
    assert_true(same < 1e-12, "set_xfrc_at_point is " + String(same) + " from the record it stands for")
    assert_true(diff == 0, "an env built without XFRC read the buffer, or the XFRC env with no wrench differs from it")
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
