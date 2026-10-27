"""v153 = v127 run.py + tier-gated scale TTA replacing pose-SAHI.
While the wall-clock controller is at tier <= MULTISCALE_MAX_TIER (and has not disabled SAHI by time), every member uses
MULTISCALE_PADDINGS (equal weights, flip each) instead of (1.2, 1.3); pose-SAHI never runs (its v127 value was +0.0001).
At tier >= 2 or after the SAHI time budget the original (1.2, 1.3) views are used, i.e. exactly v127's tier-3 path.
The non-SAHI crop-rounding correction averages over the active paddings (identical arithmetic for two paddings).
Usage: patch_v153.py SRC_RUN_PY DST_RUN_PY"""
import sys
s = open(sys.argv[1]).read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:80], s.count(old)); s = s.replace(old, new)
rep("SCORE_SOFTVIS_TAU = 0.5\n", "SCORE_SOFTVIS_TAU = 0.5\nMULTISCALE_PADDINGS = (1.15, 1.2, 1.3, 1.35)  # v153 scale TTA at tiers <= MULTISCALE_MAX_TIER\nMULTISCALE_MAX_TIER = 1\n")
rep("""    if _RUNTIME["reduced"]:
        return tuple(((TARGET_PADDING,), (1.0,)) for _ in MODEL_FILES)
""", """    if _RUNTIME["reduced"]:
        return tuple(((TARGET_PADDING,), (1.0,)) for _ in MODEL_FILES)
    if MULTISCALE_PADDINGS and not _RUNTIME["sahi_disabled"] and int(_RUNTIME.get("sahi_tier", 0)) <= MULTISCALE_MAX_TIER:
        weights = tuple(1.0 / len(MULTISCALE_PADDINGS) for _ in MULTISCALE_PADDINGS)
        return tuple((tuple(MULTISCALE_PADDINGS), weights) for _ in MODEL_FILES)
""")
rep("""        and sahi_tier < 3
        and not _RUNTIME["reduced"]
    )
""", """        and sahi_tier < 3
        and not _RUNTIME["reduced"]
        and not MULTISCALE_PADDINGS
    )
""")
rep("""                else:
                    d12 = _view_rounding_delta_from_geom(coords[bi], crop_geometry(c_box, MEMBER_PADDINGS[0][0]))
                    d13 = _view_rounding_delta_from_geom(coords[bi], crop_geometry(c_box, MEMBER_PADDINGS[0][1]))
                    delta_bi = 0.5 * (d12 + d13)
""", """                else:
                    view_pads = member_views[0][0]
                    delta_bi = sum(_view_rounding_delta_from_geom(coords[bi], crop_geometry(c_box, p)) for p in view_pads) / len(view_pads)
""")
open(sys.argv[2], 'w').write(s); print('patched')
