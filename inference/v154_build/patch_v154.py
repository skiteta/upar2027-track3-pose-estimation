"""v154 = v153 + confidence-preserving fusion: in the (SAHI-free) fusion every member's per-joint unit-peak map is scaled
back by its own per-joint peak (clamped to [1e-6, 1]) before the anatomical weighted mean, so a member's confidence is no
longer discarded. Consensus maps (person score) are unchanged. Usage: patch_v154.py SRC_V153_RUN_PY DST"""
import sys
s = open(sys.argv[1]).read()
old = """            if not routed_unit_maps:
                for base_m in base_96_maps:
                    u_base = unit_peak(base_m)
                    routed_unit_maps.append(u_base)
                    consensus_unit_maps.append(u_base)
"""
new = """            if not routed_unit_maps:
                for base_m in base_96_maps:
                    u_base = unit_peak(base_m)
                    if FUSION_KEEP_CONFIDENCE:
                        routed_unit_maps.append(u_base * base_m.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6, max=1.0))
                    else:
                        routed_unit_maps.append(u_base)
                    consensus_unit_maps.append(u_base)
"""
assert s.count(old) == 1 and 'MULTISCALE_PADDINGS' in s; s = s.replace(old, new)
old2 = "MULTISCALE_MAX_TIER = 1\n"; assert s.count(old2) == 1
s = s.replace(old2, old2 + "FUSION_KEEP_CONFIDENCE = True  # v154: member per-joint confidence kept in the fusion\n")
open(sys.argv[2], 'w').write(s); print('patched')
