"""v155 = v154 + last-resort completion guard. Once TEMPLATE_GUARD_SECONDS have elapsed since model loading started
(all slower controller stages - tier walk, centre mode, member drops, the 7,120 s single-member emergency - come first),
predict_image returns a box-relative mean-pose template (score 0.01) without any model forward, so the remaining images
finish in milliseconds and the run completes instead of being killed. Before that time the code path is unchanged.
Usage: patch_v155.py SRC_V154_RUN_PY DST TEMPLATE_JSON"""
import json, sys
s = open(sys.argv[1]).read(); tmpl = json.load(open(sys.argv[3])); assert len(tmpl) == 17
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:80], s.count(old)); s = s.replace(old, new)
rep("FUSION_KEEP_CONFIDENCE = True  # v154: member per-joint confidence kept in the fusion\n",
    "FUSION_KEEP_CONFIDENCE = True  # v154: member per-joint confidence kept in the fusion\n"
    "TEMPLATE_GUARD_SECONDS = 7150.0  # v155: after this wall time, box-relative mean-pose template only (no model forward)\n"
    f"_TEMPLATE_POSE = {tmpl!r}  # mean (x/w, y/h) per joint of the development boxes; ears = nose\n")
rep("""def predict_image(sample: dict[str, Any]) -> dict[str, Any]:
    out = _predict_image_core(sample)
""", """def _template_prediction(sample: dict[str, Any]) -> dict[str, Any]:
    keypoints, scores = [], []
    for box in sample.get("boxes") or []:
        try:
            x_coord, y_coord, width, height = (float(value) for value in box[:4])
            if not all(math.isfinite(v) for v in (x_coord, y_coord, width, height)):
                raise ValueError
        except (ValueError, TypeError):
            x_coord = y_coord = width = height = 0.0
        keypoints.append([[x_coord + tx * width, y_coord + ty * height, 1.0] for tx, ty in _TEMPLATE_POSE])
        scores.append(0.01)
    return {"keypoints": keypoints, "scores": scores}


def predict_image(sample: dict[str, Any]) -> dict[str, Any]:
    if TEMPLATE_GUARD_SECONDS is not None:
        guard_start = _RUNTIME["load_start"] or _RUNTIME["start"]
        if guard_start is not None and time.perf_counter() - guard_start >= TEMPLATE_GUARD_SECONDS:
            if not _RUNTIME.get("template_guard"):
                _RUNTIME["template_guard"] = True
                print(f"[submission] completion guard: mean-pose template for the remaining images after "
                      f"{time.perf_counter() - guard_start:.1f}s ({_RUNTIME['images']} images)", flush=True)
            _RUNTIME["template_images"] = _RUNTIME.get("template_images", 0) + 1
            return _template_prediction(sample)
    out = _predict_image_core(sample)
""")
open(sys.argv[2], 'w').write(s); print('patched')
