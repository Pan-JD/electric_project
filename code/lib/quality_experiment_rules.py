"""Binary hard filtering and minimal YOLO conversion for the raw baseline."""
from __future__ import annotations

import math


FILTER_RULES_VERSION = 3


def _finite_coords(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        result = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(x) for x in result) else None


def baseline_record(record, classes):
    """Keep every convertible selected box, without quality or duplicate filtering."""
    result = {'boxes': [], 'skipped_boxes': 0, 'reason': '',
              'normalize_exif': record.get('orientation', 1) != 1}
    width, height = record.get('width', 0), record.get('height', 0)
    if not record.get('ok') or width <= 0 or height <= 0:
        result['reason'] = 'image_unavailable'
        return result
    if record.get('annotation', {}).get('status') != 'ok':
        result['reason'] = 'annotation_unavailable'
        return result
    selected = set(classes)
    for position, original in enumerate(record.get('boxes', [])):
        if selected and original.get('class') not in selected:
            continue
        coords = _finite_coords(original.get('coords'))
        if coords is None or not (0 <= coords[0] < coords[2] <= width and 0 <= coords[1] < coords[3] <= height):
            result['skipped_boxes'] += 1
            continue
        result['boxes'].append({'class':original['class'],'coords':coords,'index':original.get('index',position)})
    if not result['boxes']:
        result['reason'] = 'no_convertible_selected_targets'
    return result


def assess_record(record, config):
    """Return image/box decisions, preserving valid targets in mixed-quality images.

    Thresholds: min(width,height) < min_side; nominal YOLO short side <=
    tiny_model_side; bounding-box area / canvas > max_box_fraction.
    Bad boxes are removed individually; other targets and image pixels remain.
    Output states are keep/reject. Canvas-level inconsistencies reject the image.
    """
    rules = config.get('hard_filter', {})
    minimum = float(rules.get('min_side', 200))
    tiny = float(rules.get('tiny_model_side', 4))
    maximum = float(rules.get('max_box_fraction', .9))
    size = rules.get('imgsz', config.get('training', {}).get('imgsz', 640))
    size = [size, size] if isinstance(size, (int, float)) else list(size)
    if len(size) == 1:
        size *= 2
    if (len(size) != 2 or not all(math.isfinite(float(x)) and float(x) > 0 for x in size)
            or not all(math.isfinite(v) for v in (minimum, tiny, maximum))
            or minimum <= 0 or tiny < 0 or not 0 < maximum <= 1):
        raise ValueError('Invalid hard-filter thresholds')
    classes = set(config.get('classes', []))
    baseline = baseline_record(record, classes)
    result = {'baseline_eligible': bool(baseline['boxes']), 'clean_status': 'reject', 'boxes_json': baseline['boxes'],
              'clean_boxes_json': [], 'mask_regions_json': [], 'reject_reasons_json': [],
              'box_decisions': [], 'kept_indices': []}
    reasons = result['reject_reasons_json']

    def reason(code):
        if code not in reasons:
            reasons.append(code)

    try:
        width, height = float(record.get('width', 0)), float(record.get('height', 0))
    except (TypeError, ValueError):
        width, height = 0, 0
    if not record.get('ok') or not all(math.isfinite(v) and v > 0 for v in (width, height)):
        reason('image_unavailable')
        return result
    if record.get('annotation', {}).get('status') != 'ok':
        reason('annotation_unavailable')
        return result
    frame_bad = not record.get('frame_ok', True) or bool(record.get('frame_reasons'))
    if frame_bad:
        reason('annotation_frame')
    low_resolution = min(width, height) < minimum
    if low_resolution:
        reason('low_resolution')
    scale = min(float(size[0])/height, float(size[1])/width)
    if not rules.get('scaleup', True):
        scale = min(1., scale)
    source_boxes = record.get('boxes') or []
    seen = set()
    for position, original in enumerate(source_boxes):
        box = dict(original)
        index = original.get('index', position)
        box['index'] = index
        category = str(original.get('class') or '')
        decision = {'index': index, 'class': category, 'reasons': [], 'action': 'keep'}
        result['box_decisions'].append(decision)
        if classes and category not in classes:
            # Classes select this experiment's detection task. Other objects are
            # left visible and omitted from its labels, including overlapping ones.
            decision['action'] = 'ignore_outside_experiment'
            continue
        coords = _finite_coords(original.get('coords'))
        geometric = []
        if coords is None:
            geometric.append('missing_or_nonfinite_coordinates')
        else:
            x1, y1, x2, y2 = coords
            if x2 <= x1 or y2 <= y1:
                geometric.append('reversed_or_zero_area')
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 > width or y1 > height or x2 < 0 or y2 < 0:
                geometric.append('outside_canvas')
        known_class = bool(category) and (not classes or category in classes)
        if not geometric and known_class:
            box['coords'] = coords
        if geometric:
            decision['reasons'].extend(geometric)
            reason('box_geometry')
        if not known_class:
            decision['reasons'].append('unknown_class')
            reason('unknown_class')
        if not geometric:
            area = (coords[2]-coords[0]) * (coords[3]-coords[1]) / (width*height)
            short = min(coords[2]-coords[0], coords[3]-coords[1]) * scale
            box['area_fraction'], box['short_at_model'] = area, short
            decision['area_fraction'], decision['short_at_model'] = area, short
            if short <= tiny:
                decision['reasons'].append('tiny_target')
                reason('tiny_target')
            if area > maximum:
                decision['reasons'].append('large_box')
                reason('large_box')
        if decision['reasons']:
            decision['action'] = 'remove_box'
            continue
        signature = category, tuple(coords)
        if signature in seen:
            decision['reasons'].append('duplicate_box')
            decision['action'] = 'remove_duplicate_box'
            reason('duplicate_box')
            continue
        seen.add(signature)
        result['clean_boxes_json'].append(box)
        result['kept_indices'].append(index)
    if not result['clean_boxes_json']:
        reason('no_valid_targets')
        return result
    result['clean_status'] = 'reject' if low_resolution or frame_bad else 'keep'
    return result
