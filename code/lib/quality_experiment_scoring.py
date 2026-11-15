"""Continuous 0..5 quality scores with normal plateaus, using cached metrics.

Values are engineering screening scores in [0, 5]. Missing dimensions retain
None and reduce coverage. The metric anchors are configurable.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path


DEFAULT_WEIGHTS = {'a1': .15, 'a2': .15, 'a3': .10, 'a4': .10, 'b1': .25, 'b2': .25}
DEFAULT_SCORING = {
    'brightness_anchors': [[0,0],[10,1],[20,2],[30,3],[40,4],[50,5],
                           [210,5],[225,4],[235,3],[245,2],[250,1],[255,0]],
    'clipping_anchors': [[0,5],[.10,5],[.20,4],[.35,3],[.50,2],[.65,1],[.80,0],[1,0]],
    'clipping_aggregation':'sum',
    'dark_detail': {'enabled':True,'mean_full_penalty':50,'mean_no_penalty':80,
                    'contrast_anchors':[[0,0],[5,1],[12,2],[25,3],[40,4],[60,5]]},
    'sharpness': {
        'image_laplacian':[[0,0],[4,1],[10,2],[25,3],[60,4],[150,5]],
        'image_tenengrad':[[0,0],[40,1],[100,2],[250,3],[650,4],[1600,5]],
        'target_laplacian':[[0,0],[3,1],[8,2],[20,3],[50,4],[120,5]],
        'target_tenengrad':[[0,0],[30,1],[80,2],[200,3],[500,4],[1200,5]],
        'image_weight':0.,
    },
    'text_overlap_anchors': [[0,5],[.02,5],[.05,4],[.10,3],[.20,2],[.35,1],[.50,0],[1,0]],
    'target_area_anchors': [[0,0],[.00001,1],[.00004,2],[.0001,3],[.0003,4],[.001,5],
                            [.50,5],[.60,4],[.70,3],[.80,2],[.90,1],[.97,0],[1,0]],
    'target_short_anchors': [[0,0],[4,0],[8,1],[12,2],[16,3],[24,4],[32,5]],
    'weak_dimension_caps': {key:[[0,.5],[1,1.5],[2,2.5],[3,3.5],[4,5],[5,5]]
                            for key in ('a1','a2','a3','a4')},
    'target_aggregation': 'mean',
}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def interpolate(value, anchors):
    if not finite(value):
        return None
    if len(anchors) < 2 or any(not finite(x) or not finite(y) or not 0 <= y <= 5 for x, y in anchors):
        raise ValueError('Scoring anchors require finite x and scores in [0,5]')
    if any(b[0] <= a[0] for a, b in zip(anchors, anchors[1:])):
        raise ValueError('Scoring anchor x values must be strictly increasing')
    if value <= anchors[0][0]:
        return float(anchors[0][1])
    if value >= anchors[-1][0]:
        return float(anchors[-1][1])
    for (left, a), (right, b) in zip(anchors, anchors[1:]):
        if left <= value <= right:
            return float(a + (b-a)*(value-left)/(right-left))


def grade(value):
    """Keep continuous scores; None is missing, zero is an assessed poor score."""
    if value is None: return None
    if not finite(value): raise ValueError('Quality scores must be finite')
    return round(max(0.,min(5.,value)),4)


def average(values):
    return sum(values)/len(values) if values else None


def dimension(score, status, evidence=None, coverage=None):
    return {'score': grade(score),
            'status': status, 'evidence': evidence or {},
            'coverage': float(coverage if coverage is not None else 1. if score is not None else 0.)}


def validate_weights(values):
    weights = {**DEFAULT_WEIGHTS, **values}
    if set(weights) != set(DEFAULT_WEIGHTS) or any(not finite(w) or w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError('weights require a1,a2,a3,a4,b1,b2 and finite nonnegative values with positive sum')
    return weights


def combine_dimensions(dimensions, weights, scoring=None):
    scoring={**DEFAULT_SCORING,**(scoring or {})}
    dimensions={k:{**v,'score':grade(v['score'])} for k,v in dimensions.items()}
    configured = sum(weights.values())
    effective={}
    for key, weight in weights.items():
        item=dimensions[key]
        coverage=item.get('coverage', 1. if item['score'] is not None else 0.)
        if not finite(coverage) or not 0<=coverage<=1:
            raise ValueError(f'{key} coverage must be finite in [0,1]')
        effective[key]=weight*coverage if item['score'] is not None else 0.
    overall=dimensions.get('b')
    if overall and overall['score'] is not None and all(dimensions[key]['score'] is None for key in ('b1','b2')):
        coverage=overall.get('coverage',1.)
        if not finite(coverage) or not 0<=coverage<=1:
            raise ValueError('b coverage must be finite in [0,1]')
        effective['b']=(weights['b1']+weights['b2'])*coverage
    assessed = sum(effective.values())
    points = sum(weight*dimensions[key]['score'] for key,weight in effective.items()
                 if dimensions[key]['score'] is not None)
    image_weight = sum(weights[k] for k in ('a1','a2','a3','a4'))
    assessed_image_weight = sum(effective[k] for k in ('a1','a2','a3','a4'))
    image_points = sum(effective[k]*dimensions[k]['score'] for k in ('a1','a2','a3','a4') if dimensions[k]['score'] is not None)
    image_mean = image_points/assessed_image_weight if assessed_image_weight else None
    raw_total = points/assessed if assessed else None
    caps={}
    for key,anchors in scoring.get('weak_dimension_caps',{}).items():
        if key not in ('a1','a2','a3','a4'):
            raise ValueError('weak_dimension_caps requires image dimension keys')
        item=dimensions[key]
        # Only fully measured, positively weighted dimensions impose a ceiling.
        if weights[key]>0 and item['score'] is not None and item.get('coverage',1.)==1.:
            caps[key]=interpolate(item['score'],anchors)
    ceiling=min(caps.values(),default=5.)
    image_score=min(image_mean,ceiling) if image_mean is not None else None
    adjusted_points=points
    if image_score is not None:
        adjusted_points+=image_score*assessed_image_weight-image_points
    total_score=min(adjusted_points/assessed,ceiling) if assessed else None
    return {'dimensions': dimensions,
            'image_score': grade(image_score),
            'image_weighted_average': round(image_mean,4) if image_mean is not None else None,
            'total_weighted_average': round(raw_total,4) if raw_total is not None else None,
            'weak_dimension_caps':caps,
            'quality_ceiling':ceiling,
            'image_coverage': round(assessed_image_weight/image_weight, 6) if image_weight else 0.,
            'total_score': grade(total_score),
            'coverage': round(assessed/configured, 6),
            'complete': math.isclose(assessed, configured),
            'weighted_points': round(points, 4),
            'assessed_weight': assessed, 'configured_weight': configured,
            'weights': weights}


def sharpness_dimension(record, boxes, cfg):
    """Joint derivative responses at the cached image/ROI measurement scales."""
    settings=cfg['sharpness']
    image_weight=settings.get('image_weight',0.)
    if not finite(image_weight) or not 0<=image_weight<=1:
        raise ValueError('sharpness.image_weight must be in [0,1]')
    def measure(metrics,scope):
        metrics=metrics or {}
        lap,ten=metrics.get('laplacian_variance'),metrics.get('tenengrad')
        if not all(finite(v) and v>=0 for v in (lap,ten)): return None
        def response(value,key):
            anchors=settings[key]
            if any(x<0 for x,_ in anchors): raise ValueError('Sharpness anchors require nonnegative metrics')
            # Log spacing prevents very large edge responses from dominating.
            return interpolate(math.log1p(value),[[math.log1p(x),y] for x,y in anchors])
        a=response(lap,scope+'_laplacian');b=response(ten,scope+'_tenengrad')
        return {'score':grade(math.sqrt(a*b)),'laplacian_score':a,'tenengrad_score':b,
                'laplacian_variance':lap,'tenengrad':ten,
                'measurement_width':metrics.get('measurement_width'),
                'measurement_height':metrics.get('measurement_height')}
    whole=measure(record.get('metrics'),'image')
    targets=[]
    for i,box in enumerate(boxes):
        value=measure(box.get('metrics'),'target')
        if value is not None: targets.append({'index':box.get('index',i),**value})
    target_coverage=len(targets)/len(boxes) if boxes else 0.
    w=image_weight if whole is not None else 0.
    t=(1-image_weight)*target_coverage
    points=(w*whole['score'] if whole is not None else 0.)+t*(average([x['score'] for x in targets]) or 0.)
    return dimension(points/(w+t) if w+t else None,'assessed' if w+t else 'missing_image_metrics',
                     {'aggregation':'image_and_target_joint_derivatives','image_weight':image_weight,
                      'whole':whole,'targets':targets,'retained_targets':len(boxes)},w+t)


def score_record(record, kept_indices, config, annotation_scores=None):
    """Score retained targets and whole-image metrics from inspect_file payloads.

    ``config`` accepts top-level ``weights`` and nested ``scoring``. a3 describes
    OCR-detected overlay candidate coverage; physical foreground occlusion stays
    unassessed and is explicitly included in its evidence.
    ``annotation_scores`` accepts b1/b2 or an overall score in 0..5, or None.
    """
    cfg = {**DEFAULT_SCORING, **config.get('scoring', {})}
    if cfg['target_aggregation'] != 'mean':
        raise ValueError('target_aggregation currently supports mean only')
    weights = validate_weights(config.get('weights', {}))
    keep = set(kept_indices)
    boxes = [box for i, box in enumerate(record.get('boxes', [])) if box.get('index', i) in keep]
    exposure = []
    for i, box in enumerate(boxes):
        m = box.get('metrics') or {}
        mean, dark, bright = (m.get(key) for key in ('mean_gray','dark_fraction_le10','bright_fraction_ge245'))
        if not all(finite(v) for v in (mean,dark,bright)):
            continue
        if not (0 <= mean <= 255 and 0 <= dark <= 1 and 0 <= bright <= 1):
            continue
        brightness = interpolate(mean, cfg['brightness_anchors'])
        clipping_mode=cfg.get('clipping_aggregation','max')
        if clipping_mode not in ('max','sum'): raise ValueError('clipping_aggregation requires max or sum')
        clipping_fraction=min(1.,dark+bright) if clipping_mode=='sum' else max(dark,bright)
        clipping = interpolate(clipping_fraction, cfg['clipping_anchors'])
        detail_cfg=cfg.get('dark_detail',{})
        detail_score=None
        contrast=m.get('contrast_p90_p10')
        if detail_cfg.get('enabled',False) and finite(contrast) and contrast>=0:
            full=detail_cfg.get('mean_full_penalty',50)
            clear=detail_cfg.get('mean_no_penalty',80)
            if not finite(full) or not finite(clear) or not 0<=full<clear<=255:
                raise ValueError('dark_detail requires 0 <= mean_full_penalty < mean_no_penalty <= 255')
            strength=max(0.,min(1.,(clear-mean)/(clear-full)))
            response=interpolate(contrast,detail_cfg['contrast_anchors'])
            detail_score=5.-strength*(5.-response)
        candidates=[brightness,clipping]+([detail_score] if detail_score is not None else [])
        exposure.append({'index': box.get('index',i), 'score': grade(min(candidates)),
                         'brightness_score': brightness, 'clipping_score': clipping,
                         'dark_detail_score':detail_score,'contrast_p90_p10':contrast,
                         'clipping_fraction':clipping_fraction,'clipping_aggregation':clipping_mode,
                         'mean_gray': mean, 'dark_fraction_le10': dark, 'bright_fraction_ge245': bright})
    dimensions = {'a1': dimension(average([x['score'] for x in exposure]),
                                  'assessed' if exposure else 'missing_target_metrics',
                                  {'aggregation':'equal_target_mean','assessed_targets':len(exposure),
                                   'retained_targets':len(boxes),'targets':exposure},
                                  len(exposure)/len(boxes) if boxes else 0.)}
    dimensions['a2'] = sharpness_dimension(record,boxes,cfg)
    dimensions['a3'] = score_text_occlusion(record, boxes, cfg, config)
    areas = []
    for i, box in enumerate(boxes):
        area = box.get('area_fraction')
        if finite(area) and 0 <= area <= 1:
            area_score=interpolate(area,cfg['target_area_anchors'])
            short=box.get('short_at_model')
            if not finite(short) or short<0: continue
            short_score=interpolate(short,cfg['target_short_anchors'])
            areas.append({'index':box.get('index', i), 'area_fraction':area,
                          'short_at_model':short,'area_score':area_score,'short_side_score':short_score,
                          'score':grade(min(area_score,short_score))})
    dimensions['a4'] = dimension(average([x['score'] for x in areas]),
                                  'assessed' if areas else 'missing_target_geometry',
                                  {'aggregation':'equal_target_mean', 'assessed_targets':len(areas),
                                   'retained_targets':len(boxes), 'targets':areas},
                                  len(areas)/len(boxes) if boxes else 0.)
    ratings = annotation_scores or {}
    for key in ('b1','b2'):
        value = ratings.get(key)
        if value is not None and (not finite(value) or int(value) != value or not 0 <= value <= 5):
            raise ValueError(f'{key} requires an integer rating in 0..5')
        dimensions[key] = dimension(value,
                                     'imported_model_rating' if value is not None else 'pending_model_rating',
                                     {'rating':value,'scale':'0..5'})
    overall=ratings.get('score')
    if overall is not None and (not finite(overall) or int(overall)!=overall or not 0<=overall<=5):
        raise ValueError('score requires an integer rating in 0..5')
    if any(dimensions[key]['score'] is not None for key in ('b1','b2')):
        dimensions['b']=dimension(None,'component_ratings_selected')
    else:
        dimensions['b']=dimension(overall,
                                   'imported_overall_annotation_rating' if overall is not None else 'pending_model_rating',
                                   {'rating':overall,'scale':'0..5',
                                    'weight':'b1+b2'})
    return combine_dimensions(dimensions, weights, cfg)


def score_text_occlusion(record, boxes, cfg, config):
    evidence = {'scope':'ocr_overlay_candidate_coverage', 'physical_occlusion_status':'not_assessed',
                'retained_targets':len(boxes), 'targets':[]}
    if record.get('ocr_status') != 'ok':
        return dimension(None, 'ocr_'+str(record.get('ocr_status','missing')), evidence)
    if not boxes:
        return dimension(None, 'missing_retained_targets', evidence)
    # Keep the experiment module lightweight until OCR geometry is actually used.
    from quality_metrics import overlap_fraction, watermark_hint, crossing_text_hint
    hint_config = {'watermark_keywords':config.get('watermark_keywords',[])}
    lines = record.get('ocr') or []
    overlay = [line for line in lines if line.get('reason') or line.get('watermark_hint') or
               watermark_hint(line.get('text',''), hint_config)]
    provided = record.get('watermark_candidates')
    if provided is not None:
        overlay.extend(provided)
    for i, box in enumerate(boxes):
        coords=box.get('coords')
        if not coords or len(coords)!=4 or not all(finite(v) for v in coords) or coords[2]<=coords[0] or coords[3]<=coords[1]:
            continue
        crossing=[]
        for line in lines:
            polygon=line.get('polygon')
            if not polygon or len(polygon)!=4 or any(len(p)!=2 or not all(finite(v) for v in p) for p in polygon):
                continue
            if crossing_text_hint(coords,polygon):
                crossing.append(line)
        fraction = overlap_fraction(coords,[line.get('polygon',[]) for line in overlay+crossing])
        evidence['targets'].append({'index':box.get('index',i),'overlap_fraction':fraction,
                                     'score':grade(interpolate(fraction,cfg['text_overlap_anchors']))})
    evidence['ocr_lines']=len(lines)
    evidence['overlay_candidates']=len(overlay)
    values=[x['score'] for x in evidence['targets']]
    return dimension(average(values),'assessed_text_only' if values else 'missing_target_geometry',evidence,
                     len(values)/len(boxes) if boxes else 0.)


SCORE_COLUMNS = {
    'sample_id':('sample_id','样本id'),
    'image_path':('image_path','image_name','filename','图片文件名','图片名称','图片名','文件名'),
    'label_path':('label_path','label_name','xml_name','标签xml文件名','xml文件名','标注文件名','标签文件名','标签名称'),
    'category':('category','class','class_code','类别'),
    'score':('score','评分'), 'b1':('b1',), 'b2':('b2',),
}


def score_columns(row):
    normalized={key.strip().casefold():value for key,value in row.items() if key is not None}
    return {key:next((normalized[alias] for alias in aliases if normalized.get(alias)), '')
            for key,aliases in SCORE_COLUMNS.items()}


def read_annotation_scores(path_or_directory):
    """Parse explicitly supplied CSVs. Identity matching belongs to the caller.

    Multiple runs must be supplied separately to avoid merging their scores.
    Missing b1/b2 stay None. A single ``score`` represents overall annotation
    quality. Invalid records are counted and skipped.
    """
    path=Path(path_or_directory)
    files=[path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file() and p.suffix.lower()=='.csv')
    if not files:
        raise ValueError(f'No annotation rating CSV files: {path}')
    records=[]
    stats={'input_rows':0,'invalid_rows':0,'ignored_files':0}
    for source in files:
        with source.open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.DictReader(stream)
            header=score_columns({key:key for key in reader.fieldnames or []})
            fields={key for key,value in header.items() if value}
            if not fields.intersection({'image_path','sample_id'}) or not fields.intersection({'b1','b2','score'}):
                stats['ignored_files']+=1
                continue
            for number,row in enumerate(reader,2):
                row=score_columns(row)
                stats['input_rows']+=1
                try:
                    image=(row.get('image_path') or '').strip()
                    sample_id=(row.get('sample_id') or '').strip()
                    if not image and not sample_id:
                        raise ValueError('empty image_path')
                    values={}
                    for key in ('b1','b2','score'):
                        raw=(row.get(key) or '').strip()
                        if not raw:
                            values[key]=None
                            continue
                        number_value=float(raw)
                        if not finite(number_value) or number_value!=int(number_value) or not 0<=number_value<=5:
                            raise ValueError('rating outside integer 0..5')
                        values[key]=int(number_value)
                    if all(value is None for value in values.values()):
                        raise ValueError('missing ratings')
                except (ValueError,TypeError,OverflowError):
                    stats['invalid_rows']+=1
                    continue
                records.append({'sample_id':sample_id,'image_path':image,'label_path':(row.get('label_path') or '').strip(),
                                 'category':row['category'].strip(),
                                 'b1':values['b1'],'b2':values['b2'],'legacy_score':values['score'],
                                 'source_file':str(source),'source_row':number})
    stats['accepted_rows']=len(records)
    return {'records':records,'stats':stats}
