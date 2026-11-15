"""Image/ROI measurements and transparent screening rules. No source-file writes."""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def load_config(path=None):
    config = json.loads(Path(__file__).with_name('quality_rules.json').read_text(encoding='utf-8'))
    if path:
        supplied = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        unknown = set(supplied) - set(config)
        if unknown:
            raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
        for key, value in supplied.items():
            if key == 'class_overrides':
                for label, overrides in value.items():
                    allowed = {'skip_blur','skip_roi_exposure','dark_mean','dark_fraction','bright_mean',
                               'bright_fraction','bright_fraction_extreme','blur_laplacian','blur_tenengrad',
                               'roi_min_raw_side','tiny_target_pixels','small_target_pixels',
                               'small_target_area_fraction','large_box_area_fraction'}
                    if set(overrides) - allowed:
                        raise ValueError(f'Unknown class rule: {label}: {set(overrides) - allowed}')
                    for rule, setting in overrides.items():
                        if rule.startswith('skip_'):
                            if not isinstance(setting, bool):
                                raise ValueError(f'{label}.{rule} must be true or false')
                        elif not isinstance(setting,(int,float)) or not math.isfinite(setting) or setting < 0:
                            raise ValueError(f'{label}.{rule} must be finite and nonnegative')
                        elif 'fraction' in rule and setting > 1:
                            raise ValueError(f'{label}.{rule} must be <= 1')
                    config[key].setdefault(label, {}).update(overrides)
            else:
                config[key] = value
    for key, value in config.items():
        if isinstance(value, (int, float)) and (not math.isfinite(value) or value < 0):
            raise ValueError(f'{key} must be finite and nonnegative')
    for key in ('image_metric_side', 'roi_metric_side', 'roi_min_raw_side', 'ocr_max_side'):
        if config[key] < 8 or int(config[key]) != config[key]:
            raise ValueError(f'{key} must be an integer >= 8')
    for key in ('dark_fraction', 'bright_fraction', 'bright_fraction_extreme',
                'small_target_area_fraction', 'large_box_area_fraction', 'near_ssim',
                'ocr_min_confidence', 'ocr_overlap_fraction', 'near_roi_iou'):
        if not 0 <= config[key] <= 1:
            raise ValueError(f'{key} must be in [0, 1]')
    if not 0 <= config['near_phash_distance'] <= 16:
        raise ValueError('near_phash_distance must be in [0, 16]')
    if config['tiny_target_pixels'] > config['small_target_pixels']:
        raise ValueError('tiny_target_pixels must be <= small_target_pixels')
    return config


def class_rules(config, label):
    return {**config, **config['class_overrides'].get(label, {})}


def resize_down(image, side):
    h, w = image.shape[:2]
    scale = min(1.0, side / max(h, w))
    return (cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1 else image), scale


def measure(image, side):
    """No upscaling. Derivatives at a declared maximum scale; exposure on full ROI."""
    original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray, scale = resize_down(original, side)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    # Limit each histogram to <= 2**20 pixels: float32 counts remain exact,
    # and OpenCV avoids NumPy's full-image int64 temporary for large photos.
    hist = np.zeros(256, dtype=np.int64)
    rows = max(1, (1 << 20) // original.shape[1])
    for start in range(0, original.shape[0], rows):
        hist += cv2.calcHist([original[start:start+rows]], [0], None, [256], [0,256]).ravel().astype(np.int64)
    cumulative = np.cumsum(hist)
    quantiles = np.searchsorted(cumulative, np.array([.1, .5, .9]) * original.size)
    p = hist[hist > 0] / original.size
    return {
        'mean_gray': float(np.dot(hist, np.arange(256)) / original.size),
        'dark_fraction_le10': float(hist[:11].sum() / original.size),
        'bright_fraction_ge245': float(hist[245:].sum() / original.size),
        'p10': int(quantiles[0]), 'p50': int(quantiles[1]), 'p90': int(quantiles[2]),
        'contrast_p90_p10': int(quantiles[2] - quantiles[0]),
        'gray_entropy': float(-(p * np.log2(p)).sum()),
        'laplacian_variance': float(cv2.Laplacian(gray, cv2.CV_32F).var()),
        'tenengrad': float((gx * gx + gy * gy).mean()),
        'measurement_width': gray.shape[1], 'measurement_height': gray.shape[0],
        'measurement_scale': scale,
    }


def exposure_flags(metrics, rules):
    if not all(k in metrics for k in ('mean_gray', 'dark_fraction_le10', 'bright_fraction_ge245')):
        return []
    flags = []
    if metrics['mean_gray'] < rules['dark_mean'] or metrics['dark_fraction_le10'] >= rules['dark_fraction']:
        flags.append('dark')
    if ((metrics['mean_gray'] > rules['bright_mean'] and
         metrics['bright_fraction_ge245'] >= rules['bright_fraction']) or
            metrics['bright_fraction_ge245'] >= rules['bright_fraction_extreme']):
        flags.append('overexposed')
    return flags


def blur_flag(metrics, rules):
    return (metrics.get('laplacian_variance', math.inf) < rules['blur_laplacian'] and
            metrics.get('tenengrad', math.inf) < rules['blur_tenengrad'])


def model_scale(width, height, imgsz, scaleup=True):
    ratio = min(imgsz[0] / height, imgsz[1] / width)
    return ratio if scaleup else min(1.0, ratio)


def box_geometry(coords, width, height, imgsz, scaleup=True):
    if coords is None or len(coords) != 4 or not all(math.isfinite(x) for x in coords):
        return {'valid': False, 'reason': 'missing_or_nonfinite_coordinates'}
    x1, y1, x2, y2 = coords
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return {'valid': False, 'reason': 'reversed_or_zero_area'}
    ratio = model_scale(width, height, imgsz, scaleup)
    return {'valid': True, 'outside': x1 < 0 or y1 < 0 or x2 > width or y2 > height,
            'width_raw': bw, 'height_raw': bh, 'short_raw': min(bw, bh),
            'area_fraction': bw * bh / (width * height),
            'width_at_model': bw * ratio, 'height_at_model': bh * ratio,
            'short_at_model': min(bw, bh) * ratio}


def annotation_signature(objects):
    canonical = sorted(json.dumps([o['class'], o['coords']], ensure_ascii=False,
                                   allow_nan=False, separators=(',', ':')) for o in objects)
    return hashlib.sha256('\n'.join(canonical).encode()).hexdigest()


def read_labels(path, width, height, names, data=None):
    if path is None:
        return {'status': 'missing', 'objects': [], 'signature': '', 'declared_size': None}
    try:
        data = path.read_bytes() if data is None else data
        objects, declared = [], None
        if path.suffix.lower() == '.xml':
            if b'<!DOCTYPE' in data.upper() or b'<!ENTITY' in data.upper():
                raise ValueError('XML DTD/entity is unsupported')
            root = ET.fromstring(data)
            if root.tag != 'annotation':
                raise ValueError('expected VOC annotation root')
            try:
                declared = [int(root.findtext('size/width')), int(root.findtext('size/height'))]
            except (TypeError, ValueError):
                pass
            for obj in root.findall('object'):
                label = (obj.findtext('name') or '').strip()
                try:
                    coords = [float(obj.findtext('bndbox/' + k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
                    if not all(math.isfinite(v) for v in coords):
                        coords = None
                except (TypeError, ValueError):
                    coords = None
                objects.append({'class': label or '__empty_class__', 'coords': coords})
        else:
            for line_number, line in enumerate(data.decode('utf-8-sig').splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f'line {line_number}: expected class xc yc w h for detection')
                index = int(fields[0])
                if index < 0 or (names and index >= len(names)):
                    raise ValueError(f'line {line_number}: class index out of range')
                xc, yc, bw, bh = map(float, fields[1:])
                if not all(math.isfinite(v) for v in (xc, yc, bw, bh)):
                    coords = None
                else:
                    coords = [(xc-bw/2)*width, (yc-bh/2)*height,
                              (xc+bw/2)*width, (yc+bh/2)*height]
                    # Avoid floating-point noise at exact normalized boundaries.
                    coords = [round(v, 8) for v in coords]
                objects.append({'class': names[index] if names else str(index), 'coords': coords})
        return {'status': 'ok', 'objects': objects, 'declared_size': declared,
                'signature': annotation_signature(objects), 'sha256': hashlib.sha256(data).hexdigest()}
    except (ET.ParseError, OSError, ValueError, UnicodeError) as exc:
        return {'status': 'error', 'error': str(exc), 'objects': [], 'signature': '', 'declared_size': None}


def orient_image(image, orientation):
    operations = {2: lambda x: cv2.flip(x, 1), 3: lambda x: cv2.rotate(x, cv2.ROTATE_180),
                  4: lambda x: cv2.flip(x, 0), 5: cv2.transpose,
                  6: lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
                  7: lambda x: cv2.flip(cv2.transpose(x), -1),
                  8: lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE)}
    return operations.get(orientation, lambda x: x)(image)


def image_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8]
    bits = dct > np.median(dct.ravel()[1:])
    bits[0, 0] = False  # Explicitly exclude the DC brightness coefficient.
    dh = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    def pack(array):
        return np.packbits(array.ravel()).tobytes().hex()
    return {'phash': pack(bits), 'dhash': pack(dh[:, 1:] > dh[:, :-1]),
            'gray64': cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).tobytes(),
            'color16': cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA).tobytes()}


def ssim64(a, b):
    a = np.frombuffer(a, np.uint8).reshape(64, 64).astype(np.float32)
    b = np.frombuffer(b, np.uint8).reshape(64, 64).astype(np.float32)
    ma, mb = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    va = cv2.GaussianBlur(a*a, (11, 11), 1.5) - ma*ma
    vb = cv2.GaussianBlur(b*b, (11, 11), 1.5) - mb*mb
    cov = cv2.GaussianBlur(a*b, (11, 11), 1.5) - ma*mb
    value = ((2*ma*mb + 6.5025)*(2*cov + 58.5225) /
             ((ma*ma + mb*mb + 6.5025)*(va + vb + 58.5225)))
    return float(value[5:-5, 5:-5].mean())


def near_match(left, right, config):
    pd = (int(left['phash'], 16) ^ int(right['phash'], 16)).bit_count()
    dd = (int(left['dhash'], 16) ^ int(right['dhash'], 16)).bit_count()
    if pd > config['near_phash_distance'] or dd > config['near_dhash_distance']:
        return None
    a, b = left['width']/left['height'], right['width']/right['height']
    if abs(a-b)/max(a, b) > config['near_aspect_relative_difference']:
        return None
    color = float(np.abs(np.frombuffer(left['color16'], np.uint8).astype(float) -
                        np.frombuffer(right['color16'], np.uint8).astype(float)).mean())
    if color > config['near_color_mae']:
        return None
    score = ssim64(left['gray64'], right['gray64'])
    if score < config['near_ssim']:
        return None
    return {'phash_distance': pd, 'dhash_distance': dd, 'ssim': score, 'color_mae': color}


def roi_appearance(image):
    features = image_features(image)
    return {'phash':features['phash'], 'dhash':features['dhash'],
            'color4':cv2.resize(image,(4,4),interpolation=cv2.INTER_AREA).tobytes().hex()}


def match_target_appearance(left_boxes, right_boxes, left_size, right_size, config):
    """Conservative matching of same-class targets, normalized boxes and local colors."""
    if not left_boxes and not right_boxes:
        return {'roi_verification':'no_eligible_targets', 'roi_pairs':0}
    if len(left_boxes) != len(right_boxes):
        return None
    def normalized(box, size):
        w,h = size
        return np.asarray(box['coords'],float)/[w,h,w,h]
    def iou(a,b):
        inter = np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
        return float(inter / ((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter))
    remaining = set(range(len(right_boxes)))
    checked = 0
    for left in left_boxes:
        a = normalized(left,left_size)
        candidates = sorted(((iou(a,normalized(right_boxes[j],right_size)),j) for j in remaining),reverse=True)
        if not candidates or candidates[0][0] < config['near_roi_iou']:
            return None
        _,j = candidates[0]
        remaining.remove(j)
        l,r = left.get('appearance'),right_boxes[j].get('appearance')
        if not l or not r:
            continue
        if ((int(l['phash'],16)^int(r['phash'],16)).bit_count()>config['near_roi_phash_distance'] or
                (int(l['dhash'],16)^int(r['dhash'],16)).bit_count()>config['near_roi_dhash_distance']):
            return None
        color_mae = np.abs(np.frombuffer(bytes.fromhex(l['color4']),np.uint8).astype(float) -
                           np.frombuffer(bytes.fromhex(r['color4']),np.uint8).astype(float)).mean()
        if color_mae > config['near_roi_color_mae']:
            return None
        checked += 1
    return {'roi_verification':'checked' if checked == len(left_boxes) else 'partial', 'roi_pairs':checked}


def normalize_text(text):
    return re.sub(r'\s+', '', text).casefold()


def text_key(text, polygon, width, height):
    center = np.asarray(polygon).mean(axis=0)
    return json.dumps([normalize_text(text), round(float(center[0])/width*20),
                       round(float(center[1])/height*20)], ensure_ascii=False)


def watermark_hint(text, config):
    value = normalize_text(text)
    if any(normalize_text(word) in value for word in config['watermark_keywords'] if word.strip()):
        return 'configured_keyword'
    if re.search(r'(?:19|20)\d{2}[-/.年]\d{1,2}[-/.月]\d{1,2}|\d{1,2}:\d{2}:\d{2}|https?://|www\.', value):
        return 'timestamp_or_url'
    return ''


def overlap_fraction(coords, polygons):
    """Union of text quadrilaterals inside a target ROI, rasterized on 256x256."""
    x1, y1, x2, y2 = coords
    mask = np.zeros((256, 256), np.uint8)
    for polygon in polygons:
        pts = np.asarray(polygon, dtype=np.float64)
        if pts.shape != (4, 2) or not np.isfinite(pts).all():
            continue
        if pts[:, 0].max() <= x1 or pts[:, 0].min() >= x2 or pts[:, 1].max() <= y1 or pts[:, 1].min() >= y2:
            continue
        pts = (pts - [x1, y1]) * [255/(x2-x1), 255/(y2-y1)]
        pts = np.clip(np.rint(pts), -100000, 100000).astype(np.int32)
        cv2.fillConvexPoly(mask, pts, 1)
    return float(mask.mean())


def crossing_text_hint(coords, polygon):
    """Large text straddling a target edge is an additional overlay review cue."""
    x1, y1, x2, y2 = coords
    pts = np.asarray(polygon, float)
    tx1, ty1 = pts.min(axis=0)
    tx2, ty2 = pts.max(axis=0)
    return ((tx2-tx1) > 1.3*(x2-x1) and (ty2-ty1) > .15*(y2-y1) and
            (tx1 < x1 or tx2 > x2) and tx1 < x2 and tx2 > x1 and ty1 < y2 and ty2 > y1)


class GPUInferenceError(RuntimeError):
    """GPU OCR errors abort the scan instead of turning all images into OCR omissions."""


class OCRReader:
    def __init__(self, config, device='cpu'):
        import threading
        if device not in ('cpu', 'cuda'):
            raise ValueError('OCR device must be cpu or cuda')
        if device == 'cuda':
            import os
            import onnxruntime as ort
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise ValueError('GPU OCR 需要 onnxruntime-gpu 和可用的 CUDAExecutionProvider')
            self._dll_handles = []
            if os.name == 'nt':
                # cuDNN dynamically loads extra engines beyond ORT's preload list.
                from importlib.metadata import distribution, PackageNotFoundError
                directories = set()
                for package in ('nvidia-cudnn-cu12','nvidia-cublas-cu12','nvidia-cufft-cu12',
                                'nvidia-cuda-runtime-cu12','nvidia-cuda-nvrtc-cu12',
                                'nvidia-curand-cu12','nvidia-nvjitlink-cu12'):
                    try:
                        installed = distribution(package)
                    except PackageNotFoundError:
                        continue
                    for file in installed.files or []:
                        if file.suffix.lower() == '.dll':
                            directories.add(str(installed.locate_file(file).parent))
                self._dll_handles = [os.add_dll_directory(p) for p in sorted(directories)]
                if directories:
                    os.environ['PATH'] = os.pathsep.join(sorted(directories))+os.pathsep+os.environ.get('PATH','')
            if hasattr(ort, 'preload_dlls'):
                ort.preload_dlls()
        from rapidocr_onnxruntime import RapidOCR
        # Ship the exact OCR weights with the project, independent of conda's package location.
        model_dir=Path(config.get('ocr_model_dir') or Path(__file__).resolve().parents[2]/'models/ocr')
        model_names={'det_model_path':'ch_PP-OCRv4_det_infer.onnx',
                     'cls_model_path':'ch_ppocr_mobile_v2.0_cls_infer.onnx',
                     'rec_model_path':'ch_PP-OCRv4_rec_infer.onnx'}
        model_paths={key:str((model_dir/name).resolve()) for key,name in model_names.items()}
        missing=[value for value in model_paths.values() if not Path(value).is_file()]
        if missing: raise ValueError('项目OCR模型缺失：'+'; '.join(missing))
        # Avoid CPU oversubscription when the outer pipeline has multiple workers.
        cuda = device == 'cuda'
        self.engine = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1,
                               det_use_cuda=cuda, cls_use_cuda=cuda, rec_use_cuda=cuda,**model_paths)
        sessions = {'det': self.engine.text_det.infer.session,
                    'cls': self.engine.text_cls.infer.session,
                    'rec': self.engine.text_rec.session.session}
        self.providers = {name: session.get_providers() for name, session in sessions.items()}
        if cuda:
            failed = [name for name, providers in self.providers.items()
                      if not providers or providers[0] != 'CUDAExecutionProvider']
            if failed:
                raise ValueError('GPU OCR 初始化失败，以下模型回退到 CPU：'+','.join(failed)+
                                 '。请检查 CUDA 12/cuDNN 9 运行库；本次扫描停止。')
            for session in sessions.values():
                session.disable_fallback()
        self.device = device
        self.lock = threading.Lock()
        self.config = config

    def __call__(self, image):
        small, scale = resize_down(image, self.config['ocr_max_side'])
        try:
            # Shared GPU reader: one model set, serialize mutable RapidOCR state.
            with self.lock:
                result, _ = self.engine(small)
        except Exception as exc:
            if self.device == 'cuda':
                raise GPUInferenceError(f'GPU OCR 推理失败：{exc}') from exc
            raise
        lines = []
        for polygon, text, confidence in result or []:
            if float(confidence) >= self.config['ocr_min_confidence']:
                lines.append({'polygon': (np.asarray(polygon, float)/scale).tolist(),
                              'text': str(text), 'confidence': float(confidence)})
        return lines


def inspect_file(path, label_path, options, config, ocr=None, data=None, sha=None, shared=None):
    result = {'path': str(path), 'label_path': str(label_path or ''), 'ok': False,
              'boxes': [], 'classes': [], 'metrics': {}, 'ocr_status': 'disabled', 'ocr': []}
    try:
        data = path.read_bytes() if data is None else data
        if not data:
            raise ValueError('empty image')
        result['sha256'] = sha or hashlib.sha256(data).hexdigest()
        shared = shared if shared and shared.get('sha256') == result['sha256'] else None
        image = None
        orientation = shared['orientation'] if shared else 1
        def decoded_image():
            nonlocal image
            if image is None:
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                if image is None:
                    raise ValueError('image decode failed')
                if options['box_frame'] == 'display':
                    image = orient_image(image, orientation)
            return image
        if shared:
            for key in ('orientation','raw_width','raw_height','metadata_error'):
                if key in shared:
                    result[key] = shared[key]
            w, h = shared['width'], shared['height']
        else:
            try:
                with Image.open(io.BytesIO(data)) as metadata:
                    orientation = int(metadata.getexif().get(274, 1))
            except (OSError, ValueError, SyntaxError) as exc:
                orientation = 0
                result['metadata_error'] = str(exc)
            result['orientation'] = orientation
            canvas = decoded_image()
            h, w = canvas.shape[:2]
            result['raw_width'], result['raw_height'] = (h,w) if options['box_frame']=='display' and orientation in (5,6,7,8) else (w,h)
        result.update(width=w, height=h, ok=True)
        ann = read_labels(label_path, w, h, options['class_names'])
        result['annotation'] = {k: v for k, v in ann.items() if k != 'objects'}
        result['classes'] = sorted({o['class'] for o in ann['objects']})
        if options['layout'] == 'classified' and not result['classes']:
            result['classes'] = [path.parent.name]
        frame_reasons = []
        if orientation not in (1,) and options['box_frame'] == 'auto':
            frame_reasons.append('EXIF_orientation_requires_box_frame')
        if ann['declared_size'] and ann['declared_size'] != [w, h]:
            frame_reasons.append('XML_size_differs_from_selected_canvas')
        if orientation not in range(1, 9):
            frame_reasons.append('invalid_EXIF_orientation')
        result['frame_ok'] = not frame_reasons
        result['frame_reasons'] = frame_reasons
        result['metrics'] = shared['metrics'] if shared else measure(decoded_image(), config['image_metric_side'])
        regions = dict(shared.get('regions', {})) if shared else {}
        for index, obj in enumerate(ann['objects']):
            geom = box_geometry(obj['coords'], w, h, options['imgsz'], options['scaleup'])
            box = {**obj, **geom, 'index': index, 'metrics': {}}
            box['eligible'] = geom['valid'] and not geom.get('outside', False) and not frame_reasons
            if box['eligible']:
                key = json.dumps(obj['coords'], separators=(',', ':'))
                if key not in regions:
                    x1, y1, x2, y2 = obj['coords']
                    crop = decoded_image()[max(0, int(y1)):min(h, math.ceil(y2)), max(0, int(x1)):min(w, math.ceil(x2))]
                    region = {'metrics': {}}
                    if crop.size and min(crop.shape[:2]) >= 3:
                        region = {'metrics': measure(crop, config['roi_metric_side']), 'appearance': roi_appearance(crop)}
                    regions[key] = region
                box.update(regions[key])
            result['boxes'].append(box)
        if options.get('ocr', bool(ocr)) and shared and shared.get('ocr_status') == 'ok':
            result['ocr'], result['ocr_status'] = shared['ocr'], 'ok'
            result['ocr_device'] = shared.get('ocr_device', 'unknown')
            result['ocr_providers'] = shared.get('ocr_providers', {})
        elif ocr:
            try:
                result['ocr'] = ocr(decoded_image())
                result['ocr_status'] = 'ok'
                if isinstance(getattr(ocr, 'device', None), str):
                    result['ocr_device'] = ocr.device
                if isinstance(getattr(ocr, 'providers', None), dict):
                    result['ocr_providers'] = ocr.providers
            except GPUInferenceError:
                raise
            except Exception as exc:
                result['ocr_status'] = 'error'
                result['ocr_error'] = f'{type(exc).__name__}: {exc}'
        if not shared:
            result['features'] = image_features(decoded_image())
        result['_regions'] = regions
    except (OSError, ValueError, cv2.error, SyntaxError) as exc:
        result['ok'] = False
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result
