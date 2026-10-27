"""압축을 푼 CVAT YOLO 데이터에서 이미지/라벨 쌍을 검사하고 80:20 복사 분할.
Ultralytics YOLO Detection 1.0 및 기존 YOLO 1.1 디렉터리를 지원.
원본은 변경하지 않는다. 같은 촬영 시퀀스가 매우 유사하면 수동 그룹 분할 권장.
"""
import argparse
import hashlib
import json
import math
import random
import shutil
from pathlib import Path


def prepare(source, destination, seed=42):
    from PIL import Image
    import yaml
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir():
        raise ValueError('Source folder does not exist: ' + str(source))
    if destination.exists():
        raise ValueError('Output already exists. Use a new --output folder to avoid overwriting.')
    if source == destination or source in destination.parents:
        raise ValueError('Output must be outside the source folder.')
    files = [p for p in source.rglob('*') if p.is_file() and '__MACOSX' not in p.parts and not p.name.startswith('.')]
    # hand 한 클래스인지 확인: class 0의 이름을 임의로 추측하지 않는다.
    known_names = []
    for p in files:
        if p.name == 'obj.names':
            known_names.append([s.strip() for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()])
        elif p.suffix.lower() in {'.yaml', '.yml'}:
            obj = yaml.safe_load(p.read_text(encoding='utf-8-sig'))
            if isinstance(obj, dict) and 'names' in obj:
                names = obj['names']
                if isinstance(names, dict):
                    if set(map(str, names)) != {'0'}:
                        raise ValueError('Expected class 0 only: ' + str(p))
                    names = [next(iter(names.values()))]
                known_names.append(names)
    if not known_names or any(names != ['hand'] for names in known_names):
        raise ValueError('Expected obj.names or data.yaml defining exactly one class: hand.')
    images = sorted(p for p in files if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if len(images) < 2:
        raise ValueError('Need at least 2 images, preferably the full Week 6 dataset of 20 or more.')
    labels = {}
    for p in files:
        if p.suffix.lower() == '.txt':
            labels.setdefault(p.stem, []).append(p)
    stems, hashes, pairs = set(), set(), []
    for p in images:
        if p.stem in stems:
            raise ValueError('Duplicate image stem: ' + p.stem)
        stems.add(p.stem)
        candidates = labels.get(p.stem, [])
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one matching label for {p.name}; found {len(candidates)}')
        label = candidates[0]
        with Image.open(p) as image:
            image.load()
            if image.getexif().get(274, 1) != 1:
                raise ValueError('Image has EXIF rotation. Verify/normalize image AND boxes before training: ' + p.name)
            digest = hashlib.sha256(str(image.size).encode() + image.convert('RGB').tobytes()).hexdigest()
        if digest in hashes:
            raise ValueError('Duplicate image content: ' + p.name)
        hashes.add(digest)
        lines = [s for s in label.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        for row in lines:
            values = row.split()
            if len(values) != 5:
                raise ValueError('Expected detection label: class cx cy w h: ' + str(label))
            cls, cx, cy, w, h = map(float, values)
            if not all(math.isfinite(v) for v in (cls, cx, cy, w, h)) or cls != 0:
                raise ValueError('Invalid class or nonfinite value: ' + str(label))
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError('Coordinates must be normalized: ' + str(label))
            if min(cx-w/2, cy-h/2) < -1e-5 or max(cx+w/2, cy+h/2) > 1+1e-5:
                raise ValueError('Box extends outside image: ' + str(label))
        pairs.append((p, label, len(lines)))
    random.Random(seed).shuffle(pairs)
    count_val = max(1, min(len(pairs)-1, round(len(pairs)*0.2)))
    splits = {'train': pairs[count_val:], 'val': pairs[:count_val]}
    for name, items in splits.items():
        if sum(n for _, _, n in items) == 0:
            raise ValueError(f'No hand boxes in {name}. Review dataset/split.')
    manifest = dict(seed=seed, method='seeded image-level 80/20 split', source=str(source), splits={})
    for name, items in splits.items():
        for folder in ('images', 'labels'):
            (destination / folder / name).mkdir(parents=True, exist_ok=True)
        manifest['splits'][name] = []
        for img, label, count in items:
            shutil.copy2(img, destination / 'images' / name / img.name)
            shutil.copy2(label, destination / 'labels' / name / (img.stem + '.txt'))
            manifest['splits'][name].append({'image': img.name, 'boxes': count})
        print(f'{name}: {len(items)} images, {sum(n for _, _, n in items)} boxes')
    config = {'path': str(destination), 'train': 'images/train', 'val': 'images/val', 'names': {0: 'hand'}}
    (destination / 'data.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    (destination / 'split_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved:', destination / 'data.yaml')
    print('Small validation sets give unstable metrics. Keep the tracking video separate from training images.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='raw_dataset')
    p.add_argument('--output', default='dataset')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    prepare(a.source, a.output, a.seed)
