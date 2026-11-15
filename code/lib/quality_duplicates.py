"""Per-class exact groups and representative-based perceptual duplicate candidates."""
from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache

from quality_metrics import match_target_appearance, near_match
from quality_logging import update_progress


class BKTree:
    """Metric index for 64-bit Hamming distance; stores only cluster representatives."""
    def __init__(self):
        self.root = None

    def add(self, value, key):
        if self.root is None:
            self.root = [value, [key], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(key)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [key], {}]
                return
            node = node[2][distance]

    def query(self, value, radius):
        stack = [self.root] if self.root else []
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                for key in node[1]:
                    yield distance, key
            stack.extend(child for edge, child in node[2].items()
                         if distance-radius <= edge <= distance+radius)


def find_duplicates(db, config, near_enabled, max_candidates, progress):
    db.executescript('''
        CREATE TABLE duplicate_members (
            kind TEXT, class TEXT, group_id TEXT, uid TEXT, representative_uid TEXT,
            is_extra INTEGER, evidence TEXT);
        CREATE TABLE near_search (uid TEXT, class TEXT, status TEXT, comparisons INTEGER);
    ''')
    classes = [r[0] for r in db.execute('SELECT DISTINCT class FROM unit_classes ORDER BY class')]

    @lru_cache(maxsize=1024)
    def features(sha):
        row = db.execute('SELECT phash,dhash,gray64,color16,width,height FROM features WHERE sha=?', (sha,)).fetchone()
        return dict(zip(('phash','dhash','gray64','color16','width','height'), row)) if row else None

    @lru_cache(maxsize=2048)
    def targets(uid, label):
        return [json.loads(r[0]) for r in db.execute(
            'SELECT payload FROM boxes WHERE uid=? AND class=? AND eligible=1 ORDER BY object_index', (uid,label))]

    def flag(uid, issue, evidence, severity='review'):
        db.execute('INSERT INTO findings VALUES (?,?,?,?,?,?)',
                   (uid, issue, 'candidate' if issue == 'near_duplicate' else 'measured', severity,
                    -1, json.dumps(evidence, ensure_ascii=False)))

    completed = 0
    for ci, label in enumerate(classes, 1):
        note = f'类别 {ci}/{len(classes)}：{label}'
        update_progress(completed, note=note+'，精确重复分组')
        rows = db.execute('''SELECT u.uid,u.sha FROM units u JOIN unit_classes c USING(uid)
                             WHERE c.class=? AND u.ok=1 ORDER BY (u.width*u.height) DESC,u.uid''', (label,)).fetchall()
        by_sha = defaultdict(list)
        for uid, sha in rows:
            by_sha[sha].append(uid)
        for sha, members in by_sha.items():
            for uid in members:
                db.execute('INSERT OR IGNORE INTO evaluated VALUES (?,?)', (uid, 'exact_duplicate'))
            if len(members) < 2:
                continue
            group = 'exact_' + sha[:20]
            for index, uid in enumerate(members):
                extra = index > 0
                evidence = {'class': label, 'group_id': group, 'representative_uid': members[0],
                            'members': len(members), 'is_extra': extra}
                db.execute('INSERT INTO duplicate_members VALUES (?,?,?,?,?,?,?)',
                           ('exact', label, group, uid, members[0], extra, json.dumps(evidence)))
                if extra:
                    flag(uid, 'exact_duplicate', evidence)
        if not near_enabled:
            db.commit()
            completed += len(rows)
            update_progress(completed, note=note+'，已完成精确重复检查')
            continue
        tree, groups = BKTree(), {}
        searched = 0
        for step, (sha, members) in enumerate(by_sha.items(), 1):
            current = features(sha)
            if current is None:
                for uid in members:
                    db.execute('INSERT INTO near_search VALUES (?,?,?,?)', (uid, label, 'no_features', 0))
                searched += len(members)
                update_progress(completed+searched, note=note+f'，近重复 {step}/{len(by_sha)} 个内容哈希')
                continue
            found, matched, examined, truncated = None, None, 0, False
            for distance, representative_sha in tree.query(int(current['phash'], 16), config['near_phash_distance']):
                if examined >= max_candidates:
                    truncated = True
                    break
                examined += 1
                representative = features(representative_sha)
                matched = near_match(current, representative, config)
                if matched:
                    roi = match_target_appearance(targets(members[0],label),
                        targets(by_sha[representative_sha][0],label),
                        (current['width'],current['height']),
                        (representative['width'],representative['height']),config)
                    if roi is not None:
                        matched.update(roi)
                        found = representative_sha
                        break
            for uid in members:
                db.execute('INSERT INTO near_search VALUES (?,?,?,?)',
                           (uid, label, 'candidate_limit' if truncated else 'complete', examined))
                if not truncated:
                    db.execute('INSERT OR IGNORE INTO evaluated VALUES (?,?)', (uid, 'near_duplicate'))
            if found is None:
                tree.add(int(current['phash'], 16), sha)
                groups[sha] = [(sha, None)]
            else:
                groups[found].append((sha, matched))
            searched += len(members)
            update_progress(completed+searched, note=note+f'，近重复 {step}/{len(by_sha)} 个内容哈希')
        update_progress(note=note+'，写入重复分组')
        for representative_sha, entries in groups.items():
            if len(entries) < 2:
                continue
            representative_uid = by_sha[representative_sha][0]
            group = 'near_' + representative_sha[:20]
            for sha, match in entries:
                for index, uid in enumerate(by_sha[sha]):
                    # Exact extras already have their own count. Count one new unique hash here.
                    extra = sha != representative_sha and index == 0
                    evidence = {'class': label, 'group_id': group, 'representative_uid': representative_uid,
                                'unique_image_hashes': len(entries), 'is_extra_after_exact': extra,
                                **(match or {})}
                    db.execute('INSERT INTO duplicate_members VALUES (?,?,?,?,?,?,?)',
                               ('near', label, group, uid, representative_uid, extra, json.dumps(evidence)))
                    if extra:
                        flag(uid, 'near_duplicate', evidence)
        db.commit()
        completed += len(rows)
        update_progress(completed, note=note+'，已完成')
    db.execute('CREATE INDEX duplicate_unit_idx ON duplicate_members(uid)')
    db.commit()
    progress(f'重复检查完成：{len(classes)} 类')
