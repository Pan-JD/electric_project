"""Compare independent score runs on the same audit identity and category."""
from collections import defaultdict
import sqlite3


class ScoreComparison:
    def __init__(self, before, after):
        self.before, self.after = before, after
        self.enabled = before.enabled and after.enabled
        self.info = {'enabled': self.enabled}
        if not self.enabled:
            return
        versions = []
        for index in (before, after):
            with sqlite3.connect(index.path.as_uri() + '?mode=ro', uri=True) as db:
                versions.append({(uid, category): score for uid, category, score in
                                 db.execute('SELECT uid,class,score FROM scores')})
        old, new = versions
        groups = defaultdict(list)
        for key in sorted(old.keys() | new.keys()):
            groups[key[1]].append((old.get(key), new.get(key)))
        self.info = {'enabled': True, 'before': before.info, 'after': after.info,
                     'by_class': {key: self.summarize(rows) for key, rows in groups.items()},
                     'overall': self.summarize([r for rows in groups.values() for r in rows])}

    @staticmethod
    def summarize(rows):
        paired = [(a, b) for a, b in rows if a is not None and b is not None]
        matrix = [[0] * 6 for _ in range(6)]
        for a, b in paired:
            matrix[a][b] += 1
        count = len(paired)
        return {'paired': count, 'up': sum(b > a for a, b in paired),
                'down': sum(b < a for a, b in paired), 'same': sum(b == a for a, b in paired),
                'before_only': sum(b is None for a, b in rows),
                'after_only': sum(a is None for a, b in rows), 'matrix': matrix,
                'before_mean': sum(a for a, b in paired)/count if count else None,
                'after_mean': sum(b for a, b in paired)/count if count else None,
                'mean_delta': sum(b-a for a, b in paired)/count if count else None}

    def values(self, db, uid, category=''):
        if not self.enabled:
            return []
        before = {r['class']: r for r in self.before.values(db, uid, category)}
        after = {r['class']: r for r in self.after.values(db, uid, category)}
        result = []
        for name in sorted(before.keys() | after.keys()):
            a, b = before.get(name), after.get(name)
            result.append({'class': name, 'before': a, 'after': b,
                           'delta': b['score']-a['score'] if a and b else None})
        return result

    def condition(self, change='', before_score='', after_score='', category=''):
        rules = {'': '1=1', 'any': '1=1', 'paired': 'a.score IS NOT NULL AND b.score IS NOT NULL',
                 'changed': 'a.score != b.score', 'up': 'b.score > a.score',
                 'down': 'b.score < a.score', 'same': 'b.score = a.score',
                 'before_only': 'b.score IS NULL', 'after_only': 'a.score IS NULL'}
        if change not in rules or any(s not in ('', '0','1','2','3','4','5') for s in (before_score, after_score)):
            raise ValueError('未知评分变化或前后分数；分数应为 0～5')
        if not (change or before_score or after_score):
            return '', []
        if not self.enabled:
            raise ValueError('请同时加载改进前与改进后的评分 CSV')
        conditions, params = ['k.uid=u.uid', rules[change]], []
        for field, value in [('k.class', category), ('a.score', before_score), ('b.score', after_score)]:
            if value:
                conditions.append(field+'=?')
                params.append(value)
        # UNION keys retain single-sided records; matching always includes category.
        sql = '''EXISTS (SELECT 1 FROM
            (SELECT uid,class FROM scoredb.scores UNION SELECT uid,class FROM afterscoredb.scores) k
            LEFT JOIN scoredb.scores a ON a.uid=k.uid AND a.class=k.class
            LEFT JOIN afterscoredb.scores b ON b.uid=k.uid AND b.class=k.class
            WHERE ''' + ' AND '.join(conditions) + ')'
        return sql, params
