import json
import csv
import re
from difflib import SequenceMatcher

PATH_2012 = "data/2012/ZA5900_variables_short.json"
PATH_2022 = "data/2022/ZA10000_variables_short.json"
OUT_FULL = "data/variable_mappings_full.csv"
OUT_CANON = "data/variable_mappings_canonical.csv"


def normalize(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def best_match(label, candidates):
    best = None
    best_score = 0.0
    for k, v in candidates.items():
        score = SequenceMatcher(None, label, v[0]).ratio()
        if score > best_score:
            best_score = score
            best = (k, v)
    return best, best_score


with open(PATH_2012, 'r', encoding='utf-8') as f:
    vars2012 = json.load(f)
with open(PATH_2022, 'r', encoding='utf-8') as f:
    vars2022 = json.load(f)

# Prepare normalized labels
norm2012 = {k: normalize(v) for k, v in vars2012.items()}
norm2022 = {k: normalize(v) for k, v in vars2022.items()}

# Create reverse mapping for 2022 labels to (key, raw label)
cands2022 = {k: (v, norm2022[k]) for k, v in vars2022.items()}

rows_full = []
rows_canon = []

used2022 = set()

for k2012, lab2012 in vars2012.items():
    n2012 = norm2012[k2012]
    # exact normalized label match
    exact = None
    for k2022, n2022 in norm2022.items():
        if n2012 == n2022:
            exact = k2022
            break
    if exact:
        rows_full.append((k2012, lab2012, exact, vars2022[exact], 1.0))
        used2022.add(exact)
        canonical = lab2012
        rows_canon.append((canonical, k2012, exact))
        continue
    # try prefix/suffix structural matches (country-specific, suffix match)
    # match by suffix after underscore if present
    suffix = None
    if '_' in k2012:
        suffix = k2012.split('_')[-1]
        # search for any 2022 key that endswith same suffix
        for k2022 in vars2022.keys():
            if k2022.endswith('_' + suffix) or k2022 == suffix:
                rows_full.append((k2012, lab2012, k2022, vars2022[k2022], 0.95))
                used2022.add(k2022)
                canonical = lab2012
                rows_canon.append((canonical, k2012, k2022))
                suffix = None
                break
        if suffix is None:
            continue
    # fallback fuzzy match on normalized labels
    best_kv = None
    best_score = 0.0
    for k2022, lab2022 in vars2022.items():
        score = SequenceMatcher(None, n2012, normalize(lab2022)).ratio()
        if score > best_score:
            best_score = score
            best_kv = (k2022, lab2022)
    if best_kv:
        rows_full.append((k2012, lab2012, best_kv[0], best_kv[1], round(best_score, 3)))
        used2022.add(best_kv[0])
        # create a short canonical question by choosing the shorter label
        canonical = lab2012 if len(lab2012) <= len(best_kv[1]) else best_kv[1]
        rows_canon.append((canonical, k2012, best_kv[0]))

# Also include 2022-only variables (not matched)
for k2022, lab2022 in vars2022.items():
    if k2022 not in used2022:
        rows_full.append(("", "", k2022, lab2022, 0.0))

# Write full mapping CSV
with open(OUT_FULL, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(["var_2012", "label_2012", "var_2022", "label_2022", "score"])
    for r in rows_full:
        w.writerow(r)

# Write canonical CSV (unique by canonical question)
with open(OUT_CANON, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(["canonical_question", "var_2012", "var_2022"])
    for canonical, v2012, v2022 in rows_canon:
        w.writerow([canonical, v2012, v2022])

print("Wrote:", OUT_FULL, OUT_CANON)
