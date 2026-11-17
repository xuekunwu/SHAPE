"""V1: the prepared folder.  usage: python check_prepare.py PREPARED_DIR"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
p = Path(sys.argv[1])
pub, priv = json.loads((p / "public/images.json").read_text()), json.loads((p / "private/images.json").read_text())
rec, splits = json.loads((p / "PREPARE.json").read_text()), json.loads((p / "private/splits.json").read_text())
units = defaultdict(set)
for r in pub: units[r["well"]].add(r["usage"])
per_plate = defaultdict(Counter)
for r in pub: per_plate[r["plate"]][r["usage"]] += 1
unit_plate = {r["well"]: r["plate"] for r in pub}
wells_plate = defaultdict(Counter)
for w, u in units.items(): wells_plate[unit_plate[w]][next(iter(u))] += 1
print("images", len(pub), "private", len(priv), "usages", dict(Counter(r["usage"] for r in pub)))
print("plate-wells", len(units), "crossing splits:", sum(len(v) != 1 for v in units.values()))
print("per plate images", {k: dict(v) for k, v in per_plate.items()})
print("per plate plate-wells", {k: dict(v) for k, v in wells_plate.items()})
n_train = sum(r["usage"] == "development_train" for r in pub)
print("image share train %.4f val %.4f" % (n_train / len(pub), 1 - n_train / len(pub)))
print("all rows two-channel:", all(r["shape"][0] == 2 and r["raw_file"] for r in pub), "| days per plate-well max", max(Counter((v["plate"], v["well"]) for v in priv.values()).values()))
print("record", {k: rec[k] for k in ("status", "protocol", "images", "plate_wells", "variant")})
