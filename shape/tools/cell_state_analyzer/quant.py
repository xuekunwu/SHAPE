"""Label-free measurements of a cell_state_analyzer embedding (no label is read anywhere here).

The tool-side `quant.py` convention (`shape/tools/<tool>/quant.py`): the analyzer evaluates the val
crops of a call with a split with it during training, and the agentic representation verifier and
the RxRx1 benchmark measure the calls with it. Ports of task 116 (SSL hub `agentic_rxrx1` @ f667f94,
H below): `H/workspace_template/tools/quality.py` (sibling-site retrieval, image leakage, the paired
delta), `H/workspace_template/tools/explog.py` (health), `H/harness/data.py` (`assign_splits`, the val
wells) and `H/harness/states.py` (`ADOPTED_PROXY`, the 2-SE rule of `decide_agent_or_frozen`).

The proxy's cell draws are seeded: the tool picks a call's best point on the SELECTION_SEED draws, and every
comparison made between calls (each shown value, delta and SE, the guard and the plateau) reads the
FRESH_SEED draws, so the draws that picked a call's best point never score it.

`meta` is one metadata dict per embedding row (a DataFrame works too). A row may carry plate,
experiment, well and site, each under its canonical or its `*_id` key (`identities`). A well becomes
`<plate>/<well>` and a site `<plate>/<site>` when the plate is known and the value is not already
prefixed by it, so equal well or site names on different plates stay different units.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

import numpy as np

IDENTITY_KEYS = ("plate", "experiment", "well", "site")
# The metadata tiers (`tier_of`), named by the columns they need; only the replicate tier has an
# adopted proxy (ADOPTED_PROXY).
TIERS = {
    "replicate": "site and a plate-qualified well on every row, at least 2 sites in most wells",
    "batch": "site and plate or experiment on every row, no replicate wells",
    "image": "site only",
}
# The label-free proxy a tier's tuning is judged by (H/harness/states.py ADOPTED_PROXY, whose "full"
# tier is the replicate tier here); changed only by a commit the user approves.
ADOPTED_PROXY = {"replicate": "sibling_site_map", "batch": None, "image": None}
SELECTION_SEED = 0          # the proxy draws the tool's best-point selection reads (val_history)
FRESH_SEED = 1              # the comparison draws: every delta, SE, guard and plateau check between calls
MIN_VAL_WELLS = 2           # val wells per plate, at least (assign_val_wells)
HEALTH_ROWS = 4000          # rows the health statistics use (a seeded subset of larger embeddings)
LEAKAGE_CHUNK = 2048        # query rows per similarity block of the leakage neighbours (memory)


def _l2(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _records(meta) -> list:
    if hasattr(meta, "to_dict"):                 # a DataFrame (e.g. an h5ad's obs)
        return meta.to_dict("records")
    return [row if isinstance(row, Mapping) else {} for row in meta]


def _value(row: Mapping, name: str) -> str | None:
    """The first non-blank of `name` / `<name>_id` in one row, as a string; None when absent (NaN too)."""
    for key in (name, f"{name}_id"):
        value = row.get(key)
        if value is None or (isinstance(value, float) and math.isnan(value)) or not str(value).strip():
            continue
        return str(value)
    return None


def identities(meta) -> list[dict]:
    """Per row {"plate", "experiment", "well", "site"}: strings, None where absent; the well and the site
    plate-qualified (module docstring)."""
    out = []
    for row in _records(meta):
        ids = {key: _value(row, key) for key in IDENTITY_KEYS}
        plate = ids["plate"]
        if plate is not None:
            for key in ("well", "site"):
                if ids[key] is not None and not ids[key].startswith(f"{plate}/"):
                    ids[key] = f"{plate}/{ids[key]}"
        out.append(ids)
    return out


def tier_of(meta) -> str:
    """The metadata tier of the rows (TIERS): replicate = site, well and plate on every row and at least
    2 sites in more than half of the wells; batch = site and plate or experiment on every row; else
    image."""
    rows = identities(meta)
    if rows and all(r["site"] and r["well"] and r["plate"] for r in rows):
        sites_of: dict[str, set] = {}
        for r in rows:
            sites_of.setdefault(r["well"], set()).add(r["site"])
        if 2 * sum(len(sites) >= 2 for sites in sites_of.values()) > len(sites_of):
            return "replicate"
    if rows and all(r["site"] and (r["plate"] or r["experiment"]) for r in rows):
        return "batch"
    return "image"


def _rank(*parts) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode("utf-8")).hexdigest()


def assign_val_wells(meta, val_fraction: float = 0.1, seed: int = 0) -> list[str]:
    """"train" / "val" per row: per plate, the first max(MIN_VAL_WELLS, round(val_fraction * n_wells))
    wells of a seeded sha256 order of the plate's (plate-qualified) wells are val, the rest train. Port
    of H/harness/data.py `assign_splits` without the held-out part; no label is read. ValueError when a
    row lacks its plate or well, or a plate has too few wells for its val wells and a train well."""
    if not 0 <= val_fraction < 1:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    rows = identities(meta)
    missing = [key for key in ("plate", "well") if any(r[key] is None for r in rows)]
    if missing:
        raise ValueError(f"the val-well split needs the metadata column(s) {missing} on every crop")
    wells_of_plate: dict[str, set] = {}
    for r in rows:
        wells_of_plate.setdefault(r["plate"], set()).add(r["well"])
    split_of: dict[str, str] = {}
    for plate, members in wells_of_plate.items():
        ordered = sorted(members, key=lambda well: _rank(seed, "val", plate, well))
        n_val = max(MIN_VAL_WELLS, round(val_fraction * len(ordered)))
        if len(ordered) - n_val < 1:
            raise ValueError(f"plate {plate} has too few wells ({len(ordered)}) for {n_val} val wells and a train well")
        for position, well in enumerate(ordered):
            split_of[well] = "val" if position < n_val else "train"
    return [split_of[r["well"]] for r in rows]


def _random_ap(n_pos: int, n_cand: int) -> float:
    """Expected AP of a uniformly random ranking of n_cand candidates with n_pos positives."""
    h = float(np.sum(1.0 / np.arange(1, n_cand + 1)))
    return h / n_cand + (n_pos - 1) / (n_cand - 1) * (1 - h / n_cand)


def sibling_site_map(vectors, meta, *, k: int = 2, draws: int = 20, seed: int = 0, scope: str = "plate") -> dict:
    """Sibling-site retrieval with k-cell site profiles (port of H/workspace_template/tools/quality.py
    `_sibling`). A site's profile is the mean of k of its L2-normalised cells, drawn without replacement
    (all its cells when it has at most k), L2-normalised; each site ranks the other sites of its `scope`
    (plate) by cosine, the positives are the other site(s) of its own well, and a query needs a positive
    and another candidate. The AP of a query is averaged over `draws` seeded draws of the cells; value =
    the mean over the queries ("units", keyed "<well>|<site>"); chance = the mean AP of a random
    ranking of the same queries. Rows without a site, well or scope value take no part; value and
    chance are None without a query. The draws follow the row order (one random key per kept row), so
    the same rows in the same order give the same cells at a seed: a caller recomputing a call's proxy
    passes the val rows in the tool's validation order (the val crops in input order)."""
    x = _l2(vectors)
    rows = identities(meta)
    kept = np.array([i for i, r in enumerate(rows) if r["site"] and r["well"] and r[scope]], dtype=np.int64)
    out = {"value": None, "chance": None, "units": {}, "n_queries": 0, "k": int(k), "draws": int(draws)}
    if not len(kept):
        return out
    keys, first, site_of = np.unique([rows[i]["site"] for i in kept], return_index=True, return_inverse=True)
    wells = np.array([rows[kept[j]]["well"] for j in first])
    scopes = np.array([rows[kept[j]][scope] for j in first])
    queries = []                                 # (site, its candidate sites, which candidates are positives)
    for value in np.unique(scopes):
        group = np.flatnonzero(scopes == value)
        for site in group:
            others = group[group != site]
            positive = wells[others] == wells[site]
            if 0 < int(positive.sum()) < len(others):
                queries.append((site, others, positive))
    if not queries:
        return out
    starts = np.concatenate([[0], np.cumsum(np.bincount(site_of))[:-1]])
    rng = np.random.default_rng(seed)
    ap = np.zeros(len(queries))
    for _ in range(int(draws)):
        order = np.lexsort((rng.random(len(kept)), site_of))         # each site's cells in a random order
        chosen = order[np.arange(len(order)) - starts[site_of[order]] < k]
        profiles = np.zeros((len(keys), x.shape[1]))
        np.add.at(profiles, site_of[chosen], x[kept[chosen]])
        profiles = _l2(profiles)
        for j, (site, others, positive) in enumerate(queries):
            hit = positive[np.argsort(-(profiles[others] @ profiles[site]), kind="stable")]
            ap[j] += float((np.cumsum(hit) / np.arange(1, len(hit) + 1))[hit].mean())
    ap /= int(draws)
    units = {f"{wells[site]}|{keys[site]}": float(value) for (site, _, _), value in zip(queries, ap)}
    chance = [_random_ap(int(positive.sum()), len(others)) for _, others, positive in queries]
    out.update(value=float(ap.mean()), chance=float(np.mean(chance)), units=units, n_queries=len(queries))
    return out


def paired_delta(units_a: Mapping, units_b: Mapping, *, draws: int = 1000, seed: int = 0) -> dict:
    """b - a on the units both have: the mean per-unit difference ("delta"), its standard error from a
    bootstrap over wells ("se"; the well of a unit is the part before "|"; None with < 2 wells), the
    number of shared units ("n") and of their wells ("n_wells"). Port of quality.py `paired_delta`."""
    shared = sorted(set(units_a) & set(units_b))
    if not shared:
        return {"delta": None, "se": None, "n": 0, "n_wells": 0}
    diff = np.array([units_b[u] - units_a[u] for u in shared], dtype=np.float64)
    _, cluster = np.unique([u.split("|")[0] for u in shared], return_inverse=True)
    sums, sizes = np.bincount(cluster, weights=diff), np.bincount(cluster)
    se = None
    if len(sums) >= 2:
        picks = np.random.default_rng(seed).integers(0, len(sums), size=(int(draws), len(sums)))
        se = float(np.std(sums[picks].sum(axis=1) / sizes[picks].sum(axis=1), ddof=1))
    return {"delta": float(diff.mean()), "se": se, "n": len(shared), "n_wells": int(len(sums))}


def beats(units_new: Mapping, units_ref: Mapping) -> dict:
    """{"beats", **paired_delta(units_ref, units_new)}: beats iff new - ref exceeds 2 SE (the 2-SE rule of
    H/harness/states.py `decide_agent_or_frozen`; never without an SE)."""
    delta = paired_delta(units_ref, units_new)
    return {"beats": delta["se"] is not None and delta["delta"] > 2 * delta["se"], **delta}


def health(vectors) -> dict:
    """Collapse indicators of an embedding (port of H/workspace_template/tools/explog.py `health`), on a
    seeded subset of HEALTH_ROWS rows when there are more: effective_rank (exp of the entropy of the
    normalised spectrum of the centred L2-normalised rows; 1 = total collapse), dead_dim_frac (share of
    dimensions with std < 1e-6), mean_cosine (mean cosine to the mean direction; ~1 = a narrow cone) and
    uniformity (Wang & Isola, t = 2, on up to 2000 rows; lower = more uniform; None with one row)."""
    x = np.asarray(vectors, dtype=np.float64)
    if len(x) > HEALTH_ROWS:
        x = x[np.random.default_rng(0).permutation(len(x))[:HEALTH_ROWS]]
    z = _l2(x)
    singular = np.linalg.svd(z - z.mean(axis=0), compute_uv=False)
    p = singular ** 2 / max(float((singular ** 2).sum()), 1e-12)
    nz = p[p > 0]
    sub = z[:2000]
    sq = np.clip(2 - 2 * (sub @ sub.T), 0, None)
    upper = np.triu_indices(len(sub), 1)
    centre = z.mean(axis=0)
    return {
        "effective_rank": float(math.exp(-(nz * np.log(nz)).sum())),
        "dead_dim_frac": float((x.std(axis=0) < 1e-6).mean()),
        "mean_cosine": float((z @ (centre / max(np.linalg.norm(centre), 1e-12))).mean()),
        "uniformity": float(np.log(np.exp(-2 * sq[upper]).mean())) if len(sub) > 1 else None,
    }


def _nearest(x: np.ndarray, k: int) -> np.ndarray:
    """Indices of each row's k nearest other rows of the L2-normalised `x` (cosine, stable order),
    computed in blocks of LEAKAGE_CHUNK query rows."""
    out = np.empty((len(x), k), dtype=np.int64)
    for start in range(0, len(x), LEAKAGE_CHUNK):
        sim = x[start:start + LEAKAGE_CHUNK] @ x.T
        block = np.arange(sim.shape[0])
        sim[block, start + block] = -np.inf
        out[start:start + LEAKAGE_CHUNK] = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    return out


def image_leakage(vectors, meta, k: int = 5) -> float | None:
    """Observed / expected share of same-site cells among a cell's k nearest neighbours (cosine) inside its
    plate (all cells at once when a row has no plate), over the plates with >= 2 sites and >= k + 2
    cells; expected = the cell's other same-site cells over the plate's other cells. 1 = the site adds
    nothing beyond the plate. None when a row has no site or no plate qualifies. Port of quality.py
    `_leakage`."""
    rows = identities(meta)
    if not rows or any(r["site"] is None for r in rows):
        return None
    sites = np.array([r["site"] for r in rows])
    groups = (np.array([r["plate"] for r in rows]) if all(r["plate"] for r in rows)
              else np.zeros(len(rows), dtype=np.int64))
    x = _l2(vectors)
    observed, expected = [], []
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        s = sites[members]
        _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
        if len(counts) < 2 or len(members) < k + 2:
            continue
        nn = _nearest(x[members], k)
        observed.append((s[nn] == s[:, None]).mean(axis=1))
        expected.append((counts[inverse] - 1) / (len(members) - 1))
    if not observed or not np.concatenate(expected).mean() > 0:
        return None
    return float(np.concatenate(observed).mean() / np.concatenate(expected).mean())
