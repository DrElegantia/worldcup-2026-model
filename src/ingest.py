"""Ingest dei dati grezzi e costruzione della tabella match pulita.

Fonte primaria: martj42/international_results (ogni match internazionale dal 1872,
con data, squadre, punteggio, torneo, citta, paese, campo neutro). Licenza aperta.

Produce:
  data/processed/matches.parquet  -> storico match giocati (per training/Elo)
  data/processed/wc2026.json      -> struttura torneo 2026 (gironi, fixture)
"""
import json
import sys
import urllib.request

import pandas as pd

from config import (RESULTS_URL, DATA_RAW, DATA_PROC, normalize_team,
                    GROUPS_2026, TEAM_TO_GROUP, TOURNAMENT_WEIGHT,
                    DEFAULT_TOURNAMENT_WEIGHT, HOSTS)


def download(force=False):
    dest = DATA_RAW / "results.csv"
    if dest.exists() and not force:
        return dest
    print(f"[ingest] download {RESULTS_URL}", file=sys.stderr)
    urllib.request.urlretrieve(RESULTS_URL, dest)
    return dest


def load_overrides():
    """Risultati verificati inseriti a mano, stopgap finche' la fonte (martj42) non
    pubblica i punteggi (lag del maintainer). data/results_override.csv:
    date,home,away,home_score,away_score. Nomi normalizzati come il dataset.
    Si applicano SOLO dove la fonte ha punteggio mancante (cedono alla fonte)."""
    path = DATA_RAW.parent / "results_override.csv"
    if not path.exists():
        return {}
    ov = pd.read_csv(path)
    out = {}
    for _, r in ov.iterrows():
        key = (str(r["date"]), normalize_team(str(r["home"])), normalize_team(str(r["away"])))
        out[key] = (int(r["home_score"]), int(r["away_score"]))
    return out


def load_raw():
    path = download()
    df = pd.read_csv(path)
    df["home_team"] = df["home_team"].map(normalize_team)
    df["away_team"] = df["away_team"].map(normalize_team)
    df["date"] = pd.to_datetime(df["date"])
    # applica override su righe con punteggio mancante (la fonte ha precedenza)
    overrides = load_overrides()
    if overrides:
        dstr = df["date"].dt.strftime("%Y-%m-%d")
        na = df["home_score"].isna() | df["away_score"].isna()
        applied = 0
        for (d, h, a), (hs, as_) in overrides.items():
            mask = na & (dstr == d) & (df["home_team"] == h) & (df["away_team"] == a)
            if mask.any():
                df.loc[mask, "home_score"] = hs
                df.loc[mask, "away_score"] = as_
                applied += int(mask.sum())
        if applied:
            print(f"[ingest] override risultati applicati: {applied}", file=sys.stderr)
    return df


def build_matches():
    """Match giocati (punteggio noto), con peso torneo. Salva parquet."""
    df = load_raw()
    played = df.dropna(subset=["home_score", "away_score"]).copy()
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)
    played["weight"] = played["tournament"].map(
        lambda t: TOURNAMENT_WEIGHT.get(t, DEFAULT_TOURNAMENT_WEIGHT))
    played["is_wc"] = played["tournament"] == "FIFA World Cup"
    played["neutral"] = played["neutral"].astype(str).str.upper().eq("TRUE")
    played = played.sort_values("date").reset_index(drop=True)
    cols = ["date", "home_team", "away_team", "home_score", "away_score",
            "tournament", "city", "country", "neutral", "weight", "is_wc"]
    out = played[cols]
    # elevazione/altitudine sede + penalita acclimatamento (GeoNames, statico)
    try:
        import geo
        out, _ = geo.add_elevation(out)
    except Exception as e:
        print(f"[ingest][WARN] altitudine non disponibile: {e}", file=sys.stderr)
        for c in ["venue_alt", "alt_pen_home", "alt_pen_away", "d_alt_pen"]:
            out[c] = 0.0
    out.to_parquet(DATA_PROC / "matches.parquet", index=False)
    print(f"[ingest] matches giocati: {len(out)}  "
          f"({out.date.min().date()} -> {out.date.max().date()})", file=sys.stderr)
    return out


def build_wc2026():
    """Estrae fixture 2026 dal dataset e costruisce struttura gironi validata."""
    df = load_raw()
    wc = df[(df.tournament == "FIFA World Cup") & (df.date.dt.year == 2026)].copy()
    fixtures = []
    for _, r in wc.iterrows():
        h, a = r.home_team, r.away_team
        g = TEAM_TO_GROUP.get(h)
        ga = TEAM_TO_GROUP.get(a)
        stage = "group" if g is not None and g == ga else "knockout"
        fixtures.append({
            "date": r.date.strftime("%Y-%m-%d"),
            "home": h, "away": a, "city": r.city, "country": r.country,
            "group": g if stage == "group" else None,
            "stage": stage,
            "played": pd.notna(r.home_score),
            "home_score": int(r.home_score) if pd.notna(r.home_score) else None,
            "away_score": int(r.away_score) if pd.notna(r.away_score) else None,
        })
    # validazione: ogni girone deve avere 4 team e 6 partite
    valid = True
    for g, teams in GROUPS_2026.items():
        gm = [f for f in fixtures if f["group"] == g]
        if len(set(teams)) != 4 or len(gm) != 6:
            print(f"[ingest][WARN] girone {g}: teams={len(set(teams))} match={len(gm)}",
                  file=sys.stderr)
            valid = False
    struct = {
        "groups": GROUPS_2026,
        "team_to_group": TEAM_TO_GROUP,
        "hosts": sorted(HOSTS),
        "fixtures": fixtures,
        "valid": valid,
    }
    with open(DATA_PROC / "wc2026.json", "w") as f:
        json.dump(struct, f, ensure_ascii=False, indent=2)
    print(f"[ingest] wc2026: {len(fixtures)} fixture, struttura valida={valid}",
          file=sys.stderr)
    return struct


if __name__ == "__main__":
    build_matches()
    build_wc2026()
