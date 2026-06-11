"""Configurazione centrale del modello Mondiale 2026.

Tutti i dati strutturali (gironi ufficiali, host, normalizzazione nomi) sono
validati contro il dataset martj42/international_results e la composizione
ufficiale del sorteggio del 5 dicembre 2025.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
SIMS = ROOT / "sims" / "2026"
DOCS = ROOT / "docs"

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# Mondiale 2026: host e date
HOSTS = {"United States", "Canada", "Mexico"}
WC2026_START = "2026-06-11"

# Normalizzazione nomi squadra: chiave = nome alternativo, valore = nome canonico (come nel dataset)
TEAM_NORMALIZE = {
    "Czechia": "Czech Republic",
    "Turkiye": "Turkey",
    "Turkey": "Turkey",
    "USA": "United States",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Cabo Verde": "Cape Verde",
}

# Gironi ufficiali 2026 (team -> lettera girone). Nomi gia canonici (dataset).
GROUPS_2026 = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

TEAM_TO_GROUP = {t: g for g, ts in GROUPS_2026.items() for t in ts}

# Parametri Elo (World Football Elo style)
ELO_BASE = 1500.0
ELO_K_BASE = 40.0            # K base, modulato da importanza torneo e scarto gol
ELO_HOME_ADV = 65.0          # vantaggio campo in punti Elo (annullato su campo neutro)
ELO_INIT_BY_CONF = {}        # opzionale: init per confederazione

# Peso importanza match per K (proxy World Football Elo)
TOURNAMENT_WEIGHT = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "Confederations Cup": 40,
    "UEFA Euro": 50,
    "UEFA Euro qualification": 35,
    "Copa America": 50,
    "African Cup of Nations": 40,
    "AFC Asian Cup": 40,
    "Gold Cup": 35,
    "UEFA Nations League": 35,
    "Friendly": 20,
}
DEFAULT_TOURNAMENT_WEIGHT = 30

# Mondiali di backtest
BACKTEST_WORLD_CUPS = [2002, 2006, 2010, 2014, 2018, 2022]

# Date di inizio dei Mondiali (cutoff backtest = giorno prima)
WC_START_DATES = {
    2002: "2002-05-31",
    2006: "2006-06-09",
    2010: "2010-06-11",
    2014: "2014-06-12",
    2018: "2018-06-14",
    2022: "2022-11-20",
    2026: "2026-06-11",
}

for p in (DATA_RAW, DATA_PROC, SIMS, DOCS):
    p.mkdir(parents=True, exist_ok=True)


def normalize_team(name: str) -> str:
    if not isinstance(name, str):
        return name
    name = name.strip()
    return TEAM_NORMALIZE.get(name, name)
