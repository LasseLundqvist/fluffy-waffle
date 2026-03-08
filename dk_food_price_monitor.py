#!/usr/bin/env python3
"""
DK Fødevarepris-Monitor

Højfrekvent inflationsmåling baseret på 50 tracer-varer fra danske supermarkeder.

Arkitektur:
  1. Scraper : Henter priser fra nemlig.com (HTML/JSON-LD – ingen API-nøgle krævet)
  2. Normalisering: Enhedspris-beregning (kr/kg, kr/L), keyword-matching
  3. Indeksberegning: Laspeyres prisindeks med COICOP-vægte fra DST
  4. Validering: Benchmarking mod Eurostat HICP månedlig data
  5. Output: JSON, CSV og rapport

Brug:
  python dk_food_price_monitor.py scrape     # Kør daglig scraping (nemlig.com)
  python dk_food_price_monitor.py index      # Beregn prisindeks
  python dk_food_price_monitor.py report     # Generer rapport
  python dk_food_price_monitor.py eurostat   # Hent Eurostat-data
  python dk_food_price_monitor.py full       # Hele pipeline
  python dk_food_price_monitor.py demo       # Syntetiske data (test uden net)
  python dk_food_price_monitor.py basket     # Vis varekurven

Krav:
  pip install requests beautifulsoup4 lxml

Forfatter: Prognosecenteret
"""

import json
import csv
import os
import re
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── Konfiguration ────────────────────────────────────────────────────────────

DATA_DIR = Path("./data/price_monitor")
DATA_DIR.mkdir(parents=True, exist_ok=True)

PRICES_DIR = DATA_DIR / "daily_prices"
PRICES_DIR.mkdir(exist_ok=True)

INDEX_DIR = DATA_DIR / "indices"
INDEX_DIR.mkdir(exist_ok=True)

EUROSTAT_DIR = DATA_DIR / "eurostat"
EUROSTAT_DIR.mkdir(exist_ok=True)

LOG_FILE = DATA_DIR / "monitor.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("dk_price_monitor")

# ── COICOP-vægtet varekurv: 50 tracer-varer ─────────────────────────────────
#
# Struktureret efter COICOP 2018 (CP01 Fødevarer og ikke-alkoholiske drikkevarer)
# Vægte baseret på DST's forbrugerprisindeks, nedskaleret til 50 varer.
# 'unit' bruges til normalisering af enhedspris.
#
# Kilder til vægte:
#   - DST PRIS111 (forbrugerprisindeks, detaljeret COICOP)
#   - Eurostat prc_hicp_inw (HICP vægte for Danmark)

BASKET = [
    # ── CP0111 Brød og kornprodukter (samlet vægt: ~12%) ──
    {"id": "B01", "name": "Rugbrød, skiveskåret",       "coicop": "CP01111", "unit": "kg",  "weight": 2.5, "search": "rugbrød skåret"},
    {"id": "B02", "name": "Franskbrød / hvedebrød",     "coicop": "CP01111", "unit": "stk", "weight": 1.5, "search": "franskbrød hvede"},
    {"id": "B03", "name": "Havregryn",                  "coicop": "CP01112", "unit": "kg",  "weight": 1.0, "search": "havregryn"},
    {"id": "B04", "name": "Pasta, spaghetti",           "coicop": "CP01113", "unit": "kg",  "weight": 1.5, "search": "spaghetti pasta"},
    {"id": "B05", "name": "Ris, langkornet",            "coicop": "CP01113", "unit": "kg",  "weight": 1.0, "search": "ris langkornet"},
    {"id": "B06", "name": "Hvedemel",                   "coicop": "CP01112", "unit": "kg",  "weight": 0.8, "search": "hvedemel"},
    {"id": "B07", "name": "Morgenbrød / boller",        "coicop": "CP01111", "unit": "stk", "weight": 1.2, "search": "morgenboller"},
    {"id": "B08", "name": "Müsli / granola",            "coicop": "CP01112", "unit": "kg",  "weight": 1.0, "search": "müsli granola"},

    # ── CP0112 Kød (samlet vægt: ~16%) ──
    {"id": "K01", "name": "Hakket oksekød 8-12%",       "coicop": "CP01121", "unit": "kg",  "weight": 3.0, "search": "hakket oksekød"},
    {"id": "K02", "name": "Kyllingebryst",              "coicop": "CP01122", "unit": "kg",  "weight": 3.5, "search": "kyllingebryst"},
    {"id": "K03", "name": "Svinekotelet / nakkefilet",  "coicop": "CP01123", "unit": "kg",  "weight": 2.5, "search": "svinekotelet nakkefilet"},
    {"id": "K04", "name": "Pålægsskinke",               "coicop": "CP01123", "unit": "kg",  "weight": 2.0, "search": "pålægsskinke"},
    {"id": "K05", "name": "Hakket svinekød",            "coicop": "CP01123", "unit": "kg",  "weight": 2.0, "search": "hakket svinekød"},
    {"id": "K06", "name": "Spegepølse / salami",        "coicop": "CP01123", "unit": "kg",  "weight": 1.5, "search": "spegepølse salami"},
    {"id": "K07", "name": "Bacon",                      "coicop": "CP01123", "unit": "kg",  "weight": 1.5, "search": "bacon"},

    # ── CP0113 Fisk og skaldyr (samlet vægt: ~4%) ──
    {"id": "F01", "name": "Laks, frisk filet",          "coicop": "CP01131", "unit": "kg",  "weight": 2.0, "search": "laksefilet frisk"},
    {"id": "F02", "name": "Tun på dåse",                "coicop": "CP01132", "unit": "kg",  "weight": 1.0, "search": "tun dåse"},
    {"id": "F03", "name": "Rejer, pillede",             "coicop": "CP01131", "unit": "kg",  "weight": 1.0, "search": "rejer pillede"},

    # ── CP0114 Mælk, ost og æg (samlet vægt: ~14%) ──
    {"id": "M01", "name": "Letmælk 1.5%",               "coicop": "CP01141", "unit": "L",   "weight": 3.0, "search": "letmælk"},
    {"id": "M02", "name": "Smør, Lurpak el.lign.",      "coicop": "CP01142", "unit": "kg",  "weight": 2.0, "search": "smør lurpak"},
    {"id": "M03", "name": "Ost, mild/mellemlagret",     "coicop": "CP01143", "unit": "kg",  "weight": 2.5, "search": "ost mild mellemlagret"},
    {"id": "M04", "name": "Æg, 10 stk frilandsæg",     "coicop": "CP01144", "unit": "stk", "weight": 1.5, "search": "æg frilands 10"},
    {"id": "M05", "name": "Yoghurt naturel",            "coicop": "CP01141", "unit": "L",   "weight": 1.5, "search": "yoghurt naturel"},
    {"id": "M06", "name": "Fløde 38%",                  "coicop": "CP01141", "unit": "L",   "weight": 1.0, "search": "fløde 38"},
    {"id": "M07", "name": "Skyr",                       "coicop": "CP01141", "unit": "kg",  "weight": 1.5, "search": "skyr"},

    # ── CP0115 Olier og fedtstoffer (samlet vægt: ~3%) ──
    {"id": "O01", "name": "Rapsolie",                   "coicop": "CP01151", "unit": "L",   "weight": 1.5, "search": "rapsolie"},
    {"id": "O02", "name": "Olivenolie",                 "coicop": "CP01151", "unit": "L",   "weight": 1.5, "search": "olivenolie"},

    # ── CP0116 Frugt (samlet vægt: ~8%) ──
    {"id": "FR01", "name": "Bananer",                   "coicop": "CP01161", "unit": "kg",  "weight": 2.0, "search": "bananer"},
    {"id": "FR02", "name": "Æbler",                     "coicop": "CP01161", "unit": "kg",  "weight": 2.0, "search": "æbler"},
    {"id": "FR03", "name": "Appelsiner / Clementiner",  "coicop": "CP01161", "unit": "kg",  "weight": 2.0, "search": "appelsiner clementiner"},
    {"id": "FR04", "name": "Jordbær / blåbær (sæson)", "coicop": "CP01161", "unit": "kg",  "weight": 2.0, "search": "jordbær blåbær"},

    # ── CP0117 Grøntsager (samlet vægt: ~10%) ──
    {"id": "G01", "name": "Kartofler",                  "coicop": "CP01171", "unit": "kg",  "weight": 2.0, "search": "kartofler"},
    {"id": "G02", "name": "Tomater",                    "coicop": "CP01172", "unit": "kg",  "weight": 2.0, "search": "tomater"},
    {"id": "G03", "name": "Løg",                        "coicop": "CP01172", "unit": "kg",  "weight": 1.0, "search": "løg gule"},
    {"id": "G04", "name": "Gulerødder",                 "coicop": "CP01172", "unit": "kg",  "weight": 1.5, "search": "gulerødder"},
    {"id": "G05", "name": "Agurk",                      "coicop": "CP01172", "unit": "stk", "weight": 1.0, "search": "agurk"},
    {"id": "G06", "name": "Iceberg salat",              "coicop": "CP01172", "unit": "stk", "weight": 1.0, "search": "iceberg salat"},
    {"id": "G07", "name": "Peberfrugt",                 "coicop": "CP01172", "unit": "kg",  "weight": 1.5, "search": "peberfrugt"},

    # ── CP0118 Sukker, syltetøj, chokolade (samlet vægt: ~6%) ──
    {"id": "S01", "name": "Sukker, hvidt",              "coicop": "CP01181", "unit": "kg",  "weight": 1.0, "search": "sukker hvidt"},
    {"id": "S02", "name": "Chokolade, mælke-",         "coicop": "CP01182", "unit": "kg",  "weight": 2.0, "search": "mælkechokolade"},
    {"id": "S03", "name": "Marmelade / syltetøj",       "coicop": "CP01183", "unit": "kg",  "weight": 1.0, "search": "marmelade syltetøj"},
    {"id": "S04", "name": "Is, 1L bøtte",               "coicop": "CP01182", "unit": "L",   "weight": 1.5, "search": "is vanille bøtte"},

    # ── CP0119 Andre fødevarer (samlet vægt: ~7%) ──
    {"id": "A01", "name": "Flåede tomater, dåse",       "coicop": "CP01191", "unit": "kg",  "weight": 1.5, "search": "flåede tomater dåse"},
    {"id": "A02", "name": "Ketchup",                    "coicop": "CP01192", "unit": "kg",  "weight": 1.0, "search": "ketchup"},
    {"id": "A03", "name": "Salt",                       "coicop": "CP01193", "unit": "kg",  "weight": 0.5, "search": "salt"},
    {"id": "A04", "name": "Kokosmælk, dåse",            "coicop": "CP01191", "unit": "L",   "weight": 1.0, "search": "kokosmælk dåse"},

    # ── CP0121-0122 Kaffe, te, drikkevarer (samlet vægt: ~10%) ──
    {"id": "D01", "name": "Kaffe, filterkaffe",         "coicop": "CP01211", "unit": "kg",  "weight": 2.5, "search": "filterkaffe"},
    {"id": "D02", "name": "Appelsinjuice",              "coicop": "CP01221", "unit": "L",   "weight": 2.0, "search": "appelsinjuice"},
    {"id": "D03", "name": "Coca-Cola / Pepsi",          "coicop": "CP01222", "unit": "L",   "weight": 2.0, "search": "coca cola 1.5"},
    {"id": "D04", "name": "Mineralvand / danskvand",    "coicop": "CP01222", "unit": "L",   "weight": 1.5, "search": "danskvand mineralvand"},
    {"id": "D05", "name": "Øl, pilsner 6-pack",        "coicop": "CP02121", "unit": "L",   "weight": 2.0, "search": "pilsner øl 6 pack"},
]

# Normaliser vægte til sum = 1
TOTAL_WEIGHT = sum(item["weight"] for item in BASKET)
for item in BASKET:
    item["norm_weight"] = item["weight"] / TOTAL_WEIGHT


# ── Dataklasser ──────────────────────────────────────────────────────────────

@dataclass
class PriceObservation:
    """En enkelt prisobservation for et produkt."""
    basket_id: str       # Reference til BASKET item
    date: str            # YYYY-MM-DD
    source: str          # 'nemlig', 'demo', 'manual'
    chain: str           # 'nemlig.com', 'Demo-butik', etc.
    product_name: str    # Faktisk produktnavn fra kilden
    price: float         # Observeret pris i DKK
    quantity: float      # Mængde (fx 0.5 for 500g)
    unit: str            # 'kg', 'L', 'stk'
    unit_price: float    # Beregnet enhedspris (kr/kg, kr/L, kr/stk)
    url: Optional[str] = None
    ean: Optional[str] = None
    on_sale: bool = False
    confidence: float = 1.0


@dataclass
class DailyIndex:
    """Dagligt prisindeks for hele kurven."""
    date: str
    composite_index: float    # Vægtet sammensat indeks (base=100)
    n_observations: int       # Antal observationer
    coverage_pct: float       # Dækningsgrad (%)
    category_indices: dict    # Indeks per COICOP-kategori
    chain_indices: dict       # Indeks per kæde
    inflation_7d: Optional[float] = None
    inflation_30d: Optional[float] = None
    inflation_yoy: Optional[float] = None


# ── Hjælpefunktion: parse mængde og enhed ───────────────────────────────────

def parse_quantity(text: str) -> tuple[float, str]:
    """
    Udtræk mængde og enhed fra produktnavn/tekst.

    Eksempler:
      "Hakket oksekød 500g"  → (0.5,  "kg")
      "Letmælk 1L"           → (1.0,  "L")
      "Æg 10 stk"            → (10.0, "stk")
    """
    text = text.lower()

    # Gram → kg
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:gr|gram|g)\b", text)
    if m:
        return float(m.group(1).replace(",", ".")) / 1000, "kg"

    # Kilogram
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*kg\b", text)
    if m:
        return float(m.group(1).replace(",", ".")), "kg"

    # Milliliter → L
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:ml|milliliter)\b", text)
    if m:
        return float(m.group(1).replace(",", ".")) / 1000, "L"

    # Centiliter → L
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*cl\b", text)
    if m:
        return float(m.group(1).replace(",", ".")) / 100, "L"

    # Liter
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:l|liter)\b", text)
    if m:
        return float(m.group(1).replace(",", ".")), "L"

    # Styk / pakke
    m = re.search(r"(\d+)\s*(?:stk|styk|pk|pack)\b", text)
    if m:
        return float(m.group(1)), "stk"

    return 1.0, "stk"


# ── Scraper: nemlig.com (HTML – ingen API-nøgle) ─────────────────────────────

class NemligScraper:
    """
    Scraper til nemlig.com priser via HTML-parsing og JSON-LD.

    nemlig.com er en ren online dagligvarebutik, så alle listepriser er
    synlige direkte i siden.  Scriptet bruger ingen privat API – kun
    det samme HTML som en browser henter.

    VIGTIGT: Respekter robots.txt og brug de inbyggede delays.
    Overvej at kontakte nemlig.com for officiel API-adgang ved
    kommerciel brug med højt volumen.
    """

    BASE_URL = "https://www.nemlig.com"

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "da-DK,da;q=0.9,en;q=0.8",
        "Referer": "https://www.nemlig.com/",
    }

    DELAY_SECONDS = 2.5  # Venlig forsinkelse mellem requests

    def __init__(self):
        try:
            import requests
            self.session = requests.Session()
            self.session.headers.update(self.HEADERS)
            self._requests_ok = True
        except ImportError:
            log.warning("requests ikke installeret – kør: pip install requests beautifulsoup4 lxml")
            self._requests_ok = False

    def search_product(self, query: str, limit: int = 5) -> list[dict]:
        """
        Søg efter et produkt på nemlig.com.
        Returnerer en liste af dicts med name, price, quantity, unit, unit_price.
        """
        if not self._requests_ok:
            return []

        try:
            url = f"{self.BASE_URL}/search?query={query}"
            resp = self.session.get(url, timeout=20)
            resp.raise_for_status()
        except Exception as e:
            log.warning(f"Netværksfejl ved søgning '{query}': {e}")
            return []

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            log.warning("beautifulsoup4 ikke installeret – pip install beautifulsoup4 lxml")
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        products = []

        # ── Forsøg 1: produktkort-elementer ────────────────────────────────
        selectors = [
            "[data-testid='product-card']",
            ".product-card",
            ".product-item",
            "article[class*='product']",
        ]
        cards = []
        for sel in selectors:
            cards = soup.select(sel)
            if cards:
                break

        for card in cards[:limit]:
            parsed = self._parse_card(card)
            if parsed:
                products.append(parsed)

        # ── Forsøg 2: JSON-LD structured data ──────────────────────────────
        if not products:
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    data = json.loads(script.string or "")
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        if isinstance(item, dict) and item.get("@type") == "Product":
                            parsed = self._parse_jsonld(item)
                            if parsed:
                                products.append(parsed)
                        if len(products) >= limit:
                            break
                except (json.JSONDecodeError, AttributeError):
                    continue
                if len(products) >= limit:
                    break

        time.sleep(self.DELAY_SECONDS)
        return products

    def _parse_card(self, card) -> Optional[dict]:
        """Parse et produktkort fra HTML."""
        try:
            name_el = card.select_one(
                "[data-testid='product-name'], .product-name, "
                "[class*='productName'], h2, h3"
            )
            price_el = card.select_one(
                "[data-testid='product-price'], .product-price, "
                "[class*='price'], .Price"
            )
            if not name_el or not price_el:
                return None

            name = name_el.get_text(strip=True)
            price_text = price_el.get_text(strip=True)

            # "29,95 kr" eller "29.95"
            m = re.search(r"(\d+)[,.](\d{2})", price_text)
            if m:
                price = float(f"{m.group(1)}.{m.group(2)}")
            else:
                m = re.search(r"(\d+)", price_text)
                if not m:
                    return None
                price = float(m.group(1))

            # Enhed fra separat element eller produktnavn
            unit_el = card.select_one(
                "[data-testid='product-unit-price'], .unit-price, "
                "[class*='unitPrice'], .product-weight"
            )
            extra = unit_el.get_text(strip=True) if unit_el else ""
            quantity, unit = parse_quantity(name + " " + extra)
            unit_price = price / quantity if quantity > 0 else price

            return {
                "name": name,
                "price": price,
                "quantity": quantity,
                "unit": unit,
                "unit_price": unit_price,
                "source": "nemlig",
                "chain": "nemlig.com",
            }
        except Exception:
            return None

    def _parse_jsonld(self, data: dict) -> Optional[dict]:
        """Parse JSON-LD Product schema."""
        try:
            name = data.get("name", "")
            offers = data.get("offers", {})
            if isinstance(offers, list):
                offers = offers[0] if offers else {}
            price = float(offers.get("price", 0))
            if price == 0:
                return None

            quantity, unit = parse_quantity(name)
            return {
                "name": name,
                "price": price,
                "quantity": quantity,
                "unit": unit,
                "unit_price": price / quantity if quantity > 0 else price,
                "source": "nemlig",
                "chain": "nemlig.com",
                "ean": data.get("gtin13") or data.get("gtin"),
            }
        except Exception:
            return None


# ── Produktmatching ──────────────────────────────────────────────────────────

class ProductMatcher:
    """
    Matcher scrapede produkter til varekurven via keyword-overlap.

    Installér fuzzywuzzy for bedre matching:
      pip install fuzzywuzzy python-Levenshtein
    """

    def __init__(self):
        try:
            from fuzzywuzzy import fuzz
            self._fuzz = fuzz
            self._has_fuzzy = True
        except ImportError:
            self._fuzz = None
            self._has_fuzzy = False

    def score(self, product_name: str, basket_item: dict) -> float:
        """Returnerer en konfidens-score (0-1)."""
        prod_lower = product_name.lower()
        terms = basket_item["search"].lower().split()
        keyword_score = sum(1 for t in terms if t in prod_lower) / max(len(terms), 1)

        if self._has_fuzzy:
            fuzzy_score = self._fuzz.token_set_ratio(
                basket_item["search"].lower(), prod_lower
            ) / 100.0
            return round(0.4 * keyword_score + 0.6 * fuzzy_score, 3)
        return round(keyword_score, 3)

    def best_match(
        self, products: list[dict], basket_item: dict, min_score: float = 0.4
    ) -> Optional[dict]:
        """Returnerer det bedste match eller None."""
        best, best_score = None, 0.0
        for p in products:
            s = self.score(p["name"], basket_item)
            if s > best_score and s >= min_score:
                best_score, best = s, {**p, "confidence": s}
        return best


# ── Eurostat HICP data ───────────────────────────────────────────────────────

class EurostatFetcher:
    """
    Henter HICP (Harmonised Index of Consumer Prices) fra Eurostat.
    Bruges som benchmark for vores scrape-baserede indeks.
    Ingen API-nøgle krævet.
    """

    API_BASE = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1"

    FOOD_COICOPS = {
        "CP0111": "Brød og kornprodukter",
        "CP0112": "Kød",
        "CP0113": "Fisk og skaldyr",
        "CP0114": "Mælk, ost og æg",
        "CP0115": "Olier og fedtstoffer",
        "CP0116": "Frugt",
        "CP0117": "Grøntsager",
        "CP0118": "Sukker, syltetøj, chokolade",
        "CP0119": "Andre fødevarer",
        "CP011":  "Fødevarer (samlet)",
        "CP012":  "Ikke-alkoholiske drikkevarer",
    }

    def fetch_hicp_denmark(self, start_year: int = 2020) -> list[dict]:
        """
        Hent månedlig HICP for Danmark for alle fødevarekategorier.
        Returnerer liste af dicts: date, coicop, coicop_name, index, country.
        """
        try:
            import requests
        except ImportError:
            log.error("requests mangler – pip install requests")
            return []

        results = []
        coicops = list(self.FOOD_COICOPS.keys())

        for coicop in coicops:
            url = (
                f"{self.API_BASE}/data/prc_hicp_midx/"
                f"M.I15.{coicop}.DK"
                f"?format=JSON&lang=en"
                f"&startPeriod={start_year}-01"
            )
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code != 200:
                    log.warning(f"Eurostat HTTP {resp.status_code} for {coicop}")
                    continue

                data = resp.json()
                time_index = (
                    data.get("dimension", {})
                    .get("time", {})
                    .get("category", {})
                    .get("index", {})
                )
                values = data.get("value", {})

                for period, idx in time_index.items():
                    val = values.get(str(idx))
                    if val is not None:
                        results.append({
                            "date": period,
                            "coicop": coicop,
                            "coicop_name": self.FOOD_COICOPS.get(coicop, coicop),
                            "index": float(val),
                            "country": "DK",
                        })

                time.sleep(0.5)

            except Exception as e:
                log.warning(f"Eurostat fejl for {coicop}: {e}")
                continue

        log.info(f"Hentet {len(results)} Eurostat HICP-observationer for Danmark")
        return results

    def save(self, data: list[dict]):
        if not data:
            return
        filepath = EUROSTAT_DIR / f"hicp_dk_{datetime.now().strftime('%Y%m%d')}.csv"
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
        log.info(f"Eurostat-data gemt: {filepath}")


# ── Indeksberegning ──────────────────────────────────────────────────────────

class IndexCalculator:
    """
    Beregner Laspeyres prisindeks med faste COICOP-vægte.

      Indeks = Σ (w_i × p_it / p_i0) × 100

    Første gang en vare observeres, sættes den som basispris (= 100).
    """

    def __init__(self, basket: list[dict]):
        self.basket = basket
        self.base_prices: dict[str, float] = {}

    def load_base_prices(self):
        base_file = DATA_DIR / "base_prices.json"
        if base_file.exists():
            with open(base_file, "r", encoding="utf-8") as f:
                self.base_prices = json.load(f)
            log.info(f"Basispriser indlæst for {len(self.base_prices)} varer")

    def save_base_prices(self):
        base_file = DATA_DIR / "base_prices.json"
        with open(base_file, "w", encoding="utf-8") as f:
            json.dump(self.base_prices, f, indent=2, ensure_ascii=False)

    def calculate_index(self, observations: list[PriceObservation], date: str) -> DailyIndex:
        # Gennemsnitspris per basket_id (håndtér flere butikker/tilbud)
        prices_by_item: dict[str, list[float]] = {}
        for obs in observations:
            prices_by_item.setdefault(obs.basket_id, []).append(obs.unit_price)
        avg_prices = {bid: sum(ps) / len(ps) for bid, ps in prices_by_item.items()}

        # Registrér nye basispriser
        for bid, price in avg_prices.items():
            if bid not in self.base_prices:
                self.base_prices[bid] = price
                log.info(f"Ny basispris: {bid} = {price:.2f} DKK")

        # Vægtet Laspeyres-sum
        weighted_sum = 0.0
        weight_used = 0.0
        cat_sums: dict[str, list[float]] = {}

        for item in self.basket:
            bid = item["id"]
            if bid not in avg_prices or bid not in self.base_prices:
                continue
            ratio = avg_prices[bid] / self.base_prices[bid]
            w = item["norm_weight"]
            weighted_sum += w * ratio
            weight_used += w

            coicop = item["coicop"][:6]
            cat_sums.setdefault(coicop, []).append((w, ratio))

        composite = (weighted_sum / weight_used * 100) if weight_used > 0 else 100.0

        # Kategori-indekser
        cat_indices = {}
        for coicop, wr_list in cat_sums.items():
            total_w = sum(w for w, _ in wr_list)
            cat_indices[coicop] = round(
                sum(w * r for w, r in wr_list) / total_w * 100, 2
            ) if total_w > 0 else 100.0

        coverage = len(avg_prices) / len(self.basket) * 100

        return DailyIndex(
            date=date,
            composite_index=round(composite, 4),
            n_observations=len(observations),
            coverage_pct=round(coverage, 1),
            category_indices=cat_indices,
            chain_indices={},
        )

    def add_inflation_rates(self, current: DailyIndex, history: list[DailyIndex]) -> DailyIndex:
        if not history:
            return current

        cur_dt = datetime.strptime(current.date, "%Y-%m-%d")

        def nearest(days: int) -> Optional[DailyIndex]:
            target = cur_dt - timedelta(days=days)
            candidates = [
                h for h in history
                if abs((datetime.strptime(h.date, "%Y-%m-%d") - target).days) <= 3
            ]
            return min(candidates, key=lambda h: abs(
                (datetime.strptime(h.date, "%Y-%m-%d") - target).days
            )) if candidates else None

        def annualised(h: DailyIndex, ref_days: int) -> Optional[float]:
            d = (cur_dt - datetime.strptime(h.date, "%Y-%m-%d")).days
            if d <= 0:
                return None
            return round(
                ((current.composite_index / h.composite_index) ** (365 / d) - 1) * 100, 2
            )

        h7 = nearest(7)
        if h7:
            current.inflation_7d = annualised(h7, 7)

        h30 = nearest(30)
        if h30:
            current.inflation_30d = annualised(h30, 30)

        h365 = nearest(365)
        if h365:
            current.inflation_yoy = round(
                (current.composite_index / h365.composite_index - 1) * 100, 2
            )

        return current


# ── DataStore ────────────────────────────────────────────────────────────────

class DataStore:
    """Persistens: gem og indlæs prisobservationer og indekser."""

    @staticmethod
    def save_observations(obs: list[PriceObservation], date: str):
        filepath = PRICES_DIR / f"prices_{date}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([asdict(o) for o in obs], f, indent=2, ensure_ascii=False)
        log.info(f"Gemt {len(obs)} observationer: {filepath}")

    @staticmethod
    def load_observations(date: str) -> list[PriceObservation]:
        filepath = PRICES_DIR / f"prices_{date}.json"
        if not filepath.exists():
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            return [PriceObservation(**d) for d in json.load(f)]

    @staticmethod
    def save_index(idx: DailyIndex):
        filepath = INDEX_DIR / f"index_{idx.date}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(idx), f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_all_indices() -> list[DailyIndex]:
        indices = []
        for fp in sorted(INDEX_DIR.glob("index_*.json")):
            with open(fp, "r", encoding="utf-8") as f:
                indices.append(DailyIndex(**json.load(f)))
        return indices

    @staticmethod
    def export_csv():
        indices = DataStore.load_all_indices()
        if not indices:
            return
        filepath = DATA_DIR / "price_index_history.csv"
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "date", "composite_index", "n_observations", "coverage_pct",
                "inflation_7d", "inflation_30d", "inflation_yoy",
            ])
            for idx in indices:
                w.writerow([
                    idx.date, idx.composite_index, idx.n_observations,
                    idx.coverage_pct, idx.inflation_7d, idx.inflation_30d,
                    idx.inflation_yoy,
                ])
        log.info(f"Indeks-historik eksporteret: {filepath}")


# ── Pipeline ─────────────────────────────────────────────────────────────────

class PriceMonitorPipeline:
    """Orkestrerer scraping, indeksberegning og rapportering."""

    def __init__(self):
        self.nemlig = NemligScraper()
        self.matcher = ProductMatcher()
        self.calculator = IndexCalculator(BASKET)
        self.store = DataStore()

    # ── Scraping ─────────────────────────────────────────────────────────────

    def run_scrape(self, date: str = None):
        """
        Hent listepriser fra nemlig.com for alle 50 kurv-varer.
        Ingen API-nøgle krævet – kun offentligt tilgængeligt HTML.
        """
        date = date or datetime.now().strftime("%Y-%m-%d")
        log.info(f"═══ Starter scraping (nemlig.com) for {date} ═══")

        observations = []

        for i, item in enumerate(BASKET):
            log.info(f"[{i+1}/{len(BASKET)}] {item['name']} – søger '{item['search']}'")

            results = self.nemlig.search_product(item["search"])
            if not results:
                log.info(f"  – ingen resultater for '{item['search']}'")
                continue

            best = self.matcher.best_match(results, item)
            if not best:
                log.info(f"  – intet match over tærsklen for '{item['name']}'")
                continue

            obs = PriceObservation(
                basket_id=item["id"],
                date=date,
                source="nemlig",
                chain="nemlig.com",
                product_name=best["name"],
                price=best["price"],
                quantity=best["quantity"],
                unit=best["unit"],
                unit_price=best["unit_price"],
                ean=best.get("ean"),
                confidence=best["confidence"],
            )
            observations.append(obs)
            log.info(
                f"  ✓ {best['name']} → {best['unit_price']:.2f} kr/{item['unit']} "
                f"(konfidens: {best['confidence']:.2f})"
            )

        self.store.save_observations(observations, date)
        log.info(f"═══ Scraping færdig: {len(observations)}/{len(BASKET)} varer ═══")
        return observations

    # ── Indeks ───────────────────────────────────────────────────────────────

    def run_index(self, date: str = None):
        date = date or datetime.now().strftime("%Y-%m-%d")
        self.calculator.load_base_prices()

        obs = self.store.load_observations(date)
        if not obs:
            log.warning(f"Ingen observationer for {date} – kør 'scrape' eller 'demo' først")
            return None

        idx = self.calculator.calculate_index(obs, date)
        history = self.store.load_all_indices()
        idx = self.calculator.add_inflation_rates(idx, history)

        self.store.save_index(idx)
        self.calculator.save_base_prices()

        log.info(
            f"Indeks {date}: {idx.composite_index:.2f} | "
            f"Dækning: {idx.coverage_pct:.0f}% | "
            f"7D: {idx.inflation_7d or 'N/A'}% | "
            f"30D: {idx.inflation_30d or 'N/A'}%"
        )
        return idx

    # ── Rapport ──────────────────────────────────────────────────────────────

    def run_report(self):
        indices = self.store.load_all_indices()
        if not indices:
            log.warning("Ingen indeksdata – kør 'scrape'+'index' eller 'demo' først")
            return

        latest = indices[-1]
        coicop_names = {
            "CP0111": "Brød & korn",    "CP0112": "Kød",
            "CP0113": "Fisk",           "CP0114": "Mælk/ost/æg",
            "CP0115": "Olier/fedt",     "CP0116": "Frugt",
            "CP0117": "Grøntsager",     "CP0118": "Sukker/chok.",
            "CP0119": "Andre",          "CP0121": "Kaffe/te",
            "CP0122": "Drikkevarer",
        }

        print("\n" + "=" * 62)
        print("  DK FØDEVAREPRIS-MONITOR – RAPPORT")
        print("=" * 62)
        print(f"  Dato:                    {latest.date}")
        print(f"  Sammensat indeks:        {latest.composite_index:.2f}  (base = 100)")
        print(f"  Observationer:           {latest.n_observations}")
        print(f"  Dækning:                 {latest.coverage_pct:.0f}%")
        print(f"  7D inflation (ann.):     {latest.inflation_7d or 'N/A'}%")
        print(f"  30D inflation (ann.):    {latest.inflation_30d or 'N/A'}%")
        print(f"  YoY inflation:           {latest.inflation_yoy or 'N/A'}%")

        if latest.category_indices:
            print(f"\n  Kategori-indekser:")
            for coicop, val in sorted(latest.category_indices.items()):
                name = coicop_names.get(coicop, coicop)
                delta = val - 100
                arrow = "↑" if delta > 0.05 else ("↓" if delta < -0.05 else "→")
                print(f"    {name:22s} {val:7.2f}  {arrow} {delta:+.1f}%")

        print(f"\n  Datapunkter i historik:  {len(indices)}")
        if len(indices) >= 2:
            first = indices[0]
            total = (latest.composite_index / first.composite_index - 1) * 100
            days = (
                datetime.strptime(latest.date, "%Y-%m-%d")
                - datetime.strptime(first.date, "%Y-%m-%d")
            ).days
            print(f"  Periode:                 {first.date} → {latest.date} ({days} dage)")
            print(f"  Samlet prisændring:      {total:+.2f}%")

        print("=" * 62)
        self.store.export_csv()

    # ── Eurostat ─────────────────────────────────────────────────────────────

    def run_eurostat(self):
        fetcher = EurostatFetcher()
        data = fetcher.fetch_hicp_denmark()
        fetcher.save(data)

    # ── Demo (syntetiske data) ────────────────────────────────────────────────

    def run_demo(self):
        """
        Genererer realistiske syntetiske priser og kører hele pipeline.
        Nyttigt til test uden netadgang.
        """
        import random
        random.seed(42)

        date = datetime.now().strftime("%Y-%m-%d")
        log.info(f"═══ DEMO: syntetiske priser for {date} ═══")

        # Repræsentative basispriser (DKK pr. enhed)
        base = {
            "B01": 18.0, "B02": 15.0, "B03": 12.0, "B04": 14.0, "B05": 22.0,
            "B06": 10.0, "B07":  4.0, "B08": 35.0,
            "K01": 55.0, "K02": 65.0, "K03": 45.0, "K04": 80.0, "K05": 40.0,
            "K06": 90.0, "K07": 35.0,
            "F01": 120.0, "F02": 60.0, "F03": 95.0,
            "M01": 10.0, "M02": 22.0, "M03": 65.0, "M04": 35.0, "M05": 12.0,
            "M06": 18.0, "M07": 20.0,
            "O01": 25.0, "O02": 60.0,
            "FR01": 14.0, "FR02": 20.0, "FR03": 18.0, "FR04": 45.0,
            "G01": 10.0, "G02": 25.0, "G03":  8.0, "G04": 10.0, "G05": 12.0,
            "G06": 15.0, "G07": 30.0,
            "S01": 12.0, "S02": 70.0, "S03": 40.0, "S04": 35.0,
            "A01": 12.0, "A02": 25.0, "A03":  8.0, "A04": 20.0,
            "D01": 50.0, "D02": 18.0, "D03": 12.0, "D04":  5.0, "D05": 45.0,
        }

        observations = []
        for item in BASKET:
            p = round(base.get(item["id"], 20.0) * (1 + random.uniform(-0.05, 0.05)), 2)
            observations.append(PriceObservation(
                basket_id=item["id"],
                date=date,
                source="demo",
                chain="Demo-butik",
                product_name=item["name"],
                price=p,
                quantity=1.0,
                unit=item["unit"],
                unit_price=p,
                confidence=1.0,
            ))

        self.store.save_observations(observations, date)
        log.info(f"Demo: {len(observations)} syntetiske observationer genereret")
        self.run_index(date)
        self.run_report()

    # ── Fuld pipeline ─────────────────────────────────────────────────────────

    def run_full(self):
        date = datetime.now().strftime("%Y-%m-%d")
        self.run_scrape(date)
        self.run_index(date)
        self.run_report()


# ── CLI ──────────────────────────────────────────────────────────────────────

USAGE = """
╔══════════════════════════════════════════════════════════════════╗
║  DK Fødevarepris-Monitor                                        ║
║  Prisovervågning for 50 tracer-varer i danske supermarkeder     ║
║  Kilde: nemlig.com (HTML-scraping – ingen API-nøgle krævet)    ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Kommandoer:                                                     ║
║    scrape    Hent listepriser fra nemlig.com                    ║
║    index     Beregn Laspeyres prisindeks                        ║
║    report    Vis rapport og eksportér CSV                       ║
║    eurostat  Hent Eurostat HICP benchmark (ingen nøgle)        ║
║    full      Kør hele pipeline: scrape → index → rapport        ║
║    demo      Kør med syntetiske data (test uden net)           ║
║    basket    Vis varekurven (50 varer)                         ║
║                                                                  ║
║  Data gemmes i: ./data/price_monitor/                           ║
╚══════════════════════════════════════════════════════════════════╝
"""


def print_basket():
    print(f"\n{'─' * 92}")
    print("  VAREKURV: 50 tracer-varer med COICOP-vægte")
    print(f"{'─' * 92}")
    print(f"  {'ID':<6} {'Vare':<37} {'COICOP':<10} {'Enhed':<5} {'Vægt':>6} {'Norm%':>7}")
    print(f"{'─' * 92}")

    prev_coicop = ""
    for item in BASKET:
        grp = item["coicop"][:6]
        if grp != prev_coicop:
            prev_coicop = grp
            print(f"  {'─' * 87}")
        print(
            f"  {item['id']:<6} {item['name']:<37} {item['coicop']:<10} "
            f"{item['unit']:<5} {item['weight']:>5.1f} {item['norm_weight']*100:>6.1f}%"
        )

    print(f"{'─' * 92}")
    print(f"  {'TOTAL':>55} {TOTAL_WEIGHT:>5.1f} {100.0:>6.1f}%")
    print(f"  Antal varer: {len(BASKET)}")
    print(f"{'─' * 92}\n")


def main():
    if len(sys.argv) < 2:
        print(USAGE)
        return

    cmd = sys.argv[1].lower().lstrip("-")
    pipe = PriceMonitorPipeline()

    dispatch = {
        "scrape":   pipe.run_scrape,
        "index":    pipe.run_index,
        "report":   pipe.run_report,
        "eurostat": pipe.run_eurostat,
        "full":     pipe.run_full,
        "demo":     pipe.run_demo,
        "basket":   print_basket,
    }

    if cmd in dispatch:
        dispatch[cmd]()
    else:
        print(f"Ukendt kommando: '{cmd}'")
        print(USAGE)


if __name__ == "__main__":
    main()
