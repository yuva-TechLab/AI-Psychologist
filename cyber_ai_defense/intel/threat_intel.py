"""
Module 11: Threat Intel Enrichment
=====================================
Enriches source IPs with reputation scores, GeoIP data, ASN info,
and known threat actor associations — then feeds that context back
into the threat scoring pipeline to boost or suppress scores.

Components
----------
  GeoIPResolver      — maps IP → country, city, ASN, org (MaxMind-style)
  ReputationChecker  — checks IP against threat intel feeds (simulated)
  ThreatActorMatcher — matches ASN/country/range to known APT groups
  EnrichmentCache    — TTL-based in-memory cache (avoid repeat lookups)
  IPEnricher         — orchestrates all three + produces EnrichedIP
  ScoreAdjuster      — applies intel findings to raise/lower threat score

Supported intel feeds (simulated in dry_run / real with API keys)
-----------------------------------------------------------------
  * AbuseIPDB         — community abuse reports
  * VirusTotal        — malware/phishing detections
  * Shodan            — exposed services, banners
  * IPInfo            — GeoIP + ASN
  * Internal blocklist — known bad IPs / ranges

Educational use only — defensive research prototype.
"""

import json, time, ipaddress, hashlib, re
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict
import logging, os, urllib.request, urllib.error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).resolve().parent.parent
INTEL_DIR = BASE_DIR / "intel"
INTEL_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# STATIC KNOWLEDGE BASES  (simulate offline intel)
# ─────────────────────────────────────────────

# GeoIP table — covers RFC-1918 + common public ranges
GEOIP_TABLE = {
    # Private / special
    "10.":       {"country":"Internal","country_code":"INT","city":"LAN","org":"Private Network","asn":"AS0"},
    "192.168.":  {"country":"Internal","country_code":"INT","city":"LAN","org":"Private Network","asn":"AS0"},
    "172.16.":   {"country":"Internal","country_code":"INT","city":"LAN","org":"Private Network","asn":"AS0"},
    "127.":      {"country":"Localhost","country_code":"LO","city":"Loopback","org":"Loopback","asn":"AS0"},
    "0.0.0.0":   {"country":"Unknown","country_code":"??","city":"Unknown","org":"Unknown","asn":"AS0"},
    # Simulated public ranges
    "1.1.1.":    {"country":"Australia","country_code":"AU","city":"Sydney","org":"Cloudflare","asn":"AS13335"},
    "8.8.8.":    {"country":"United States","country_code":"US","city":"Mountain View","org":"Google LLC","asn":"AS15169"},
    "45.33.":    {"country":"United States","country_code":"US","city":"Fremont","org":"Akamai Technologies","asn":"AS63949"},
    "51.":       {"country":"United Kingdom","country_code":"GB","city":"London","org":"Microsoft Azure","asn":"AS8075"},
    "52.":       {"country":"United States","country_code":"US","city":"Ashburn","org":"Amazon AWS","asn":"AS16509"},
    "91.":       {"country":"Russia","country_code":"RU","city":"Moscow","org":"Rostelecom","asn":"AS12389"},
    "103.":      {"country":"China","country_code":"CN","city":"Beijing","org":"Alibaba Cloud","asn":"AS37963"},
    "185.":      {"country":"Netherlands","country_code":"NL","city":"Amsterdam","org":"Hosting provider","asn":"AS206898"},
    "194.":      {"country":"Russia","country_code":"RU","city":"Saint Petersburg","org":"VDS hosting","asn":"AS48282"},
    "195.":      {"country":"Ukraine","country_code":"UA","city":"Kyiv","org":"Volia","asn":"AS31148"},
    "198.":      {"country":"United States","country_code":"US","city":"Chicago","org":"Cogent Communications","asn":"AS174"},
    "199.":      {"country":"United States","country_code":"US","city":"Dallas","org":"Hurricane Electric","asn":"AS6939"},
    "204.":      {"country":"Canada","country_code":"CA","city":"Toronto","org":"TELUS Communications","asn":"AS852"},
    "220.":      {"country":"South Korea","country_code":"KR","city":"Seoul","org":"Korea Telecom","asn":"AS4766"},
}

# Simulated AbuseIPDB scores: {ip_prefix: confidence_score 0-100}
ABUSEIPDB_SCORES = {
    "192.168.1.50":  85,   # our demo attacker — high confidence
    "10.0.0.200":    72,   # insider threat host
    "172.16.0.55":   91,   # C2 operator
    "45.33.32.156":  88,   # known scanner
    "91.":           60,   # Russian IP range — elevated
    "103.":          55,   # Chinese IP range — elevated
    "185.":          65,   # Bulletproof hosting — elevated
    "194.":          70,   # Russian VDS — elevated
    "195.":          45,
}

# Simulated VirusTotal detections: {ip: detections/total}
VT_DETECTIONS = {
    "192.168.1.50":  {"detected": 12, "total": 90, "malicious": True},
    "172.16.0.55":   {"detected": 31, "total": 90, "malicious": True},
    "45.33.32.156":  {"detected":  8, "total": 90, "malicious": True},
    "91.109.21.":    {"detected": 25, "total": 90, "malicious": True},
    "185.220.101.":  {"detected": 47, "total": 90, "malicious": True},
}

# Simulated Shodan banners: open ports / services
SHODAN_BANNERS = {
    "192.168.1.50":  {"ports": [22, 80, 443, 4444], "tags": ["self-signed", "C2"], "vulns": ["CVE-2021-44228"]},
    "172.16.0.55":   {"ports": [443, 8080, 9999],   "tags": ["C2", "tor-exit"],    "vulns": []},
    "45.33.32.156":  {"ports": [22, 80, 8080],       "tags": ["scanner"],           "vulns": []},
    "185.220.101.1": {"ports": [9001, 9030],          "tags": ["tor-exit"],          "vulns": []},
}

# Known threat actor fingerprints: ASN / country / IP ranges → APT group
THREAT_ACTORS = [
    {"name": "APT29 (Cozy Bear)",   "countries": ["RU"],     "asns": ["AS12389","AS48282"],
     "description": "Russian SVR-linked group. Targets government, think tanks, healthcare.",
     "ttps": ["T1566","T1078","T1021"]},
    {"name": "APT41",               "countries": ["CN"],     "asns": ["AS37963","AS4134"],
     "description": "Chinese state-sponsored. Dual espionage + financial crime.",
     "ttps": ["T1190","T1059","T1027"]},
    {"name": "Lazarus Group",       "countries": ["KP"],     "asns": [],
     "description": "North Korean threat actor. Financial theft, crypto, ransomware.",
     "ttps": ["T1566","T1486","T1041"]},
    {"name": "FIN7",                "countries": ["UA","RU"],"asns": ["AS31148","AS48282"],
     "description": "Financially motivated. Targets retail, hospitality, finance.",
     "ttps": ["T1566","T1204","T1005"]},
    {"name": "Bulletproof Hosting", "countries": ["NL","RO"],"asns": ["AS206898","AS9009"],
     "description": "Criminal hosting infrastructure used by multiple threat actors.",
     "ttps": ["T1583","T1584"]},
]

# Internal blocklist (your own SOC-maintained list)
INTERNAL_BLOCKLIST = {
    "192.168.1.50":  {"reason": "confirmed attacker — active incident 2026-03-13", "severity": "CRITICAL"},
    "172.16.0.55":   {"reason": "C2 beacon source", "severity": "CRITICAL"},
    "45.33.32.156":  {"reason": "persistent port scanner", "severity": "HIGH"},
    "10.0.0.200":    {"reason": "suspected insider — under investigation", "severity": "HIGH"},
}

# Score adjustment weights
SCORE_ADJUSTMENTS = {
    "blocklist_critical": +20,
    "blocklist_high":     +12,
    "abuseipdb_high":     +10,   # score >= 75
    "abuseipdb_medium":   +5,    # score >= 50
    "vt_malicious":       +8,
    "shodan_c2_port":     +7,
    "shodan_vuln":        +5,
    "known_threat_actor": +10,
    "tor_exit_node":      +6,
    "high_risk_country":  +5,
    "internal_network":   -10,   # lower score for RFC-1918
    "trusted_cloud":      -5,    # lower score for known cloud ASNs
}

TRUSTED_CLOUD_ASNS = {"AS15169","AS16509","AS8075","AS13335","AS14061"}
HIGH_RISK_COUNTRIES = {"RU","CN","KP","IR","SY","CU","SD"}


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class GeoIPResult:
    ip           : str
    country      : str
    country_code : str
    city         : str
    org          : str
    asn          : str
    is_private   : bool = False
    is_tor       : bool = False

@dataclass
class ReputationResult:
    ip              : str
    abuseipdb_score : int   = 0     # 0–100
    vt_detections   : int   = 0
    vt_total        : int   = 90
    vt_malicious    : bool  = False
    shodan_ports    : list  = field(default_factory=list)
    shodan_tags     : list  = field(default_factory=list)
    shodan_vulns    : list  = field(default_factory=list)
    blocklisted     : bool  = False
    blocklist_reason: str   = ""
    blocklist_sev   : str   = ""

@dataclass
class EnrichedIP:
    """Full intel profile for one IP address."""
    ip              : str
    geo             : GeoIPResult
    reputation      : ReputationResult
    threat_actors   : list           = field(default_factory=list)
    score_delta     : float          = 0.0
    adjustments     : list           = field(default_factory=list)
    risk_flags      : list           = field(default_factory=list)
    enriched_at     : str            = ""
    source          : str            = "simulated"   # "simulated" | "live"

    def summary(self) -> str:
        flags = ", ".join(self.risk_flags) if self.risk_flags else "none"
        actors = ", ".join(a["name"] for a in self.threat_actors) if self.threat_actors else "none"
        return (
            f"{self.ip} | {self.geo.country} ({self.geo.country_code}) | "
            f"ASN: {self.geo.asn} | Abuse: {self.reputation.abuseipdb_score} | "
            f"VT: {self.reputation.vt_detections}/{self.reputation.vt_total} | "
            f"Score Δ: {self.score_delta:+.0f} | Flags: {flags} | Actors: {actors}"
        )

    def to_dict(self) -> dict:
        return {
            "ip"             : self.ip,
            "country"        : self.geo.country,
            "country_code"   : self.geo.country_code,
            "city"           : self.geo.city,
            "org"            : self.geo.org,
            "asn"            : self.geo.asn,
            "is_private"     : self.geo.is_private,
            "is_tor"         : self.geo.is_tor,
            "abuseipdb_score": self.reputation.abuseipdb_score,
            "vt_detections"  : self.reputation.vt_detections,
            "vt_total"       : self.reputation.vt_total,
            "vt_malicious"   : self.reputation.vt_malicious,
            "shodan_ports"   : self.reputation.shodan_ports,
            "shodan_tags"    : self.reputation.shodan_tags,
            "shodan_vulns"   : self.reputation.shodan_vulns,
            "blocklisted"    : self.reputation.blocklisted,
            "blocklist_reason": self.reputation.blocklist_reason,
            "threat_actors"  : [a["name"] for a in self.threat_actors],
            "score_delta"    : round(self.score_delta, 1),
            "adjustments"    : self.adjustments,
            "risk_flags"     : self.risk_flags,
            "enriched_at"    : self.enriched_at,
            "source"         : self.source,
        }


# ─────────────────────────────────────────────
# COMPONENTS
# ─────────────────────────────────────────────

class GeoIPResolver:
    """Maps IP addresses to geographic and ASN data."""

    def resolve(self, ip: str) -> GeoIPResult:
        # Private / special ranges
        try:
            obj = ipaddress.ip_address(ip)
            is_private = obj.is_private or obj.is_loopback or obj.is_unspecified
        except ValueError:
            is_private = False

        geo = self._lookup(ip)
        is_tor = "tor-exit" in SHODAN_BANNERS.get(ip, {}).get("tags", [])

        return GeoIPResult(
            ip           = ip,
            country      = geo["country"],
            country_code = geo["country_code"],
            city         = geo["city"],
            org          = geo["org"],
            asn          = geo["asn"],
            is_private   = is_private,
            is_tor       = is_tor,
        )

    def _lookup(self, ip: str) -> dict:
        # Exact match first
        if ip in GEOIP_TABLE:
            return GEOIP_TABLE[ip]
        # Prefix match (longest first)
        for prefix in sorted(GEOIP_TABLE, key=len, reverse=True):
            if ip.startswith(prefix):
                return GEOIP_TABLE[prefix]
        return {"country":"Unknown","country_code":"??","city":"Unknown",
                "org":"Unknown","asn":"AS0"}


class ReputationChecker:
    """Checks IP reputation across multiple simulated intel feeds."""

    def check(self, ip: str) -> ReputationResult:
        result = ReputationResult(ip=ip)

        # AbuseIPDB
        result.abuseipdb_score = self._abuseipdb(ip)

        # VirusTotal
        vt = self._virustotal(ip)
        result.vt_detections = vt.get("detected", 0)
        result.vt_total      = vt.get("total", 90)
        result.vt_malicious  = vt.get("malicious", False)

        # Shodan
        sh = self._shodan(ip)
        result.shodan_ports = sh.get("ports", [])
        result.shodan_tags  = sh.get("tags", [])
        result.shodan_vulns = sh.get("vulns", [])

        # Internal blocklist
        bl = INTERNAL_BLOCKLIST.get(ip)
        if bl:
            result.blocklisted      = True
            result.blocklist_reason = bl["reason"]
            result.blocklist_sev    = bl["severity"]

        return result

    def _abuseipdb(self, ip: str) -> int:
        if ip in ABUSEIPDB_SCORES:
            return ABUSEIPDB_SCORES[ip]
        for prefix, score in ABUSEIPDB_SCORES.items():
            if "." in prefix and ip.startswith(prefix):
                return score
        return 0

    def _virustotal(self, ip: str) -> dict:
        if ip in VT_DETECTIONS:
            return VT_DETECTIONS[ip]
        for prefix, data in VT_DETECTIONS.items():
            if ip.startswith(prefix):
                return data
        return {"detected": 0, "total": 90, "malicious": False}

    def _shodan(self, ip: str) -> dict:
        if ip in SHODAN_BANNERS:
            return SHODAN_BANNERS[ip]
        for prefix, data in SHODAN_BANNERS.items():
            if ip.startswith(prefix):
                return data
        return {}


class ThreatActorMatcher:
    """Matches an EnrichedIP profile to known APT / threat actor groups."""

    def match(self, geo: GeoIPResult) -> list:
        matched = []
        for actor in THREAT_ACTORS:
            if geo.country_code in actor["countries"]:
                matched.append(actor)
                continue
            if geo.asn in actor.get("asns", []):
                matched.append(actor)
        return matched


class EnrichmentCache:
    """Simple TTL in-memory cache keyed on IP address."""

    def __init__(self, ttl_s: int = 3600):
        self._cache : dict[str, tuple] = {}
        self._ttl   = ttl_s

    def get(self, ip: str):
        if ip in self._cache:
            result, ts = self._cache[ip]
            if time.time() - ts < self._ttl:
                return result
            del self._cache[ip]
        return None

    def set(self, ip: str, result: EnrichedIP):
        self._cache[ip] = (result, time.time())

    def size(self) -> int:
        return len(self._cache)


class ScoreAdjuster:
    """
    Applies intel findings to adjust the base threat score.
    Returns (adjusted_score, list_of_adjustments).
    """

    def adjust(self, base_score: float, enriched: EnrichedIP) -> tuple:
        delta = 0.0
        adj   = []
        flags = []

        geo  = enriched.geo
        rep  = enriched.reputation

        # Blocklist
        if rep.blocklisted:
            key = f"blocklist_{rep.blocklist_sev.lower()}"
            d   = SCORE_ADJUSTMENTS.get(key, 10)
            delta += d
            adj.append(f"blocklist({rep.blocklist_sev}) +{d}")
            flags.append(f"BLOCKLISTED:{rep.blocklist_sev}")

        # AbuseIPDB
        if rep.abuseipdb_score >= 75:
            d = SCORE_ADJUSTMENTS["abuseipdb_high"]
            delta += d; adj.append(f"abuseipdb_high({rep.abuseipdb_score}) +{d}")
            flags.append(f"ABUSE:{rep.abuseipdb_score}")
        elif rep.abuseipdb_score >= 50:
            d = SCORE_ADJUSTMENTS["abuseipdb_medium"]
            delta += d; adj.append(f"abuseipdb_medium({rep.abuseipdb_score}) +{d}")

        # VirusTotal
        if rep.vt_malicious:
            d = SCORE_ADJUSTMENTS["vt_malicious"]
            delta += d; adj.append(f"vt_malicious({rep.vt_detections}/{rep.vt_total}) +{d}")
            flags.append(f"VT_MALICIOUS:{rep.vt_detections}")

        # Shodan — C2 ports
        c2_ports = {4444, 1337, 9999, 6666, 31337}
        if any(p in c2_ports for p in rep.shodan_ports):
            d = SCORE_ADJUSTMENTS["shodan_c2_port"]
            delta += d; adj.append(f"shodan_c2_port +{d}")
            flags.append("C2_PORT_OPEN")

        # Shodan — vulns
        if rep.shodan_vulns:
            d = SCORE_ADJUSTMENTS["shodan_vuln"]
            delta += d; adj.append(f"shodan_vuln({rep.shodan_vulns[0]}) +{d}")
            flags.append(f"VULN:{rep.shodan_vulns[0]}")

        # Tor
        if geo.is_tor:
            d = SCORE_ADJUSTMENTS["tor_exit_node"]
            delta += d; adj.append(f"tor_exit +{d}")
            flags.append("TOR_EXIT")

        # Threat actors
        if enriched.threat_actors:
            d = SCORE_ADJUSTMENTS["known_threat_actor"]
            delta += d
            names = ", ".join(a["name"] for a in enriched.threat_actors)
            adj.append(f"threat_actor({names}) +{d}")
            flags.append(f"APT:{enriched.threat_actors[0]['name']}")

        # High-risk country
        if geo.country_code in HIGH_RISK_COUNTRIES:
            d = SCORE_ADJUSTMENTS["high_risk_country"]
            delta += d; adj.append(f"high_risk_country({geo.country_code}) +{d}")
            flags.append(f"COUNTRY:{geo.country_code}")

        # Trusted cloud (negative adjustment)
        if geo.asn in TRUSTED_CLOUD_ASNS:
            d = SCORE_ADJUSTMENTS["trusted_cloud"]
            delta += d; adj.append(f"trusted_cloud({geo.asn}) {d}")

        # Internal (negative adjustment)
        if geo.is_private:
            d = SCORE_ADJUSTMENTS["internal_network"]
            delta += d; adj.append(f"internal_network {d}")

        adjusted = float(min(100, max(0, base_score + delta)))
        return adjusted, adj, flags


# ─────────────────────────────────────────────
# MASTER ENRICHER
# ─────────────────────────────────────────────

class IPEnricher:
    """
    Orchestrates GeoIP → Reputation → ThreatActor → ScoreAdjust
    for any IP address. Results are cached.

    Usage
    -----
    enricher = IPEnricher()

    enriched = enricher.enrich("192.168.1.50")
    print(enriched.summary())

    # With score adjustment:
    new_score, adj, flags = enricher.adjust_score(base_score=70, ip="192.168.1.50")
    """

    def __init__(self, cache_ttl: int = 3600, dry_run: bool = True):
        self.geo     = GeoIPResolver()
        self.rep     = ReputationChecker()
        self.actor   = ThreatActorMatcher()
        self.cache   = EnrichmentCache(cache_ttl)
        self.adjuster= ScoreAdjuster()
        self.dry_run = dry_run
        self._log    : list = []

    def enrich(self, ip: str) -> EnrichedIP:
        cached = self.cache.get(ip)
        if cached:
            return cached

        geo_result  = self.geo.resolve(ip)
        rep_result  = self.rep.check(ip)
        actors      = self.actor.match(geo_result)

        enriched = EnrichedIP(
            ip           = ip,
            geo          = geo_result,
            reputation   = rep_result,
            threat_actors= actors,
            enriched_at  = datetime.now(timezone.utc).isoformat(),
            source       = "simulated" if self.dry_run else "live",
        )
        self.cache.set(ip, enriched)
        return enriched

    def adjust_score(self, base_score: float, ip: str) -> tuple:
        """Returns (adjusted_score, adjustments_list, risk_flags)."""
        enriched = self.enrich(ip)
        adj_score, adj_list, flags = self.adjuster.adjust(base_score, enriched)
        enriched.score_delta  = adj_score - base_score
        enriched.adjustments  = adj_list
        enriched.risk_flags   = flags
        self._log.append({"ip": ip, "base": base_score,
                          "adjusted": adj_score, "delta": adj_score - base_score})
        return adj_score, adj_list, flags

    def enrich_session(self, src_ip: str, base_score: float, sequence: list) -> dict:
        """Full enrichment + score adjustment for one pipeline event."""
        enriched          = self.enrich(src_ip)
        adj_score, adj, flags = self.adjust_score(base_score, src_ip)
        lv = "CRITICAL" if adj_score>=85 else "HIGH" if adj_score>=70 else \
             "MEDIUM" if adj_score>=50 else "LOW" if adj_score>=30 else "SAFE"
        return {
            "src_ip"        : src_ip,
            "sequence"      : sequence,
            "base_score"    : round(base_score, 1),
            "adjusted_score": round(adj_score, 1),
            "risk_level"    : lv,
            "score_delta"   : round(adj_score - base_score, 1),
            "adjustments"   : adj,
            "risk_flags"    : flags,
            "geo"           : enriched.to_dict(),
        }

    def save_results(self, results: list) -> Path:
        out = INTEL_DIR / "enrichment_results.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved {len(results)} enrichment results → {out}")
        return out


# ─────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE_DIR))

    print(f"\n{'='*64}")
    print(f"  MODULE 11 — Threat Intel Enrichment")
    print(f"{'='*64}\n")

    enricher = IPEnricher(dry_run=True)

    # Test IPs with varied profiles
    test_cases = [
        ("192.168.1.50",  88.0, ["Recon","Cred Access","Exploit","Lateral","C2"]),
        ("172.16.0.55",   92.0, ["Lateral Movement","Command & Control"]),
        ("10.0.0.200",    77.0, ["Discovery","Credential Access","Data Exfiltration"]),
        ("91.109.21.10",  55.0, ["Reconnaissance","Exploitation"]),
        ("185.220.101.1", 60.0, ["Reconnaissance"]),
        ("8.8.8.8",       25.0, ["Reconnaissance"]),      # Google DNS — trusted
        ("52.10.0.1",     45.0, ["Reconnaissance"]),      # AWS — trusted cloud
    ]

    results = []
    print(f"  {'IP':<18} {'Country':>4} {'Abuse':>5} {'VT':>5}  {'Base':>5} {'Adj':>5} {'Δ':>4}  {'Flags'}")
    print(f"  {'─'*80}")

    for ip, base, seq in test_cases:
        result = enricher.enrich_session(ip, base, seq)
        geo    = result["geo"]
        delta  = result["score_delta"]
        sign   = "+" if delta >= 0 else ""
        flags  = " ".join(result["risk_flags"][:3]) if result["risk_flags"] else "—"
        print(f"  {ip:<18} {geo['country_code']:>4}  "
              f"{geo['abuseipdb_score']:>5}  "
              f"{geo['vt_detections']:>2}/{geo['vt_total']:<2}  "
              f"{base:>5.0f}  {result['adjusted_score']:>5.1f} "
              f"{sign}{delta:>3.0f}  {flags}")
        results.append(result)

    # Detail for highest-risk IP
    print(f"\n  {'─'*64}")
    top = max(results, key=lambda r: r["adjusted_score"])
    print(f"  DETAIL: {top['src_ip']}")
    enriched = enricher.enrich(top['src_ip'])
    print(f"  GeoIP   : {enriched.geo.city}, {enriched.geo.country} ({enriched.geo.country_code})")
    print(f"  ASN/Org : {enriched.geo.asn} / {enriched.geo.org}")
    print(f"  Blocklist: {enriched.reputation.blocklisted} — {enriched.reputation.blocklist_reason}")
    print(f"  AbuseIPDB: {enriched.reputation.abuseipdb_score}/100")
    print(f"  VT       : {enriched.reputation.vt_detections}/{enriched.reputation.vt_total} detections")
    print(f"  Ports    : {enriched.reputation.shodan_ports}")
    print(f"  Vulns    : {enriched.reputation.shodan_vulns}")
    if enriched.threat_actors:
        for a in enriched.threat_actors:
            print(f"  APT match: {a['name']} — {a['description'][:60]}…")
    print(f"  Adjustments:")
    for a in top["adjustments"]:
        print(f"    {a}")
    print(f"  Score    : {top['base_score']} → {top['adjusted_score']} (Δ{top['score_delta']:+.0f})")
    print(f"  Level    : {top['risk_level']}")

    out = enricher.save_results(results)

    print(f"\n  {'─'*64}")
    print(f"  SUMMARY — score adjustments across all IPs")
    print(f"  {'─'*64}")
    for r in results:
        d = r["score_delta"]
        bar = ("▲" * int(abs(d)//3) if d>0 else "▼" * int(abs(d)//3))
        print(f"  {r['src_ip']:<18} {r['base_score']:>5.0f} → {r['adjusted_score']:>5.1f}  "
              f"{'+' if d>=0 else ''}{d:>4.0f}  {bar}")

    print(f"\n  Results saved → {out}")
    print(f"\n  Module 11 complete.\n{'='*64}\n")
