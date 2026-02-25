"""
MitreWhitelistLoader.py
-----------------------
Charge dynamiquement la whitelist des groupes CTI (threat actors, APT groups)
depuis le STIX officiel MITRE ATT&CK Enterprise, avec cache local et TTL.

Flux :
    1. Vérification du cache local  (mitre_whitelist_cache.json)
       → valide si âge < ttl_days  → retourne le set mis en cache
    2. Si cache absent ou périmé    → requête HTTPS vers GitHub (STIX JSON)
       → extrait noms + aliases de tous les `intrusion-set`
       → sauvegarde cache + timestamp
    3. Fallback                     → si réseau indisponible et aucun cache,
       retourne un set minimal codé en dur pour ne pas bloquer le pipeline.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone

import requests

log = logging.getLogger(__name__)

# URL du STIX Enterprise MITRE ATT&CK (mise à jour régulière par MITRE)
_MITRE_STIX_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/"
    "enterprise-attack/enterprise-attack.json"
)

# Cache stocké dans le même dossier que ce script
_CACHE_FILE = os.path.join(os.path.dirname(__file__), "mitre_whitelist_cache.json")

# Fallback minimal si réseau ET cache tous les deux indisponibles
_FALLBACK_WHITELIST: set[str] = {
    "apt28", "apt29", "apt1", "apt10", "apt40",
    "lazarus", "fancy bear", "cozy bear", "sandworm",
    "crowdstrike", "fireeye", "mandiant", "microsoft",
}


class MitreWhitelistLoader:
    """
    Charge et met en cache la whitelist CTI depuis MITRE ATT&CK.

    Paramètres
    ----------
    ttl_days : int
        Durée de validité du cache en jours (défaut : 7).
    timeout : int
        Timeout HTTP en secondes (défaut : 15).
    """

    def __init__(self, ttl_days: int = 7, timeout: int = 15):
        self.ttl_days = ttl_days
        self.timeout = timeout

    # ── Interface publique ─────────────────────────────────────────────────────

    def get_whitelist(self) -> set[str]:
        """
        Retourne le set de noms/aliases CTI (lowercased).
        Utilise le cache si valide, sinon rafraîchit depuis MITRE.
        """
        cached = self._load_cache()
        if cached is not None:
            log.info(f"[MitreWhitelistLoader] Cache valide : {len(cached)} entrées")
            return cached

        log.info("[MitreWhitelistLoader] Rafraîchissement depuis MITRE ATT&CK...")
        fetched = self._fetch_from_mitre()
        if fetched:
            self._save_cache(fetched)
            log.info(f"[MitreWhitelistLoader] {len(fetched)} entrées chargées et mises en cache")
            return fetched

        # Fallback : on tente de lire un cache périmé plutôt que rien
        stale = self._load_cache(ignore_ttl=True)
        if stale:
            log.warning(
                "[MitreWhitelistLoader] Réseau indisponible — cache périmé utilisé "
                f"({len(stale)} entrées)"
            )
            return stale

        log.warning(
            "[MitreWhitelistLoader] Réseau indisponible et aucun cache — "
            "fallback minimal activé"
        )
        return _FALLBACK_WHITELIST.copy()

    # ── Cache ──────────────────────────────────────────────────────────────────

    def _load_cache(self, ignore_ttl: bool = False) -> set[str] | None:
        """Lit le cache local. Retourne None si absent ou périmé (sauf ignore_ttl)."""
        if not os.path.isfile(_CACHE_FILE):
            return None
        try:
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not ignore_ttl:
                fetched_at = datetime.fromisoformat(data["fetched_at"])
                expiry = fetched_at + timedelta(days=self.ttl_days)
                if datetime.now(timezone.utc) > expiry:
                    return None  # périmé
            return set(data["whitelist"])
        except Exception as e:
            log.warning(f"[MitreWhitelistLoader] Lecture du cache échouée : {e}")
            return None

    def _save_cache(self, whitelist: set[str]) -> None:
        """Persiste le set dans le cache JSON avec timestamp UTC."""
        try:
            payload = {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "ttl_days": self.ttl_days,
                "whitelist": sorted(whitelist),
            }
            with open(_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning(f"[MitreWhitelistLoader] Sauvegarde du cache échouée : {e}")

    # ── Fetch MITRE ────────────────────────────────────────────────────────────

    def _fetch_from_mitre(self) -> set[str] | None:
        """
        Télécharge le STIX Enterprise et extrait les noms + aliases
        de tous les objets `intrusion-set` (threat actors / APT groups).
        """
        try:
            resp = requests.get(_MITRE_STIX_URL, timeout=self.timeout)
            resp.raise_for_status()
            stix = resp.json()
        except Exception as e:
            log.error(f"[MitreWhitelistLoader] Requête MITRE échouée : {e}")
            return None

        whitelist: set[str] = set()
        for obj in stix.get("objects", []):
            if obj.get("type") != "intrusion-set":
                continue
            # Nom principal (ex. "APT28")
            name = obj.get("name", "")
            if name:
                whitelist.add(name.lower())
            # Aliases (ex. ["Fancy Bear", "STRONTIUM", "Sofacy"])
            for alias in obj.get("aliases", []):
                if alias:
                    whitelist.add(alias.lower())

        return whitelist if whitelist else None
