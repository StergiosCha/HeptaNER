#!/usr/bin/env python3
"""
Genre-discriminative symbolic engine for Heptanesian notarial acts.
Scans normalized act text for formulaic phrases diagnostic of specific legal genres.

Each rule maps a normalized phrase → (genre, weight, description).
Weight reflects how discriminative the phrase is:
  3 = near-unique to one genre (e.g. "ψιχικις μοι σοτιριας" → Testamentum)
  2 = strongly associated but can appear in related genres
  1 = weakly associated, appears across genres

The engine produces a ranked list of genre candidates with evidence.
"""

import sys
import os
import re
import unicodedata
from typing import List, Dict, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from corpus_linguistics import normalize_text


# ── Genre-Discriminative Rules ─────────────────────────────────────────────
# Format: (normalized_phrase, target_genre, weight, human_readable_description)

RULES: List[Tuple[str, str, int, str]] = [

    # ═══ VENDITIO (Sale) ═══
    ("εποιλισεν", "Venditio", 3,
     "επούλησεν — 'sold': the core verb of sale transactions"),
    ("αγοραστιν", "Venditio", 3,
     "αγοραστήν — 'buyer': identifies a purchaser"),
    ("εποιλισαν", "Venditio", 3,
     "επούλησαν — 'they sold': plural form of sale verb"),
    ("τιμιν δοικατα", "Venditio", 2,
     "τιμήν δουκάτα — 'price in ducats': explicit price statement"),
    ("δια τιμιν", "Venditio", 2,
     "διά τιμήν — 'for a price': price clause"),

    # ═══ EMPHYTEUSIS ═══
    ("εμφιτεισεν", "Emphyteusis", 3,
     "εμφύτευσεν — 'granted in emphyteusis': the core verb"),
    ("κανονιν", "Emphyteusis", 3,
     "κανόνιν — 'canon/annual rent': the hallmark of emphyteusis"),
    ("δεκατον", "Emphyteusis", 2,
     "δέκατον — 'tithe/tenth': typical emphyteutic obligation"),
    ("σινιθιαν τον χοραφιον", "Emphyteusis", 2,
     "συνήθειαν τῶν χωραφίων — 'custom of the fields': agricultural obligation clause"),

    # ═══ TESTAMENTUM (Will) ═══
    ("ψιχικις μοι σοτιριας", "Testamentum", 3,
     "ψυχικής μου σωτηρίας — 'for my soul's salvation': testament opening formula"),
    ("ασθενις ον κε κλινιρις", "Testamentum", 3,
     "ασθενής ών και κλινήρης — 'being sick and bedridden': testator's condition formula"),
    ("αδιλον κε αορον θανατον", "Testamentum", 3,
     "ἄδηλον καί ἄωρον θάνατον — 'uncertain and untimely death': mortality clause"),
    ("αφινο", "Testamentum", 2,
     "αφήνω — 'I leave/bequeath': dispositional verb of wills"),
    ("κλιρονομον μοι", "Testamentum", 2,
     "κληρονόμον μου — 'my heir': heir designation"),
    ("δια ψιχικις μοι", "Testamentum", 3,
     "διά ψυχικής μου — 'for my soul': pious bequest formula"),
    ("αδιορθοτα μινοσιν", "Testamentum", 3,
     "αδιόρθωτα μείνωσιν — 'remain unresolved': fear of dying intestate"),

    # ═══ PRAEVENDITIO (Forward sale) ═══
    ("οπλιγαδο", "Praevenditio", 3,
     "οπληγάδο — 'pledged/obligated': grape/olive produce pledged before harvest"),
    ("ις τιν ερχομενιν εσοδιαν", "Praevenditio", 3,
     "εις την ερχομένην εσωδίαν — 'in the coming harvest': delivery at future harvest"),
    ("ις τιν ερχομενιν", "Praevenditio", 2,
     "εις την ερχομένην — 'in the coming [season]': future delivery marker"),
    ("λαδι ξεστες", "Praevenditio", 2,
     "λάδι ξέστες — 'oil measures': agricultural produce as payment"),
    ("κρασι μετρα", "Praevenditio", 2,
     "κρασί μέτρα — 'wine measures': wine as future delivery"),
    ("να τοι δοσοιν λαδ", "Praevenditio", 3,
     "να του δώσουν λάδ(ι) — 'to give him oil': forward delivery of oil"),

    # ═══ PROCURATIO (Power of attorney) ═══
    ("κοιμεσον κε πρεκοιρατοραν", "Procuratio", 3,
     "κουμέσον και πρεκουράτοραν — 'commissioner and procurator': appointment formula"),
    ("πρεκοιρατοραν", "Procuratio", 3,
     "πρεκουράτοραν — 'procurator': the procurator title"),
    ("πασαν δικεον κε οφελον", "Procuratio", 3,
     "πᾶσαν δίκαιον και ὄφελον — 'every right and benefit': scope of procuration"),
    ("ις το να γιρειι", "Procuratio", 2,
     "εις το να γυρεύη — 'to seek': purpose of procuration"),
    ("κοιντοιτοιρος", "Procuratio", 2,
     "κουντουτούρος — 'conductor/agent': official role"),

    # ═══ QUIETANTIA (Receipt) ═══
    ("πλιρομενος κε ειχαριστιμενος", "Quietantia", 3,
     "πληρωμένος και ευχαριστημένος — 'paid and satisfied': receipt confirmation formula"),
    ("αποπλιρομι", "Quietantia", 3,
     "αποπληρωμή — 'full payment': payment completion"),
    ("αποπλιρομι κε τελια εξοφλισι", "Quietantia", 3,
     "αποπληρωμή και τελεία εξόφληση — 'full payment and complete settlement'"),
    ("κραζετε πλιρομενος", "Quietantia", 3,
     "κράζεται πληρωμένος — 'calls himself paid': formal receipt declaration"),
    ("σοα κε ανελιπες", "Quietantia", 2,
     "σώα και ανελλιπώς — 'whole and complete': payment completeness formula"),

    # ═══ PERMUTATIO (Exchange) ═══
    ("ανταλαγιν", "Permutatio", 3,
     "ανταλλαγήν — 'exchange': the core noun of exchange contracts"),
    ("εποιισαν ανταλαγιν", "Permutatio", 3,
     "εποίησαν ανταλλαγήν — 'they made an exchange': exchange verb phrase"),
    ("απο ενος μεροις", "Permutatio", 2,
     "από ενός μέρους — 'from one part': bilateral structure marker"),
    ("απο ετεροι μεροις", "Permutatio", 2,
     "από ετέρου μέρους — 'from the other part': bilateral structure marker"),
    ("ο ις προς τον ετερον", "Permutatio", 2,
     "ο εις προς τον έτερον — 'the one to the other': mutual exchange formula"),

    # ═══ PROCEDURAL ACTA ═══
    ("κοφθοιν κε ιλιονοιν", "Procedural acta", 3,
     "κόφθουν και ηλλοιώνουν — 'cut and alter': document cancellation formula"),
    ("κριτε αλπιτρε", "Procedural acta", 3,
     "κριταί άλπιτραι — 'judges arbiters': arbitration formula"),
    ("σινιβαστε", "Procedural acta", 3,
     "συνηβασταί — 'reconcilers': arbitration role"),
    ("παρετιτε κε αφινι", "Procedural acta", 3,
     "παραιτείται και αφίνει — 'renounces and releases': renunciation formula"),
    ("αδελφοσινι", "Procedural acta", 2,
     "αδελφοσύνη — 'brotherhood': confraternity procedural context"),
    ("να μιδεν εχι ισχιν", "Procedural acta", 3,
     "να μηδέν έχη ισχύν — 'to have no force': document annulment formula"),

    # ═══ CONCESSIO ECCLESIASTICA ═══
    ("ιεροσινις θιον αξιομα", "Concessio ecclesiastica", 3,
     "ιεροσύνης θείον αξίωμα — 'the divine office of priesthood': ordination context"),
    ("επιτροποι τις εκκλισιας", "Concessio ecclesiastica", 3,
     "επίτροποι της εκκλησίας — 'trustees of the church': ecclesiastical administration"),
    ("ναον τις ιπεραγιας", "Concessio ecclesiastica", 3,
     "ναόν της Υπεραγίας — 'temple of the Most Holy': church property"),
    ("εκκλισιαστικον", "Concessio ecclesiastica", 2,
     "εκκλησιαστικόν — 'ecclesiastical': general church context"),

    # ═══ DIVISIO / PARTITIO ═══
    ("να ιμιρασοιν", "Divisio / Partitio", 3,
     "να ημυράσουν — 'to divide': division verb"),
    ("μεριδιον", "Divisio / Partitio", 3,
     "μερίδιον — 'portion/share': property share"),
    ("ιμοιρασαν", "Divisio / Partitio", 3,
     "ημοίρασαν — 'they divided': division verb past"),
    ("εν το μεσο ετον", "Divisio / Partitio", 3,
     "εν το μέσο ετών — 'in the middle of years': partition boundary"),

    # ═══ MUTUUM (Loan) ═══
    ("να χρεοστοιν", "Mutuum", 3,
     "να χρεωστούν — 'to owe': debt obligation"),
    ("ελαβαν εκ τον", "Mutuum", 2,
     "έλαβαν εκ τον — 'they took from': borrowing formula"),
    ("εδανισα", "Mutuum", 3,
     "εδάνεισα — 'I lent': lending verb"),
    ("δανιον", "Mutuum", 3,
     "δάνειον — 'loan': the loan noun"),

    # ═══ INSTRUMENTUM OBLIGATIONIS ═══
    ("ιποσχετε να δοσι", "Instrumentum obligationis", 2,
     "υπόσχεται να δώση — 'promises to give': debt promise"),
    ("ρεστο", "Instrumentum obligationis", 2,
     "ρέστο — 'remainder/balance': outstanding debt"),

    # ═══ SOCIETAS (Partnership) ═══
    ("ολιτριβιον", "Societas", 3,
     "ολοτρίβιον — 'olive press': partnership asset"),
    ("μισιακον", "Societas", 3,
     "μυσιακόν — 'in sharecropping': partnership arrangement"),
    ("δια να δοιλειι", "Societas", 2,
     "διά να δουλεύη — 'to work': labor partnership"),

    # ═══ DONATIO ═══
    ("δοροκε", "Donatio", 3,
     "δωροκεῖ — 'donates': donation verb (rare but diagnostic)"),
    ("χαρισμα", "Donatio", 3,
     "χάρισμα — 'gift': donation noun"),

    # ═══ LOCATIO CONDUCTIO REI (Lease) ═══
    ("αχρι δια χρονοις", "Locatio conductio rei", 2,
     "άχρι διά χρόνους — 'for [a number of] years': fixed-term lease"),
    ("δια μισιακον", "Locatio conductio rei", 2,
     "διά μυσιακόν — 'for sharecropping': lease arrangement"),

    # ═══ TRANSACTIO (Settlement) ═══
    ("σινιβασοσιν", "Transactio", 3,
     "συνηβάσωσιν — 'to reconcile/settle': settlement verb"),
    ("κε ιλθαν ις σιμβιβασιν", "Transactio", 3,
     "και ήλθαν εις συμβίβασιν — 'they came to a settlement'"),

    # ═══ GENERIC TRANSFER (shared by Venditio, Emphyteusis, Donatio, Permutatio) ═══
    # These contribute weak evidence to multiple genres
    ("εδοσεν, επαρεδοσεν", "Venditio", 1,
     "έδωσεν, επαρέδωσεν — 'gave and delivered': generic transfer formula"),
    ("επαρεδοσεν", "Venditio", 1,
     "επαρέδωσεν — 'delivered': generic transfer verb"),
    ("ελειθεροσεν", "Venditio", 1,
     "ελευθέρωσεν — 'freed/transferred': generic transfer liberation"),

    # NOTE: "κληρονόμοις και διαδόχοις" (heirs and successors) and "εις τον αιώνα"
    # (in perpetuity) are BOILERPLATE that appear in 30-40% of all acts across many
    # genres. They are NOT discriminative for Emphyteusis and have been REMOVED as
    # rules to avoid systematic false positives.
]


def run_symbolic_engine(act_text: str) -> Dict:
    """
    Run the symbolic engine on a single act.

    Returns:
        {
            "genre_scores": {genre: total_score, ...},
            "top_candidate": "GenreName" or None,
            "confidence": "HIGH" | "MEDIUM" | "LOW" | "NONE",
            "evidence": [
                {"genre": ..., "phrase": ..., "weight": ..., "description": ...},
                ...
            ]
        }
    """
    norm_text = normalize_text(act_text)

    genre_scores = defaultdict(float)
    evidence = []

    for phrase, genre, weight, description in RULES:
        if phrase in norm_text:
            genre_scores[genre] += weight
            evidence.append({
                "genre": genre,
                "phrase": phrase,
                "weight": weight,
                "description": description,
            })

    if not genre_scores:
        return {
            "genre_scores": {},
            "top_candidate": None,
            "confidence": "NONE",
            "evidence": [],
        }

    # Rank genres
    ranked = sorted(genre_scores.items(), key=lambda x: -x[1])
    top_genre = ranked[0][0]
    top_score = ranked[0][1]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    # Confidence based on score and margin
    if top_score >= 6 and (top_score - second_score) >= 3:
        confidence = "HIGH"
    elif top_score >= 3 and (top_score - second_score) >= 1:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "genre_scores": dict(ranked),
        "top_candidate": top_genre,
        "confidence": confidence,
        "evidence": evidence,
    }


def format_evidence_for_prompt(result: Dict) -> str:
    """
    Format symbolic engine output as human-readable text for the LLM prompt.
    """
    if result["confidence"] == "NONE":
        return "No diagnostic formulas were detected in this act. Classify based on content analysis alone."

    lines = []
    lines.append(f"Top symbolic candidate: {result['top_candidate']} (confidence: {result['confidence']})")
    lines.append(f"Genre scores: {result['genre_scores']}")
    lines.append("")
    lines.append("Matched diagnostic formulas:")

    for ev in result["evidence"]:
        lines.append(f"  [{ev['genre']}] (weight={ev['weight']}) {ev['description']}")

    if result["confidence"] == "LOW":
        lines.append("")
        lines.append("NOTE: Low confidence — multiple genres have similar scores. Use your own analysis.")

    return "\n".join(lines)


# ── CLI Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import pandas as pd

    # Load benchmark
    df = pd.read_excel("testing_acts.xlsx", sheet_name="Additional acts")
    df["Act ID"] = df["Act ID"].apply(lambda x: str(int(float(x))))
    gt = {(str(r["Notary"]).upper(), str(r["Act ID"])): str(r["CORE CATEGORY_LA"])
          for _, r in df.iterrows()}

    with open("data/benchmark_40_acts.json") as f:
        acts = json.load(f)

    correct = 0
    total = 0
    no_hit = 0

    print(f"{'Genre':<30} {'Act':<18} {'Conf':<8} {'Prediction':<30} {'Match'}")
    print("=" * 120)

    for a in acts:
        key = (a["corpus"].upper(), str(a["act_id"]))
        true_genre = gt.get(key, "?")

        result = run_symbolic_engine(a["content"])
        pred = result["top_candidate"] or "—"
        conf = result["confidence"]

        match = "✓" if pred == true_genre else "✗"
        if pred == "—":
            no_hit += 1
            match = "—"
        else:
            total += 1
            if pred == true_genre:
                correct += 1

        print(f"{true_genre:<30} {a['corpus']}/{a['act_id']:<12} {conf:<8} {pred:<30} {match}")

    print(f"\n{'='*120}")
    print(f"Symbolic engine accuracy: {correct}/{total} = {correct/total*100:.1f}% (on {total} acts with predictions)")
    print(f"No prediction: {no_hit}/40 acts")
    print(f"Overall: {correct}/40 = {correct/40*100:.1f}%")
