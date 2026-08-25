"""Build cosmic_signature_etiology.tsv from COSMIC's own signature pages.

One-time (re-runnable) scraper. Fetches every current COSMIC SBS signature
page (https://cancer.sanger.ac.uk/signatures/sbs/<name>/), extracts the
"Proposed aetiology" text, and classifies each signature as a likely
sequencing artifact and/or treatment-associated based on that text --
directly from COSMIC, not from a third-party tool's interpretation of it.

Re-run this whenever COSMIC releases a new signature version to refresh
the data file (new signatures default to "unclassified" -- neither
artifact nor treatment-associated -- which is the correct default per
the project's exclusion-not-inclusion design: a signature only gets
special handling with positive evidence for it).

Usage: python data-raw/build_cosmic_signature_etiology.py
Output: src/sigmutsel/data/cosmic_signature_etiology.tsv
"""

import re
import time
import urllib.request
from datetime import date
from pathlib import Path

# COSMIC v3.6 SBS signature catalog (101 signatures), confirmed by
# inspecting SigProfilerAssignment==1.1.5's bundled
# COSMIC_v3.6_SBS_GRCh38_exome.txt header directly.
SIGNATURES = [
    "SBS1", "SBS2", "SBS3", "SBS4", "SBS5", "SBS6", "SBS7a", "SBS7b",
    "SBS7c", "SBS7d", "SBS8", "SBS9", "SBS10a", "SBS10b", "SBS10c",
    "SBS10d", "SBS11", "SBS12", "SBS13", "SBS14", "SBS15", "SBS16",
    "SBS17a", "SBS17b", "SBS18", "SBS19", "SBS20", "SBS21", "SBS22a",
    "SBS22b", "SBS22c", "SBS23", "SBS24", "SBS25", "SBS26", "SBS27",
    "SBS28", "SBS29", "SBS30", "SBS31", "SBS32", "SBS33", "SBS34",
    "SBS35", "SBS36", "SBS37", "SBS38", "SBS39", "SBS40a", "SBS40b",
    "SBS40c", "SBS41", "SBS42", "SBS43", "SBS44", "SBS45", "SBS46",
    "SBS47", "SBS48", "SBS49", "SBS50", "SBS51", "SBS52", "SBS53",
    "SBS54", "SBS55", "SBS56", "SBS57", "SBS58", "SBS59", "SBS60",
    "SBS84", "SBS85", "SBS86", "SBS87", "SBS88", "SBS89", "SBS90",
    "SBS91", "SBS92", "SBS93", "SBS94", "SBS95", "SBS96", "SBS97",
    "SBS98", "SBS99", "SBS100", "SBS101", "SBS102", "SBS103", "SBS104",
    "SBS105", "SBS106", "SBS107", "SBS108", "SBS109", "SBS110",
    "SBS111", "SBS112", "SBS113",
]

BASE_URL = "https://cancer.sanger.ac.uk/signatures/sbs/{name}/"

ARTIFACT_KEYWORDS = ("artefact", "artifact")

# Named agents/drugs COSMIC's aetiology text uses for treatment-associated
# signatures, plus generic phrasing -- built from reading the actual pages
# for CES's 9 candidate signatures (SBS11, 25, 31, 32, 35, 86, 87, 90, 99)
# as a starting point, then applied to the full catalog so anything COSMIC
# itself attributes to a treatment gets caught even if CES's list missed it.
TREATMENT_KEYWORDS = (
    "chemotherap",
    "treatment with",
    "previous treatment",
    "drug",
    "temozolomide",
    "platinum",
    "azathioprine",
    "thiopurine",
    "duocarmycin",
    "melphalan",
    "immunosuppress",
)

TAG_RE = re.compile(r"<[^>]+>")
SECTION_RE = re.compile(
    r'<section id="proposed-aetiology"[^>]*>(.*?)</section>', re.DOTALL
)
PARA_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.DOTALL)


def fetch(name: str) -> str:
    req = urllib.request.Request(
        BASE_URL.format(name=name.lower()),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def extract_aetiology(html: str) -> str:
    m = SECTION_RE.search(html)
    if not m:
        return ""
    section_html = m.group(1)
    paragraphs = PARA_RE.findall(section_html)
    text = " ".join(TAG_RE.sub("", p) for p in paragraphs)
    return " ".join(text.split())


def classify(text: str) -> tuple[bool, bool]:
    lowered = text.lower()
    is_artifact = any(kw in lowered for kw in ARTIFACT_KEYWORDS)
    is_treatment = any(kw in lowered for kw in TREATMENT_KEYWORDS)
    return is_artifact, is_treatment


# Manually reviewed overrides for cases where keyword matching alone
# is misleading -- found by reading the full aetiology text of every
# keyword-flagged candidate before trusting the automated pass (per
# the project's "candidate list, then verify" policy, not an
# automation shortcut). Re-running this script after a COSMIC update
# should re-check these two specifically in case the page text has
# changed, but they should not silently disappear.
#
# SBS3: keyword-matched on "platinum"/"treatment" (its own aetiology
# is homologous-recombination/BRCA1/BRCA2 deficiency); the treatment
# language describes SBS3 as a *predictive biomarker* of platinum-
# therapy response, not a signature *caused by* treatment. One of the
# most biologically important signatures in the catalog -- must not
# be excluded as treatment-associated.
#
# SBS98: keyword-matched on "treatment" appearing only in a comparison
# to SBS87 ("similar to SBS87 ... which has been linked to thiopurine
# treatment"). SBS98's own aetiology is "Unknown" -- the mention
# describes a *different* signature, not SBS98 itself.
#
# SBS113: aetiology is "Ganciclovir exposure" -- ganciclovir is a
# therapeutic antiviral drug (used clinically, including in
# immunosuppressed/transplant patients), not an environmental/dietary/
# occupational exposure like the other "exposure"-worded signatures
# (UV, aristolochic acid, aflatoxin, haloalkanes, colibactin). Missed
# by the keyword list since the text doesn't use "treatment",
# "chemotherap-", or any of the specific drug names already listed --
# found by manually reviewing every "exposure"-mentioning signature's
# full text, not just the keyword-matched ones.
TREATMENT_OVERRIDES = {
    "SBS3": False,
    "SBS98": False,
    "SBS113": True,
}


def main() -> None:
    out_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "sigmutsel"
        / "data"
        / "cosmic_signature_etiology.tsv"
    )
    today = date.today().isoformat()
    rows = []
    for name in SIGNATURES:
        url = BASE_URL.format(name=name.lower())
        try:
            html = fetch(name)
            aetiology = extract_aetiology(html)
        except Exception as exc:  # noqa: BLE001 -- log and continue
            print(f"  FAILED {name}: {exc}")
            aetiology = ""
        is_artifact, is_treatment = classify(aetiology)
        if name in TREATMENT_OVERRIDES:
            is_treatment = TREATMENT_OVERRIDES[name]
        flag = []
        if is_artifact:
            flag.append("ARTIFACT")
        if is_treatment:
            flag.append("TREATMENT")
        print(f"{name:10s} {','.join(flag) or '-':18s} {aetiology[:80]}")
        rows.append(
            {
                "signature": name,
                "aetiology_text": aetiology,
                "artifact": is_artifact,
                "treatment_associated": is_treatment,
                "source_url": url,
                "retrieved_date": today,
            }
        )
        time.sleep(0.4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(
            "signature\taetiology_text\tartifact\ttreatment_associated\t"
            "source_url\tretrieved_date\n"
        )
        for row in rows:
            fh.write(
                "{signature}\t{aetiology_text}\t{artifact}\t"
                "{treatment_associated}\t{source_url}\t{retrieved_date}\n"
                .format(**row)
            )
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
