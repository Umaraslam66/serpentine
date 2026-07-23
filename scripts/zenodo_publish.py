#!/usr/bin/env python3
"""Publish the paper PDF to Zenodo with full metadata.

SAFETY: publishing mints a DOI and is IRREVERSIBLE. By default this script only
creates (or updates) a DRAFT deposition and prints its review URL; pass
--publish to actually publish. Test the whole flow first against the sandbox:

    export ZENODO_SANDBOX_TOKEN=...   # from https://sandbox.zenodo.org
    python3 scripts/zenodo_publish.py --sandbox

    export ZENODO_TOKEN=...           # from https://zenodo.org/account/settings/applications/
    python3 scripts/zenodo_publish.py            # real draft, review in browser
    python3 scripts/zenodo_publish.py --publish  # real DOI, no undo

The GitHub-release DOI (the .zenodo.json integration) archives the CODE; this
script deposits the PAPER as its own record. The two are linked via
related_identifiers.
"""
import argparse
import json
import os
import sys
import urllib.request

PDF = os.path.join(os.path.dirname(__file__), "..", "paper", "main.pdf")

METADATA = {
    "metadata": {
        "title": "When Mamba Needs Attention",
        "upload_type": "publication",
        "publication_type": "preprint",
        "description": (
            "<p>We ask whether a linear-time state-space (Mamba) encoder can drive routing "
            "decisions as well as a quadratic-time attention encoder on Euclidean TSP-100, "
            "under identical POMO reinforcement learning and a byte-identical decoder, with "
            "only the encoder swapped and parameter counts matched. The obvious recipe&mdash;a "
            "Hilbert-curve serialization scanned by a unidirectional Mamba&mdash;is killed: at a "
            "matched 250k-step budget it reaches a 7.50&nbsp;&plusmn;&nbsp;1.03 single-trajectory "
            "optimality gap over three seeds versus attention's 2.95%. Diagnostics localize the "
            "failure to a short, one-directional receptive field; an ordering ablation shows a "
            "locality prior confers no benefit for causal scans and roughly doubles seed "
            "variance. Repair follows the diagnosis: pooled global channels fail, a "
            "param-matched bidirectional scan reaches multistart parity with attention "
            "(1.94&nbsp;&plusmn;&nbsp;0.09 vs 1.91) and cuts seed variance tenfold, and a hybrid "
            "inserting one attention layer into a bidirectional-Mamba stack (4.5% fewer "
            "parameters) beats attention on multistart tour quality at a converged 500k budget "
            "(worst hybrid seed 1.365 &lt; best attention 1.589). All claims are scoped to "
            "N=100; the O(N) payoff remains untested.</p>"
            "<p>Code, curves, and pre-registered decision rules: "
            "<a href=\"https://github.com/Umaraslam66/serpentine\">github.com/Umaraslam66/serpentine</a></p>"
        ),
        "creators": [{"name": "Aslam, Umar", "affiliation": "Independent Researcher"}],
        "access_right": "open",
        "license": "cc-by-4.0",
        "language": "eng",
        "keywords": [
            "combinatorial optimization", "travelling salesman problem",
            "state-space models", "Mamba", "attention", "hybrid architectures",
            "space-filling curves", "reinforcement learning", "POMO",
            "neural combinatorial optimization",
        ],
        "related_identifiers": [
            {"identifier": "https://github.com/Umaraslam66/serpentine",
             "relation": "isSupplementedBy", "scheme": "url"},
        ],
    }
}


def req(method, url, token, data=None, headers=None):
    h = {"Authorization": f"Bearer {token}"}
    h.update(headers or {})
    r = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(r) as resp:
            body = resp.read()
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        sys.exit(f"HTTP {e.code} on {method} {url}\n{body}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    ap.add_argument("--publish", action="store_true",
                    help="actually publish (mints a DOI, IRREVERSIBLE). Default: draft only.")
    ap.add_argument("--deposition", help="existing draft deposition id to update instead of creating")
    a = ap.parse_args()

    base = "https://sandbox.zenodo.org/api" if a.sandbox else "https://zenodo.org/api"
    env = "ZENODO_SANDBOX_TOKEN" if a.sandbox else "ZENODO_TOKEN"
    token = os.environ.get(env) or sys.exit(f"set {env} (never pass tokens as arguments)")

    pdf = os.path.abspath(PDF)
    if not os.path.exists(pdf):
        sys.exit(f"missing {pdf} — compile first (cd paper && tectonic main.tex)")

    if a.deposition:
        dep_id = a.deposition
        _, dep = req("GET", f"{base}/deposit/depositions/{dep_id}", token)
    else:
        _, dep = req("POST", f"{base}/deposit/depositions", token,
                     data=json.dumps({}).encode(), headers={"Content-Type": "application/json"})
        dep_id = dep["id"]
        print(f"created draft deposition {dep_id}")

    req("PUT", f"{base}/deposit/depositions/{dep_id}", token,
        data=json.dumps(METADATA).encode(), headers={"Content-Type": "application/json"})
    print("metadata set")

    bucket = dep["links"]["bucket"]
    with open(pdf, "rb") as f:
        req("PUT", f"{bucket}/when-mamba-needs-attention.pdf", token, data=f.read(),
            headers={"Content-Type": "application/octet-stream"})
    print("uploaded paper/main.pdf as when-mamba-needs-attention.pdf")

    if a.publish:
        confirm = input("Type PUBLISH to mint the DOI (irreversible): ")
        if confirm != "PUBLISH":
            sys.exit("aborted; draft kept")
        _, pub = req("POST", f"{base}/deposit/depositions/{dep_id}/actions/publish", token)
        print(f"PUBLISHED. DOI: {pub.get('doi')}\nRecord: {pub['links'].get('record_html')}")
    else:
        print(f"DRAFT ready — review at: {dep['links'].get('html')}\n"
              f"Publish later with: python3 scripts/zenodo_publish.py --deposition {dep_id} --publish")


if __name__ == "__main__":
    main()
