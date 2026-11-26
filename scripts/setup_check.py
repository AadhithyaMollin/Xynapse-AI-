#!/usr/bin/env python
"""Xynapse AI environment check.
Usage: python scripts/setup_check.py [--skip-network] [--skip-esm]
Exit code 0 = all REQUIRED checks passed."""
import argparse, importlib, os, shutil, subprocess, sys, time

PY_MODULES = [  # (import name, pip name)
    ("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("sklearn", "scikit-learn"),
    ("lightgbm", "lightgbm"), ("torch", "torch"), ("transformers", "transformers"),
    ("Bio", "biopython"), ("biotite", "biotite"), ("freesasa", "freesasa"), ("pydssp", "pydssp"),
    ("streamlit", "streamlit"), ("plotly", "plotly"), ("py3Dmol", "py3Dmol"),
    ("requests", "requests"), ("pyarrow", "pyarrow"), ("yaml", "pyyaml"), ("tqdm", "tqdm"),
]
BINARIES = {  # name -> (candidate executables, version args, required?)
    "mmseqs": (["mmseqs"], ["version"], True),
    "mafft": (["mafft"], ["--version"], True),
    "iqtree": (["iqtree2", "iqtree", "iqtree3"], ["--version"], False),
}
URLS = {  # endpoint -> (url, required?)
    "UniProt REST": ("https://rest.uniprot.org/uniprotkb/P04637.fasta", True),
    "AlphaFold DB API": ("https://alphafold.ebi.ac.uk/api/prediction/P04637", True),
    "RCSB PDB": ("https://files.rcsb.org/header/1TUP.pdb", True),
    "ClinVar FTP": ("https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz", True),
    "Zenodo (Mega-scale stability data)": ("https://zenodo.org/records/7992926", False),
    "ProteinGym": ("https://proteingym.org/download", False),
}
results = []  # (status, name, detail, required)


def record(ok, name, detail, required=True):
    results.append(("PASS" if ok else ("FAIL" if required else "WARN"), name, detail, required))


def check_modules():
    for mod, pip_name in PY_MODULES:
        try:
            m = importlib.import_module(mod)
            record(True, f"python:{mod}", getattr(m, "__version__", "ok"))
        except Exception as e:  # noqa
            record(False, f"python:{mod}", f"{type(e).__name__}: {e} -> pip install {pip_name}")


def check_binaries():
    for name, (cands, args, req) in BINARIES.items():
        exe = next((shutil.which(c) for c in cands if shutil.which(c)), None)
        if not exe:
            record(False, f"bin:{name}", f"not found (tried {cands})", req)
            continue
        try:
            p = subprocess.run([exe] + args, capture_output=True, text=True, timeout=20)
            out = (p.stdout + p.stderr).strip().splitlines()
            record(True, f"bin:{name}", f"{exe} | {out[0] if out else 'ok'}", req)
        except Exception as e:  # noqa
            record(False, f"bin:{name}", f"{type(e).__name__}: {e}", req)


def check_network():
    import requests
    for name, (url, req) in URLS.items():
        t = time.time()
        try:
            r = requests.get(url, stream=True, timeout=25, headers={"User-Agent": "xynapse-setup-check"})
            ok = r.status_code < 400
            code = r.status_code
            r.close()
            record(ok, f"net:{name}", f"HTTP {code} in {time.time()-t:.1f}s", req)
        except Exception as e:  # noqa
            record(False, f"net:{name}", f"{type(e).__name__}: {e}", req)


def check_gpu_and_esm(skip_esm):
    try:
        import torch
    except Exception as e:  # noqa
        record(False, "gpu", f"torch unavailable: {e}", False)
        return
    cuda = torch.cuda.is_available()
    if cuda:
        record(True, "gpu", torch.cuda.get_device_name(0), False)
    else:
        record(False, "gpu", "none (CPU only) -> use a Colab GPU runtime for ESM-2", False)
    if skip_esm:
        return
    check_esm()


ESM_SMOKE_CODE = """\
import torch
torch.set_num_threads(1)
from transformers import AutoTokenizer, EsmForMaskedLM
name = "facebook/esm2_t6_8M_UR50D"
tok = AutoTokenizer.from_pretrained(name)
model = EsmForMaskedLM.from_pretrained(name).eval()
seq = "MKTAYIAKQRQISFVKSHFSRQ"
with torch.no_grad():
    logits = model(**tok(seq, return_tensors="pt")).logits
assert logits.shape[1] == len(seq) + 2, logits.shape
print("ESM_OK", tuple(logits.shape))
"""


def check_esm():
    """Run the ESM smoke test in a clean subprocess.

    torch and lightgbm each ship their own libomp on macOS. Importing lightgbm
    first makes a multi-threaded torch forward pass segfault (exit 139) and
    kills this whole report. A separate process + single-threaded torch avoids
    the clash and turns a crash into an ordinary FAIL line."""
    env = {**os.environ, "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
           "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    try:
        p = subprocess.run([sys.executable, "-c", ESM_SMOKE_CODE],
                           capture_output=True, text=True, timeout=300, env=env)
    except subprocess.TimeoutExpired:
        record(False, "esm2_t6_8M smoke test", "timed out after 300s")
        return
    if p.returncode == 0:
        line = next((l for l in p.stdout.splitlines() if l.startswith("ESM_OK")), "ESM_OK ok")
        record(True, "esm2_t6_8M smoke test", line.replace("ESM_OK", "logits", 1))
        return
    why = (f"crash (signal {-p.returncode}) -> lightgbm/torch OpenMP clash"
           if p.returncode < 0 else f"exit {p.returncode}")
    tail = [l for l in (p.stderr or "").splitlines() if l.strip()]
    if tail:
        why += f": {tail[-1][:160]}"
    record(False, "esm2_t6_8M smoke test", why)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-network", action="store_true")
    ap.add_argument("--skip-esm", action="store_true")
    a = ap.parse_args()
    print(f"Python {sys.version.split()[0]} on {sys.platform} ({sys.executable})")
    check_modules()
    check_binaries()
    if not a.skip_network:
        check_network()
    check_gpu_and_esm(a.skip_esm)
    w = max(len(r[1]) for r in results)
    for status, name, detail, _ in results:
        print(f"[{status}] {name:<{w}}  {detail}")
    failed = [r for r in results if r[0] == "FAIL"]
    print(f"\n{len(results)-len(failed)} of {len(results)} checks OK; {len(failed)} required failure(s).")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
