import csv, glob, io, os, sys, unicodedata
from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from datasets import Dataset, Audio

CACHE = "/root/.cache/huggingface/datasets/SPRINGLab___indic_tts_telugu/default/0.0.0/7ab3c18491bef3f5d4e820c2701b5f78f54031d6"
W = "/kaggle/working/telugu_manifest"
RAW = "/kaggle/working/FastSpeech/Telugu/data/raw/male"
MFA = "/kaggle/working/mfa_corpus/male"
os.makedirs(RAW, exist_ok=True); os.makedirs(MFA, exist_ok=True)

sys.path.insert(0, "/kaggle/working/FastSpeech/Telugu")
os.chdir("/kaggle/working/FastSpeech/Telugu")
import text as te
from text.telugu import normalize_text
assert "/FastSpeech/Telugu/" in te.__file__, te.__file__
print("frontend:", te.__file__, flush=True)

rows = list(csv.DictReader(open(W + "/male_capped.csv", encoding="utf-8-sig", newline="")))
want = {r["clip_id"]: r for r in rows}
print("target clips:", len(want), flush=True)

def mfa_text(s):
    t = normalize_text(s)
    t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)
    return " ".join(t.split())

def rs(x, sr_in, sr_out):
    g = gcd(sr_in, sr_out)
    return np.clip(resample_poly(x, sr_out // g, sr_in // g), -1.0, 1.0).astype("float32")

shards = sorted(glob.glob(os.path.join(CACHE, "*.arrow")))
offset = 0; done = bad = 0
report = []
for shard in shards:
    ds = Dataset.from_file(shard).cast_column("audio", Audio(decode=False))
    for j in range(len(ds)):
        row = ds[j]
        stem = os.path.splitext(os.path.basename(row["audio"]["path"]))[0]
        if stem not in want:
            continue
        r = want[stem]
        try:
            data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            sf.write(os.path.join(RAW, stem + ".wav"), rs(data, sr, 22050), 22050, subtype="PCM_16")
            with open(os.path.join(RAW, stem + ".lab"), "w", encoding="utf-8") as f:
                f.write(r["transcript"].strip() + "\n")
            mt = mfa_text(r["transcript"])
            if mt:
                sf.write(os.path.join(MFA, stem + ".wav"), rs(data, sr, 16000), 16000, subtype="PCM_16")
                with open(os.path.join(MFA, stem + ".lab"), "w", encoding="utf-8") as f:
                    f.write(mt + "\n")
                report.append((stem, "ok", ""))
            else:
                report.append((stem, "empty_mfa_text", ""))
        except Exception as e:
            bad += 1
            report.append((stem, "error", repr(e)))
            print("FAIL", stem, repr(e), flush=True)
        done += 1
        if done % 200 == 0:
            print("exported", done, "bad", bad, flush=True)
    offset += len(ds)

with open("/kaggle/working/export_male_log.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["clip_id", "status", "detail"]); w.writerows(report)
print("DONE exported", done, "bad", bad, "of", len(want), flush=True)
