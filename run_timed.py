import subprocess, sys, time
import pandas as pd

# (label for Table 3.4, script file) - edit the list to match your filenames
SCRIPTS = [
    ("Fuzzy augmentation, Exp A (no selection bias)", "fuzzy_augmentation.py"),
    ("Fuzzy augmentation, Exp B (hidden rejects)",    "fuzzy_augmentation_tail.py"),
    ("SHAP reason reports",                           "generate_reason.py"),
    ("DiCE counterfactuals",                          "dice_constrained.py"),
    ("Transition matrices",                           "transition_matrices_fit.py"),
    ("Monte Carlo simulator",                         "simulator.py"),
    ("Ten-seed repeated runs",                        "repeated_runs.py"),
]

rows = []
for label, script in SCRIPTS:
    print(f"Running {script} ...")
    t0 = time.perf_counter()
    result = subprocess.run([sys.executable, script])
    secs = time.perf_counter() - t0
    status = "ok" if result.returncode == 0 else f"FAILED ({result.returncode})"
    rows.append({"component": label, "script": script,
                 "seconds": round(secs, 1), "minutes": round(secs / 60, 2),
                 "status": status})
    print(f"  {status}, {secs:.1f}s\n")

df = pd.DataFrame(rows)
total = pd.DataFrame([{"component": "Total", "script": "",
                       "seconds": df["seconds"].sum().round(1),
                       "minutes": df["minutes"].sum().round(2), "status": ""}])
out = pd.concat([df, total], ignore_index=True)
out.to_csv("runtime_table.csv", index=False)
print(out.to_string(index=False))