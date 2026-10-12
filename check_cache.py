import pickle
c = pickle.load(open("stack_cache.pkl", "rb"))
print(type(c))
print(list(c.keys()) if hasattr(c, "keys") else len(c))

for k, v in c.items():
    print(k, type(v), getattr(v, "shape", None))

import time
t0 = time.perf_counter()
# ... your existing stack-run call here ...
print(f"Full stack run: {(time.perf_counter() - t0) / 60:.1f} min")

for name in ("oof", "test"):
    print(name)
    for k, v in c[name].items():
        print("  ", k, type(v), getattr(v, "shape", None))