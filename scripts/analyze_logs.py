import re
from collections import defaultdict

log_path = 'artifacts/killer_5000_FULL.md'
try:
    with open(log_path, 'r') as f:
        content = f.read()
except FileNotFoundError:
    print("Log file not found.")
    exit()

runs = content.split('## Run')
stats = {
    'blend': defaultdict(lambda: {'ok': 0, 'fail': 0}),
    'rep': defaultdict(lambda: {'ok': 0, 'fail': 0}),
    'layer': defaultdict(lambda: {'ok': 0, 'fail': 0}),
    'cat_blend': defaultdict(lambda: defaultdict(lambda: {'ok': 0, 'fail': 0}))
}

for run in runs[1:]:  # Skip header
    is_ok = '✅' in run.split('\n')[0]
    
    # Extract params
    m = re.search(r'\*\*Category:\*\* (\w+) \| \*\*Blend:\*\* ([\d\.]+) \| \*\*Repulsion:\*\* (-[\d\.]+) \| \*\*Layers:\*\* (\d+-\d+)', run)
    if m:
        cat, blend, rep, layers = m.groups()
        
        # Update stats
        key = 'ok' if is_ok else 'fail'
        stats['blend'][blend][key] += 1
        stats['rep'][rep][key] += 1
        stats['layer'][layers][key] += 1
        stats['cat_blend'][cat][blend][key] += 1

print("\n📊 OPTIMIZATION ANALYSIS\n")

print("--- BEST BLEND ---")
for b, s in sorted(stats['blend'].items(), key=lambda x: x[1]['ok']/(x[1]['ok']+x[1]['fail']+0.01), reverse=True):
    total = s['ok'] + s['fail']
    rate = s['ok'] / total * 100
    print(f"Blend {b}: {rate:.1f}% ({s['ok']}/{total})")

print("\n--- BEST REPULSION ---")
for r, s in sorted(stats['rep'].items(), key=lambda x: x[1]['ok']/(x[1]['ok']+x[1]['fail']+0.01), reverse=True):
    total = s['ok'] + s['fail']
    rate = s['ok'] / total * 100
    print(f"Rep {r}: {rate:.1f}% ({s['ok']}/{total})")

print("\n--- BEST LAYERS ---")
for l, s in sorted(stats['layer'].items(), key=lambda x: x[1]['ok']/(x[1]['ok']+x[1]['fail']+0.01), reverse=True):
    total = s['ok'] + s['fail']
    rate = s['ok'] / total * 100
    print(f"Layers {l}: {rate:.1f}% ({s['ok']}/{total})")

print("\n--- BEST BLEND PER CATEGORY ---")
for cat, blends in stats['cat_blend'].items():
    best_b = max(blends.items(), key=lambda x: x[1]['ok']/(x[1]['ok']+x[1]['fail']+0.01))
    rate = best_b[1]['ok'] / (best_b[1]['ok'] + best_b[1]['fail']) * 100
    print(f"{cat}: Blend {best_b[0]} ({rate:.1f}%)")
