
import csv
from collections import defaultdict

with open('scripts/training_set_v2.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if not rows:
    print('No rows found in CSV.')
    exit()

numeric_cols = [
    'price_norm','price_median_ratio','brand_match','cond_norm',
    'seller_rating_norm','seller_feedback_norm','product_rating_norm',
    'product_review_norm','description_len_norm','free_ship'
]

print('--- Statistical Range Check ---')
for col in numeric_cols:
    if col not in rows[0]:
        print(f'Column {col} missing!')
        continue
    try:
        vals = [float(r[col]) for r in rows if r[col]]
        if vals:
            print(f'{col:22}: min={min(vals):.3f}, max={max(vals):.3f}, avg={sum(vals)/len(vals):.3f}')
        else:
            print(f'{col:22}: EMPTY')
    except Exception as e:
        print(f'{col:22}: Error {e}')

print('\n--- Relevance Variety Check (Mechanical Keyboard) ---')
kb_rows = [r for r in rows if r['query'] == 'mechanical keyboard']
per_item = defaultdict(list)
for r in kb_rows:
    per_item[r['item_id']].append((r['persona_id'], r['relevance']))

# Check first 3 items
items = list(per_item.keys())[:3]
for item_id in items:
    print(f'Item {item_id}:')
    # Filter to show only unique personas for this item
    seen = set()
    for persona, rel in per_item[item_id]:
        if persona not in seen:
            print(f'  - {persona:18}: {rel}')
            seen.add(persona)
