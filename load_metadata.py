"""
Load Amazon product metadata and save as CSV.
Extracts product info from meta_Musical_Instruments.json.gz
"""

import gzip
import csv
import json

METADATA_FILE = 'data/meta_Musical_Instruments.json.gz'
OUTPUT_FILE = 'data/metadata.csv'

def load_metadata():
    products = []
    
    print(f"Loading metadata from {METADATA_FILE}...")
    
    with gzip.open(METADATA_FILE, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                
                asin = record.get('asin', '')
                title = record.get('title', '')
                price = record.get('price', 'N/A')
                brand = record.get('brand', '')
                categories = record.get('category', [])
                
                if not asin or not title:
                    continue
                
                title = title.strip()
                
                if title.startswith('http'):
                    continue
                if 'getTime' in title:
                    continue
                if len(title) < 4:
                    continue
                
                if isinstance(categories, list) and len(categories) > 0:
                    if isinstance(categories[0], list) and len(categories[0]) > 0:
                        category = categories[0][-1]
                    else:
                        category = categories[-1] if isinstance(categories[-1], str) else 'Musical Instruments'
                else:
                    category = 'Musical Instruments'
                
                if price and isinstance(price, str):
                    price = price.strip()
                    if not price:
                        price = 'N/A'
                else:
                    price = 'N/A'
                
                products.append({
                    'item_id': asin,
                    'product_title': title,
                    'price': price,
                    'brand': brand if brand else '',
                    'category': category
                })
                
            except (json.JSONDecodeError, Exception):
                continue
    
    print(f"Loaded {len(products):,} products")
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['item_id', 'product_title', 'price', 'brand', 'category'])
        writer.writeheader()
        writer.writerows(products)
    
    print(f"Saved to {OUTPUT_FILE}")
    
    print("\nSample of 5 rows:")
    print("-" * 100)
    for p in products[:5]:
        print(f"item_id: {p['item_id']}")
        print(f"title:   {p['product_title'][:70]}...")
        print(f"price:   {p['price']}")
        print(f"brand:   {p['brand']}")
        print(f"category:{p['category']}")
        print("-" * 100)

if __name__ == '__main__':
    load_metadata()
