import json
import sys
import io
from vector_db import VectorDatabase

# Force stdout to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def run_query():
    print("Loading database...")
    db = VectorDatabase.load("vector_db.npz")
    
    query_image_path = db._records[0]["image_path"]
    print(f"\n=> Using the first image as query: {query_image_path}")
    
    k = 5
    print(f"\nQuerying top {k} similar images...")
    
    results = db.query_by_path(query_image_path, k=k)
    
    print("\n" + "="*70)
    print("  QUERY RESULTS")
    print("="*70)
    
    for i, r in enumerate(results):
        print(f"Rank {r['rank']}: {r['image_path']}")
        print(f"  - Label: {r['label']}")
        print(f"  - Euclidean distance: {r['distance']:.4f}")
        print("-" * 70)

if __name__ == "__main__":
    run_query()
