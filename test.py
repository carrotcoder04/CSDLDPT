from vector_db import VectorDatabase

loaded_db = VectorDatabase.load("vector_db.npz")
print(loaded_db._records[0]['image_path'])