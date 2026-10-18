import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW7')
collection = chroma_client.get_or_create_collection('HW7NewsCollection')
print(f"Total documents in collection: {collection.count()}")