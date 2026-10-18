import pandas as pd
import sys

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from openai import OpenAI
client = OpenAI(api_key=)
chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW7')
collection = chroma_client.get_or_create_collection('HW7NewsCollection')
df = pd.read_csv("news.csv")

for i in range(len(df)):
    row = df.iloc[i]
    company = row["company_name"]
    document_text = row["Document"]
    date = row["Date"]
    url = row["URL"]

    text_to_embed = f"Company: {company}\n{document_text}"

    response = client.embeddings.create(
        input=text_to_embed,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding

    collection.add(
        documents=[document_text],
        ids=[f"article_{i}"],
        embeddings=[embedding],
        metadatas=[{"company_name": company, "date": date, "url": url}]
    )

    if i % 100 == 0:
        print(f"Embedded {i} of {len(df)} articles")    

print(df.shape)
print(df.columns.tolist())
print(df.head(3))
print(f"Total documents in collection: {collection.count()}")