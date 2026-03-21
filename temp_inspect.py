import endee

client = endee.Endee()
try:
    print("Indexes prior:", client.list_indexes())
    idx = client.get_index("clinical_rag")
    print("got index")
except Exception as e:
    print("get_index error:", e)
    idx = client.create_index("clinical_rag", dimension=384, space_type="cosine")
    print("created index")

text = "Hello world"
try:
    idx.add(documents=[text], embeddings=[[0.1]*384], metadatas=[{"source":"test"}], ids=["id1"])
    print("added docs successfully")
except Exception as e:
    print("add error:", e)

try:
    res = idx.query(query_texts=["hello"], query_embeddings=[[0.1]*384], n_results=1)
    print("query result:", res)
except Exception as e:
    print("query error:", e)
