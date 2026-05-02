import chromadb
from sentence_transformers import SentenceTransformer
from scraper import scrape_startup

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def chunk_text(texte: str, taille_max: int = 500, overlap: int = 50) -> list[str]:
    """Découpe par paragraphes, regroupe les petits, coupe les grands."""
    paragraphes = [p.strip() for p in texte.split("\n") if p.strip()]
    
    chunks = []
    chunk_courant = ""
    
    for para in paragraphes:
        if len(chunk_courant) + len(para) < taille_max:
            chunk_courant += " " + para
        else:
            if len(chunk_courant) > 100:
                chunks.append(chunk_courant.strip())
            chunk_courant = chunk_courant[-overlap:] + " " + para
    
    if chunk_courant.strip():
        chunks.append(chunk_courant.strip())
    
    return chunks

def indexer_startup(data: dict, collection_name: str = "startups"):
    """Embed et stocke les chunks dans ChromaDB."""
    
    client = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        client.delete_collection(collection_name)
    except:
        pass
    
    collection = client.create_collection(collection_name)
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"Indexation de {data['nom']}...")
    
    tous_les_chunks = []
    metadatas = []
    ids = []
    idx = 0
    
    for source in data["sources"]:
        if not source["contenu"].strip():
            print(f"  ⚠️ {source['url']} — contenu vide, ignoré")
            continue
            
        chunks = chunk_text(source["contenu"])
        
        if not chunks:
            print(f"  ⚠️ {source['url']} — aucun chunk généré, ignoré")
            continue
            
        print(f"  {source['url']} → {len(chunks)} chunks")
        
        for chunk in chunks:
            tous_les_chunks.append(chunk)
            metadatas.append({
                "startup": data["nom"],
                "url": source["url"]
            })
            ids.append(f"chunk_{idx}")
            idx += 1
    
    # Vérification critique avant d'indexer
    if not tous_les_chunks:
        print("⚠️ Aucun chunk à indexer — toutes les URLs ont échoué ou sont vides")
        return None
    
    print(f"Création de {len(tous_les_chunks)} embeddings...")
    embeddings = model.encode(tous_les_chunks).tolist()
    
    collection.add(
        documents=tous_les_chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✓ {len(tous_les_chunks)} chunks indexés dans ChromaDB")
    return collection

if __name__ == "__main__":
    data = scrape_startup("Histia", [
        "https://www.histia.net/fr",
        "https://epa-paris-saclay.fr/actualites-et-decryptages/toutes-nos-publications/une-ia-pour-de-linformation-fiable/",
    ])
    indexer_startup(data)