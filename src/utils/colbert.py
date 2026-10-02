import requests
import time

COLBERT_API_URL = "http://20.102.90.50:2017/wiki17_abstracts/search"

def fetch_colbert_passages(query: str, k: int = 10, retries: int = 3):
    """Hits the public Stanford ColBERT endpoint with retry logic."""
    print(f"🔍 Querying ColBERT API for: '{query}'")
    
    for attempt in range(retries):
        try:
            # The API expects query parameters
            response = requests.get(
                COLBERT_API_URL, 
                params={"query": query, "k": k}, 
                timeout=10 
            )
            response.raise_for_status()
            results = response.json()
            
            paragraphs = []
            for doc in results.get("topk", []):
                title = doc.get("title", "Unknown")
                text = doc.get("text", "")
                paragraphs.append(f"{title}: {text}")
                
            print(f"✅ Fetched {len(paragraphs)} paragraphs in {response.elapsed.total_seconds():.2f}s")
            return paragraphs
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {e}")
            time.sleep(1) # Wait 1 second before retrying
            
    print("❌ ColBERT API unreachable after retries.")
    return []

# --- Test it directly ---
if __name__ == "__main__":
    # Notice this query has a lexical gap (doesn't name Alexander Graham Bell)
    test_query = "Where was the inventor of the telephone born?"
    passages = fetch_colbert_passages(test_query)
    
    for i, p in enumerate(passages[:3], 1):
        print(f"\n[{i}] {p[:150]}...")