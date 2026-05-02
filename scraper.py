import requests
from bs4 import BeautifulSoup

def scrape_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Supprime les balises inutiles
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Nettoie les lignes vides multiples
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)

def scrape_startup(nom: str, urls: list[str]) -> dict:
    print(f"Scraping {nom}...")
    textes = []
    for url in urls:
        try:
            texte = scrape_url(url)
            textes.append({"url": url, "contenu": texte})
            print(f"  ✓ {url} ({len(texte)} caractères)")
        except Exception as e:
            print(f"  ✗ {url} : {e}")
    return {"nom": nom, "sources": textes}

if __name__ == "__main__":
    # Test avec Histia
    data = scrape_startup("Histia", [
        "https://www.histia.net/fr",
        "https://epa-paris-saclay.fr/actualites-et-decryptages/toutes-nos-publications/une-ia-pour-de-linformation-fiable/"
    ])
    for source in data["sources"]:
        print(f"\n--- {source['url']} ---")
        print(source["contenu"][:500])