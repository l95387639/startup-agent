import asyncio
import aiohttp
from bs4 import BeautifulSoup
import time

async def scrape_url_async(session: aiohttp.ClientSession, url: str) -> dict:
    """Scrape une URL de façon asynchrone."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
            response.raise_for_status()
            html = await response.text()
            
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            
            text = soup.get_text(separator="\n")
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            contenu = "\n".join(lines)
            
            print(f"  ✓ {url[:60]} ({len(contenu)} caractères)")
            return {"url": url, "contenu": contenu}
    except Exception as e:
        print(f"  ✗ {url[:60]} : {e}")
        return {"url": url, "contenu": ""}

async def scrape_startup_async(nom: str, urls: list[str]) -> dict:
    """Scrape toutes les URLs en parallèle."""
    print(f"Scraping async de {nom} ({len(urls)} URLs en parallèle)...")
    
    async with aiohttp.ClientSession() as session:
        taches = [scrape_url_async(session, url) for url in urls]
        sources = await asyncio.gather(*taches)
    
    # Filtre les URLs vides
    sources = [s for s in sources if s["contenu"]]
    return {"nom": nom, "sources": sources}

def scrape_startup(nom: str, urls: list[str]) -> dict:
    """Wrapper synchrone pour utiliser le scraper async."""
    return asyncio.run(scrape_startup_async(nom, urls))

if __name__ == "__main__":
    # Comparaison séquentiel vs async
    urls = [
        "https://www.histia.net/fr",
        "https://epa-paris-saclay.fr/actualites-et-decryptages/toutes-nos-publications/une-ia-pour-de-linformation-fiable/",
        "https://21st.centralesupelec.com/nos-startups/histia",
        "https://www.rimbaud-tech.fr/startup/leopold-servais/",
        "https://questforchange.eu/blog/communaute/histia/",
    ]

    # Test async
    debut = time.time()
    data = scrape_startup("Histia", urls)
    print(f"\nAsync : {time.time() - debut:.2f} secondes")
    print(f"{len(data['sources'])} sources récupérées")