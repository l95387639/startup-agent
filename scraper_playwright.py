import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

async def scrape_url_playwright(url: str) -> str:
    """Scrape une URL avec Playwright — contourne les protections."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Simule un vrai navigateur
        await page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        
        try:
            await page.goto(url, timeout=15000, wait_until="networkidle")
            html = await page.content()
            
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            
            text = soup.get_text(separator="\n")
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            contenu = "\n".join(lines)
            
            print(f"  ✓ Playwright : {url[:60]} ({len(contenu)} caractères)")
            return contenu
        except Exception as e:
            print(f"  ✗ Playwright : {url[:60]} : {e}")
            return ""
        finally:
            await browser.close()

async def scrape_startup_intelligent(nom: str, urls: list[str]) -> dict:
    """Essaie d'abord aiohttp, bascule sur Playwright si bloqué."""
    import aiohttp
    from bs4 import BeautifulSoup

    print(f"Scraping intelligent de {nom}...")
    sources = []

    async with aiohttp.ClientSession() as session:
        for url in urls:
            headers = {"User-Agent": "Mozilla/5.0"}
            try:
                async with session.get(
                    url, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=8)
                ) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, "html.parser")
                        for tag in soup(["script", "style", "nav", "footer", "header"]):
                            tag.decompose()
                        text = soup.get_text(separator="\n")
                        lines = [l.strip() for l in text.splitlines() if l.strip()]
                        contenu = "\n".join(lines)
                        print(f"  ✓ aiohttp : {url[:60]}")
                        sources.append({"url": url, "contenu": contenu})
                    else:
                        # Site bloqué — bascule sur Playwright
                        print(f"  → Status {response.status}, bascule sur Playwright...")
                        contenu = await scrape_url_playwright(url)
                        if contenu:
                            sources.append({"url": url, "contenu": contenu})
            except Exception:
                # Timeout ou erreur — bascule sur Playwright
                print(f"  → Échec aiohttp, bascule sur Playwright...")
                contenu = await scrape_url_playwright(url)
                if contenu:
                    sources.append({"url": url, "contenu": contenu})

    return {"nom": nom, "sources": sources}

def scrape_startup(nom: str, urls: list[str]) -> dict:
    """Wrapper synchrone."""
    return asyncio.run(scrape_startup_intelligent(nom, urls))

if __name__ == "__main__":
    # Test sur un site qui bloque requests
    data = scrape_startup("Histia", [
        "https://www.histia.net/fr",
        "https://jobs.stationf.co/companies/histia",  # bloque souvent requests
    ])
    for s in data["sources"]:
        print(f"\n--- {s['url']} ---")
        print(s["contenu"][:300])