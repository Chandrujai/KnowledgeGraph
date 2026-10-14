"""
Tamil Nadu Agriculture - Farmers Welfare Department
Knowledge Graph RAG: scrape scheme pages, store graph + text chunks in Neo4j,
and answer questions with OpenAI.

Usage:
  python agriculture_kg_rag.py scrape
  python agriculture_kg_rag.py ask "Who is eligible for seed subsidies?"
  python agriculture_kg_rag.py
"""
import os
import re
import sys
import time
import hashlib
from urllib.parse import urljoin, urldefrag, urlparse

import requests
import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()

START_URL = os.getenv(
    "START_URL",
    "https://www.tn.gov.in/scheme_list.php?dep_id=Mg==",
)
DEPARTMENT = "Agriculture - Farmers Welfare Department"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
MAX_PAGES = int(os.getenv("MAX_PAGES", "500"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.8"))
USER_AGENT = "TN-Agriculture-KG-RAG/1.0 (polite research crawler)"

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Set OPENAI_API_KEY in your .env file.")
if not NEO4J_PASSWORD:
    raise RuntimeError("Set NEO4J_PASSWORD in your .env file.")

openai_client = OpenAI()
driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)
session_http = requests.Session()
session_http.headers.update({"User-Agent": USER_AGENT})


def canonical_url(url):
    """Remove fragments and common tracking parameters; keep scheme/host/path."""
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None
    if not parsed.netloc:
        return None
    return url


def fetch_soup(url):
    response = session_http.get(url, timeout=25)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" not in content_type and not response.text.lstrip().startswith("<"):
        return None
    return BeautifulSoup(response.text, "lxml")


def clean_text(soup):
    for node in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
        node.decompose()
    main = soup.find("main") or soup.find("article") or soup.find(id=re.compile(r"content|main", re.I))
    root = main or soup.body or soup
    lines = [re.sub(r"\s+", " ", x).strip() for x in root.stripped_strings]
    return "\n".join(line for line in lines if line)


def page_title(soup, fallback):
    h = soup.find("h1") or soup.find("h2") or soup.find("title")
    return re.sub(r"\s+", " ", h.get_text(" ", strip=True)) if h else fallback


def find_scheme_links(listing_soup, listing_url):
    """
    Find links associated with the department heading.
    First try the heading's local list/container; if the legacy page structure
    does not expose a clear section, use same-site links from this department
    listing page, excluding navigation and obvious unrelated links.
    Review the printed URLs before relying on a first full production crawl.
    """
    heading = None
    for tag in listing_soup.find_all(["h1", "h2", "h3", "h4", "strong", "b", "p", "td", "div"]):
        text = re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip().lower()
        if "agriculture" in text and "farmers welfare" in text:
            heading = tag
            break

    links = []
    if heading:
        # Look in progressively larger local containers, preferring a small one.
        for parent in heading.parents:
            if getattr(parent, "name", None) in ("body", "html"):
                break
            found = parent.find_all("a", href=True)
            if len(found) >= 2:
                links = found
                break

        # Also inspect following siblings until another department heading.
        if not links:
            for sibling in heading.next_elements:
                if getattr(sibling, "name", None) in ("h1", "h2", "h3", "h4"):
                    label = sibling.get_text(" ", strip=True).lower()
                    if "department" in label or "schemes" in label:
                        break
                if getattr(sibling, "name", None) == "a" and sibling.get("href"):
                    links.append(sibling)

    # Legacy pages may not use semantic HTML. Fall back to all useful links
    # on the specific department listing page, but keep the same host.
    if not links:
        links = listing_soup.find_all("a", href=True)

    host = urlparse(listing_url).netloc.lower()
    excluded = (
        "home", "contact", "feedback", "login", "accessibility", "sitemap",
        "screen reader", "skip to", "english", "தமிழ்"
    )
    result = {}
    for a in links:
        href = canonical_url(urljoin(listing_url, a.get("href", "")))
        label = re.sub(r"\s+", " ", a.get_text(" ", strip=True)).strip()
        if not href or not label or len(label) < 3:
            continue
        if urlparse(href).netloc.lower() != host:
            continue
        if any(word in label.lower() for word in excluded):
            continue
        if href.rstrip("/") == listing_url.rstrip("/"):
            continue
        result[href] = label
    return result


def split_chunks(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if boundary > start + size // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def embed_texts(texts):
    """Embed in modest batches to keep requests manageable."""
    vectors = []
    for i in range(0, len(texts), 64):
        batch = texts[i:i + 64]
        result = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend([item.embedding for item in result.data])
    return vectors


def setup_neo4j():
    with driver.session() as session:
        session.run("""
            CREATE CONSTRAINT scheme_url IF NOT EXISTS
            FOR (s:Scheme) REQUIRE s.url IS UNIQUE
        """)
        session.run("""
            CREATE CONSTRAINT department_name IF NOT EXISTS
            FOR (d:Department) REQUIRE d.name IS UNIQUE
        """)
        session.run("""
            CREATE CONSTRAINT chunk_id IF NOT EXISTS
            FOR (c:TextChunk) REQUIRE c.id IS UNIQUE
        """)
        session.run("""
            CREATE VECTOR INDEX scheme_chunk_embeddings IF NOT EXISTS
            FOR (c:TextChunk) ON (c.embedding)
            OPTIONS {indexConfig: {
              `vector.dimensions`: 1536,
              `vector.similarity_function`: 'cosine'
            }}
        """)
        session.run(
            "MERGE (d:Department {name: $name})",
            name=DEPARTMENT,
        )


def store_page(title, url, text, chunks):
    page_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    chunk_vectors = embed_texts(chunks) if chunks else []
    with driver.session() as session:
        session.run("""
            MERGE (d:Department {name: $department})
            MERGE (s:Scheme {url: $url})
            SET s.name = $title,
                s.text = $text,
                s.content_hash = $content_hash,
                s.source_url = $url,
                s.scraped_at = datetime()
            MERGE (d)-[:OFFERS_SCHEME]->(s)
        """, department=DEPARTMENT, url=url, title=title,
             text=text, content_hash=page_hash)

        # Remove old chunks so a re-scrape does not leave stale text behind.
        session.run("""
            MATCH (s:Scheme {url: $url})-[:HAS_CHUNK]->(c:TextChunk)
            DETACH DELETE c
        """, url=url)

        for index, (chunk, vector) in enumerate(zip(chunks, chunk_vectors)):
            chunk_id = hashlib.sha256(f"{url}|{index}|{chunk}".encode()).hexdigest()
            session.run("""
                MATCH (s:Scheme {url: $url})
                CREATE (c:TextChunk {
                    id: $id,
                    text: $text,
                    chunk_index: $index,
                    source_url: $url,
                    embedding: $embedding
                })
                CREATE (s)-[:HAS_CHUNK]->(c)
            """, url=url, id=chunk_id, text=chunk, index=index,
                 embedding=vector)


def scrape_all():
    setup_neo4j()
    print(f"Fetching department listing: {START_URL}")
    listing = fetch_soup(START_URL)
    if listing is None:
        raise RuntimeError("The listing URL did not return an HTML page.")

    links = find_scheme_links(listing, START_URL)
    print(f"Discovered {len(links)} candidate links:")
    for label, url in list((label, url) for url, label in links.items())[:30]:
        print(f"  - {label}: {url}")
    if not links:
        print("No scheme links found. Inspect the listing HTML and adjust find_scheme_links().")
        return

    # Crawl only links discovered on the department listing. This avoids
    # accidentally spidering the entire government website.
    successful = 0
    failures = []
    for number, (url, label) in enumerate(links.items(), start=1):
        if number > MAX_PAGES:
            print(f"Stopping at MAX_PAGES={MAX_PAGES}.")
            break
        try:
            print(f"[{number}/{min(len(links), MAX_PAGES)}] {label} — {url}")
            soup = fetch_soup(url)
            if soup is None:
                print("  Skipped: not an HTML page (possibly a PDF/download).")
                continue
            title = page_title(soup, label)
            text = clean_text(soup)
            if len(text) < 80:
                print("  Skipped: very little readable text.")
                continue
            chunks = split_chunks(text)
            store_page(title, url, text, chunks)
            successful += 1
            print(f"  Stored {len(text)} characters in {len(chunks)} chunks.")
        except Exception as exc:
            failures.append((url, str(exc)))
            print(f"  ERROR: {exc}")
        time.sleep(REQUEST_DELAY)

    print(f"\nDone. Stored {successful} pages.")
    if failures:
        print(f"{len(failures)} failures:")
        for url, error in failures:
            print(f"  {url} :: {error}")


def retrieve(question, top_k=5):
    query_vector = openai_client.embeddings.create(
        model=EMBEDDING_MODEL, input=question
    ).data[0].embedding
    with driver.session() as session:
        result = session.run("""
            CALL db.index.vector.queryNodes(
                'scheme_chunk_embeddings', $top_k, $embedding
            )
            YIELD node, score
            MATCH (s:Scheme)-[:HAS_CHUNK]->(node)
            RETURN s.name AS scheme, s.url AS url,
                   node.text AS text, score
            ORDER BY score DESC
        """, top_k=top_k, embedding=query_vector)
        return list(result)


def graph_context(question, limit=10):
    """
    Lightweight graph lookup: identify scheme names mentioned in the question.
    The semantic vector results provide the main recall path.
    """
    with driver.session() as session:
        result = session.run("""
            MATCH (d:Department)-[:OFFERS_SCHEME]->(s:Scheme)
            WHERE toLower(s.name) CONTAINS toLower($q)
               OR toLower($q) CONTAINS toLower(s.name)
            RETURN s.name AS name, s.url AS url, s.text AS text
            LIMIT $limit
        """, q=question, limit=limit)
        return list(result)


def answer_question(question):
    vector_hits = retrieve(question)
    graph_hits = graph_context(question)
    context_parts = []
    seen = set()
    for hit in vector_hits:
        key = (hit["url"], hit["text"])
        if key not in seen:
            seen.add(key)
            context_parts.append(
                f"Scheme: {hit['scheme']}\nSource: {hit['url']}\n"
                f"Retrieved text:\n{hit['text']}"
            )
    for hit in graph_hits:
        key = (hit["url"], hit["text"][:300])
        if key not in seen:
            seen.add(key)
            context_parts.append(
                f"Graph match: {hit['name']}\nSource: {hit['url']}\n"
                f"Page text:\n{(hit['text'] or '')[:2500]}"
            )

    if not context_parts:
        print("No content found. Run `python agriculture_kg_rag.py scrape` first.")
        return

    context = "\n\n---\n\n".join(context_parts)
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions about Tamil Nadu's Agriculture - Farmers "
                    "Welfare Department schemes. Use only the supplied retrieved "
                    "government-page context. Do not invent eligibility, subsidy "
                    "amounts, deadlines, or application steps. If the source does "
                    "not provide an answer, say so clearly. Cite the relevant page "
                    "URL(s) in your answer. Distinguish historical information from "
                    "current rules when dates are shown."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nRetrieved context:\n{context}",
            },
        ],
    )
    print(response.choices[0].message.content)


def main():
    try:
        if len(sys.argv) > 1 and sys.argv[1].lower() == "scrape":
            scrape_all()
        elif len(sys.argv) > 1 and sys.argv[1].lower() == "ask":
            question = " ".join(sys.argv[2:]).strip()
            if not question:
                question = input("Ask a question about agriculture schemes: ").strip()
            answer_question(question)
        else:
            print("Tamil Nadu Agriculture Knowledge Graph RAG")
            print("1. Scrape and index scheme pages")
            print("2. Ask a question")
            choice = input("Choose 1 or 2: ").strip()
            if choice == "1":
                scrape_all()
            elif choice == "2":
                answer_question(input("Question: ").strip())
            else:
                print("Choose 1 or 2.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
