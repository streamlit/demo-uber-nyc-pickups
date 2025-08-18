"""
Atena — IA Didattica con Memoria PDF/CSV

✔️ FIX: niente crash quando stdin non è disponibile (TTY assente). CLI passa a modalità “scripted”.
✔️ Loop interattivo SOLO se TTY presente; altrimenti usa --once/--queries/--queries-file o prompt demo.
✔️ Mock/offline: risposte includono sempre "Fonte non disponibile (mock)" come richiesto.
✔️ Lazy import per LangChain/FAISS/OpenAIEmbeddings: l'app funziona anche senza.
✔️ Test inclusi (non modificare i già presenti; aggiunti nuovi test). Esegui: `python this_file.py --run-tests`

Esecuzione con Streamlit (se installato):
    streamlit run this_file.py

CLI interattiva (se TTY presente):
    python this_file.py --cli

CLI non interattiva (sandbox/headless):
    # singola domanda
    python this_file.py --cli --once "Spiega la legge di Lambert–Beer"
    # più domande via JSON
    python this_file.py --cli --queries '["Cos è la NMR 1H?","Che cos è un immunosensore competitivo?"]'
    # più domande via file (una per riga)
    python this_file.py --cli --queries-file prompts.txt

Varianti mock/offline (nessuna rete):
    export ATENA_DRYRUN=1
    # oppure
    python this_file.py --cli --base-url mock://local
"""

from __future__ import annotations

import os
import io
import sys
import json
import time
import tempfile
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import requests

# ---------------------------------------------------------------------------
# Rilevamento opzionale di Streamlit
# ---------------------------------------------------------------------------
try:
    import streamlit as st  # type: ignore
    STREAMLIT_AVAILABLE = True
except ModuleNotFoundError:
    st = None  # type: ignore
    STREAMLIT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config generale e prompt predefinito (tono didattico + rigore citazioni)
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "Sei Atena, un’assistente esperta in chimica analitica. Spieghi con metodo didattico, "
    "imponi rigore e precisione: ogni affermazione deve essere accompagnata da citazioni chiare, "
    "con riferimenti a fonti verificabili (autore, titolo, anno, DOI/URL). "
    "Se la fonte non è disponibile, dichiara esplicitamente 'Fonte non disponibile'."
)

# ---------------------------------------------------------------------------
# Utilità: modalità MOCK (nessuna rete) per ambienti offline
# ---------------------------------------------------------------------------

def is_mock_mode(base_url: str) -> bool:
    if os.getenv("ATENA_DRYRUN", "").strip().lower() in {"1", "true", "yes"}:
        return True
    return base_url.strip().lower().startswith("mock://")


def mock_completion(messages: List[Dict[str, str]]) -> str:
    """Genera una risposta deterministica per test/offline.
    Riporta l'ultima domanda e, se presente, un breve contesto.
    """
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), {"content": ""})
    content = last_user.get("content", "").strip()

    # Estrai eventuale sezione "Contesto rilevante dai documenti"
    ctx = ""
    if "Contesto rilevante" in content:
        parts = content.split("Contesto rilevante", 1)
        if len(parts) > 1:
            ctx = parts[1][:500].strip()

    reply = (
        "[MOCK OFFLINE] Risposta didattica.\n"
        "Fonte non disponibile (mock).\n\n"
        "• Domanda ricevuta:\n"
        f"{content[:1200]}\n\n"
        "• Metodo: ambiente offline; illustro principi e buone pratiche di citazione.\n"
        "• Citazioni: indica autore, titolo, anno, DOI/URL. In assenza, scrivi 'Fonte non disponibile'.\n"
    )
    if ctx:
        reply += "\n• Estratto contesto (parziale):\n" + ctx
    return reply

# ---------------------------------------------------------------------------
# Funzione di completamento chat (OpenAI-compatible). Supporta mock.
# ---------------------------------------------------------------------------

def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.4,
) -> str:
    if is_mock_mode(base_url):
        return mock_completion(messages)

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        raise RuntimeError(f"Risposta inattesa: {json.dumps(data)[:1000]}")

# ---------------------------------------------------------------------------
# RAG: costruzione opzionale del vector store (LangChain/FAISS se disponibili)
# ---------------------------------------------------------------------------

@dataclass
class RAGArtifacts:
    vectorstore: Any
    meta: Dict[str, Any]


def _lazy_import_langchain():
    """Importa i moduli LangChain/FAISS solo se presenti."""
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
        from langchain_community.document_loaders import PyPDFLoader, CSVLoader  # type: ignore
        from langchain_community.vectorstores import FAISS  # type: ignore
        from langchain_openai import OpenAIEmbeddings  # type: ignore
        return RecursiveCharacterTextSplitter, PyPDFLoader, CSVLoader, FAISS, OpenAIEmbeddings
    except ModuleNotFoundError:
        return None, None, None, None, None


def build_vectorstore_from_streamlit_uploads(
    uploaded_files: List[Any], api_key: str, base_url: str
) -> Optional[RAGArtifacts]:
    (
        RecursiveCharacterTextSplitter,
        PyPDFLoader,
        CSVLoader,
        FAISS,
        OpenAIEmbeddings,
    ) = _lazy_import_langchain()

    if not all([RecursiveCharacterTextSplitter, FAISS, OpenAIEmbeddings]):
        if STREAMLIT_AVAILABLE:
            st.info("Memoria documenti non attiva: moduli LangChain/FAISS non installati.")
        return None

    docs = []
    temp_paths = []
    for f in uploaded_files:
        # Salva su file temporaneo perché molti loader richiedono un path
        suffix = ".pdf" if (getattr(f, "type", "").endswith("pdf")) else ".csv"
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tf.write(f.read())
        tf.flush()
        tf.close()
        temp_paths.append(tf.name)

        if suffix == ".pdf" and PyPDFLoader is not None:
            loader = PyPDFLoader(tf.name)
            docs.extend(loader.load())
        elif suffix == ".csv" and CSVLoader is not None:
            loader = CSVLoader(tf.name)
            docs.extend(loader.load())

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    # Embeddings — consentire mock offline: in tal caso saltiamo embeddings reali
    if is_mock_mode(base_url):
        # Vectorstore fittizio: indicizzazione minimale in RAM
        class ToyVS:
            def __init__(self, texts):
                self.texts = texts

            def as_retriever(self, search_kwargs=None):
                k = (search_kwargs or {}).get("k", 3)
                class R:
                    def __init__(self, texts, k):
                        self.texts = texts
                        self.k = k
                    def get_relevant_documents(self, q):
                        qs = set(q.lower().split())
                        scored = []
                        for t in self.texts:
                            s = len(qs.intersection(set(t.page_content.lower().split())))
                            scored.append((s, t))
                        scored.sort(key=lambda x: -x[0])
                        return [t for _, t in scored[: self.k]]
                return R(self.texts, k)
        vectorstore = ToyVS(split_docs)
        return RAGArtifacts(vectorstore=vectorstore, meta={"backend": "toy"})

    # Embeddings reali
    embeddings = OpenAIEmbeddings(api_key=api_key, base_url=base_url)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return RAGArtifacts(vectorstore=vectorstore, meta={"backend": "faiss"})

# ---------------------------------------------------------------------------
# App Streamlit (se disponibile)
# ---------------------------------------------------------------------------

def run_streamlit_app():
    st.set_page_config(page_title="Atena IA Didattica", page_icon="📚", layout="wide")
    st.title("📚 Atena — Intelligenza Artificiale Didattica con Memoria PDF/CSV")

    with st.sidebar:
        st.header("⚙️ Impostazioni")
        st.caption(
            "Compatibile con API tipo OpenAI. Collegabile a LM Studio o Ollama.\n"
            "Per test offline spunta 'Modalità test offline' oppure usa base_url mock://local."
        )

        base_url: str = st.text_input(
            "Base URL",
            value=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        offline = st.checkbox("Modalità test offline (mock)", value=False)
        if offline:
            base_url = "mock://local"

        api_key: str = st.text_input(
            "API Key (se richiesta)",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
        )

        model: str = st.text_input("Modello", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

        system_prompt: str = st.text_area(
            "System prompt",
            value=DEFAULT_SYSTEM_PROMPT,
            height=180,
        )

        temperature: float = st.slider("Creatività (temperature)", 0.0, 1.5, 0.4, 0.1)

        st.divider()
        clear_btn = st.button("🧹 Azzera conversazione")

        # File uploader per memoria
        uploaded_files = st.file_uploader(
            "Carica PDF o CSV",
            type=["pdf", "csv"],
            accept_multiple_files=True,
        )

    if "messages" not in st.session_state or clear_btn:
        st.session_state["messages"] = []

    if "rag" not in st.session_state:
        st.session_state["rag"] = None

    # Caricamento e indicizzazione documenti
    if uploaded_files:
        rag = build_vectorstore_from_streamlit_uploads(uploaded_files, api_key, base_url)
        if rag:
            st.session_state["rag"] = rag
            st.success("📚 Documenti caricati e indicizzati con successo!")

    # Mostra cronologia
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"]) 

    # Input utente
    user_input = st.chat_input("Scrivi qui il tuo messaggio…")

    if user_input:
        history: List[Dict[str, str]] = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        history.extend(st.session_state["messages"])

        # Recupero documenti pertinenti (se RAG attivo)
        rel_docs = []
        if st.session_state.get("rag"):
            retriever = st.session_state["rag"].vectorstore.as_retriever(search_kwargs={"k": 3})
            rel_docs = retriever.get_relevant_documents(user_input)
            context = "\n\n".join([d.page_content for d in rel_docs])
            if rel_docs:
                user_input = (
                    f"Domanda: {user_input}\n\n"
                    f"Contesto rilevante dai documenti (usare come citazione):\n{context}\n\n"
                    f"Nota: cita sempre la fonte indicando nome file e pagina/riga se possibile."
                )

        history.append({"role": "user", "content": user_input})

        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            with st.chat_message("assistant"):
                reply = chat_completion(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    messages=history,
                    temperature=temperature,
                )
                st.markdown(reply)
                if rel_docs:
                    st.caption("📖 Fonti utilizzate:")
                    for i, d in enumerate(rel_docs, 1):
                        st.caption(f"{i}. Estratto: {d.page_content[:200]}… (meta: {d.metadata})")
            st.session_state["messages"].append({"role": "assistant", "content": reply})
        except Exception as e:
            msg = f"❌ Errore: {e}"
            with st.chat_message("assistant"):
                st.error(msg)
            st.session_state["messages"].append({"role": "assistant", "content": msg})

    # Esportazione conversazione
    if st.session_state["messages"]:
        md_lines = ["# Trascrizione chat\n"]
        for m in st.session_state["messages"]:
            role = "Utente" if m["role"] == "user" else "Atena"
            md_lines.append(f"**{role}:**\n\n{m['content']}\n")
        md_text = "\n".join(md_lines)
        st.download_button(
            label="💾 Scarica conversazione (.md)",
            data=md_text,
            file_name="chat_Atena.md",
            mime="text/markdown",
        )

# ---------------------------------------------------------------------------
# CLI utilities (robusti a TTY assente)
# ---------------------------------------------------------------------------

def _supports_interactive_stdin() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def _build_history(system_prompt: str, prior: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    hist = [{"role": "system", "content": system_prompt}]
    if prior:
        hist.extend(prior)
    return hist


def _answer_single_turn(
    question: str,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    temperature: float = 0.4,
    context_docs: Optional[List[str]] = None,
) -> str:
    user_msg = question
    if context_docs:
        ctx = "\n\n".join(context_docs)
        user_msg = (
            f"Domanda: {question}\n\n"
            f"Contesto rilevante dai documenti (usare come citazione):\n{ctx}\n\n"
            f"Nota: cita sempre la fonte indicando nome file e pagina/riga se possibile."
        )
    history = _build_history(system_prompt)
    history.append({"role": "user", "content": user_msg})
    return chat_completion(base_url, api_key, model, history, temperature)


def run_cli_interactive(base_url: str, api_key: str, model: str, system_prompt: str, temperature: float):
    print("== Atena (CLI interattiva) ==")
    print("Suggerimento: Ctrl+C per uscire.")
    history: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    while True:
        try:
            user_inp = input("Tu: ").strip()
        except (EOFError, OSError):
            # ambiente senza stdin -> fallback scripted
            print("\n[!] Input interattivo non disponibile. Passo alla modalità non interattiva (scripted).")
            return False  # indica al caller di fare fallback
        if not user_inp:
            continue
        if user_inp.lower() in {"/exit", ":q", "quit"}:
            print("Arrivederci.")
            return True

        history.append({"role": "user", "content": user_inp})
        try:
            reply = chat_completion(base_url, api_key, model, history, temperature)
            print("Atena:", reply, "\n")
            history.append({"role": "assistant", "content": reply})
        except Exception as e:
            print("Errore:", e)
            time.sleep(0.5)


def run_cli_scripted(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    temperature: float,
    queries: List[str],
):
    print("== Atena (CLI scripted — non interattiva) ==")
    for i, q in enumerate(queries, 1):
        print(f"\n[{i}] Tu:", q)
        try:
            reply = _answer_single_turn(q, base_url, api_key, model, system_prompt, temperature)
            print("Atena:", reply)
        except Exception as e:
            print("Errore:", e)


def run_cli_entry(args: argparse.Namespace):
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL", "mock://local")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    model = args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    system_prompt = DEFAULT_SYSTEM_PROMPT
    temperature = args.temperature

    # Determina se interattivo è possibile
    interactive_possible = _supports_interactive_stdin() or args.force_interactive

    # Raccolta query per scripted mode
    scripted_queries: List[str] = []
    if args.once:
        scripted_queries = [args.once]
    elif args.queries:
        try:
            scripted_queries = json.loads(args.queries)
            assert isinstance(scripted_queries, list)
        except Exception:
            print("⚠️  --queries deve essere una lista JSON di stringhe.")
            scripted_queries = []
    elif args.queries_file:
        try:
            with open(args.queries_file, "r", encoding="utf-8") as fh:
                scripted_queries = [ln.strip() for ln in fh if ln.strip()]
        except Exception as e:
            print(f"⚠️  Impossibile leggere {args.queries_file}: {e}")

    # Default di sicurezza per ambienti sandbox: se niente input, usa una demo
    if not interactive_possible and not scripted_queries:
        scripted_queries = [
            "Spiega la legge di Lambert–Beer.",
            "Cos’è un immunosensore elettrochimico competitivo al cortisolo?",
        ]

    if interactive_possible and not args.force_scripted:
        ok = run_cli_interactive(base_url, api_key, model, system_prompt, temperature)
        if ok is True:
            return
        # altrimenti fallback a scripted

    run_cli_scripted(base_url, api_key, model, system_prompt, temperature, scripted_queries)

# ---------------------------------------------------------------------------
# Test di base (esistenti) + aggiunte
# ---------------------------------------------------------------------------

def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def run_tests():
    print("Esecuzione test...")

    # Test 1: rilevamento Streamlit
    _assert(isinstance(STREAMLIT_AVAILABLE, bool), "STREAMLIT_AVAILABLE deve essere bool")

    # Test 2: mock completion
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "Domanda: prova mock.\n\nContesto rilevante dai documenti: lorem ipsum."},
    ]
    out = mock_completion(messages)
    _assert("[MOCK OFFLINE]" in out, "mock_completion deve indicare mock")
    _assert("prova mock" in out, "mock_completion deve riepilogare la domanda")
    _assert("Fonte non disponibile (mock)" in out, "mock_completion deve segnalare fonte mock")

    # Test 3: chat_completion in mock mode
    reply = chat_completion("mock://local", "", "dummy", messages, 0.4)
    _assert("[MOCK OFFLINE]" in reply, "chat_completion mock deve restituire risposta mock")

    # Test 4: RAG toy retriever (senza langchain/embeddings reali)
    os.environ["ATENA_DRYRUN"] = "1"
    class D:  # documento minimale compatibile
        def __init__(self, text):
            self.page_content = text
            self.metadata = {"source": "test"}
    texts = [D("La legge di Lambert-Beer: A = ε·l·c"), D("Spettroscopia UV-Vis"), D("NMR 1H: spostamenti chimici")] 
    class _ToyVS:
        def __init__(self, docs):
            self.texts = docs
        def as_retriever(self, search_kwargs=None):
            k = (search_kwargs or {}).get("k", 2)
            class R:
                def __init__(self, docs, k):
                    self.docs, self.k = docs, k
                def get_relevant_documents(self, q):
                    qs = set(q.lower().split())
                    scored = []
                    for t in self.docs:
                        s = len(qs.intersection(set(t.page_content.lower().split())))
                        scored.append((s, t))
                    scored.sort(key=lambda x: -x[0])
                    return [t for _, t in scored[: self.k]]
            return R(self.texts, k)
    vs = _ToyVS(texts)
    retr = vs.as_retriever({"k": 2})
    res = retr.get_relevant_documents("Lambert-Beer assorbanza")
    _assert(len(res) >= 1, "La retrieval toy deve restituire almeno 1 documento")

    # Test 5 (nuovo): CLI scripted non interattivo non deve lanciare eccezioni
    try:
        run_cli_scripted(
            base_url="mock://local",
            api_key="",
            model="dummy",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            temperature=0.4,
            queries=["Test domanda 1", "Test domanda 2"],
        )
    except Exception as e:
        raise AssertionError(f"CLI scripted non dovrebbe fallire: {e}")

    # Test 6 (nuovo): _answer_single_turn produce stringa non vuota
    ans = _answer_single_turn(
        "Spiega NMR 1H",
        base_url="mock://local",
        api_key="",
        model="dummy",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        temperature=0.4,
    )
    _assert(isinstance(ans, str) and len(ans) > 0, "_answer_single_turn deve restituire testo")

    print("✅ Tutti i test superati!")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Atena — CLI/Streamlit")
    p.add_argument("--run-tests", action="store_true", help="Esegue i test e termina.")
    p.add_argument("--cli", action="store_true", help="Avvia modalità CLI (interattiva se TTY).")
    p.add_argument("--force-interactive", action="store_true", help="Forza loop interattivo anche senza TTY (sconsigliato).")
    p.add_argument("--force-scripted", action="store_true", help="Forza modalità scripted (non interattiva).")
    p.add_argument("--once", type=str, default="", help="Esegue una singola domanda.")
    p.add_argument("--queries", type=str, default="", help="Lista JSON di domande per modalità scripted.")
    p.add_argument("--queries-file", type=str, default="", help="File con una domanda per riga.")
    p.add_argument("--base-url", type=str, default="", help="Override base URL API.")
    p.add_argument("--api-key", type=str, default="", help="Override API key.")
    p.add_argument("--model", type=str, default="", help="Override modello.")
    p.add_argument("--temperature", type=float, default=0.4, help="Creatività (temperature).")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])

    if args.run_tests:
        run_tests()
        sys.exit(0)

    if args.cli or not STREAMLIT_AVAILABLE:
        run_cli_entry(args)
    else:
        run_streamlit_app()
