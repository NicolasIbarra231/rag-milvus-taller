"""Segmentación portada sin cambios desde el notebook (Fase 2, Parte 3).

Las funciones devuelven *offsets*: el texto de cada chunk se corta después del
original, por eso se cumple siempre `texto[char_start:char_end] == chunk`.
Estrategia elegida por el equipo: `struct256o32` (oraciones completas hasta
W=256 tokens, solapamiento máximo O=32).
"""
from __future__ import annotations

import bisect
import hashlib
import re


class Tokens:
    """Posiciones de los tokens de un texto, para contar tokens en cualquier tramo."""

    def __init__(self, texto, tokenizer):
        self.spans = tokenizer(texto)  # [(inicio, fin), ...] en caracteres
        self.inicios = [s for s, _ in self.spans]

    def contar(self, s, e):
        return bisect.bisect_left(self.inicios, e) - bisect.bisect_left(self.inicios, s)

    def indices(self, s, e):
        return bisect.bisect_left(self.inicios, s), bisect.bisect_left(self.inicios, e)


def crear_tokenizer(nombre):
    """Devuelve una función texto -> [(inicio, fin), ...]."""
    if nombre == "regex":  # sin descargas, pero cuenta MENOS tokens que el real
        patron = re.compile(r"\w+|[^\w\s]")
        return lambda t: [m.span() for m in patron.finditer(t)]
    from tokenizers import Tokenizer

    tk = Tokenizer.from_pretrained(nombre)

    def tokenizar(t):
        spans = []
        for s, e in tk.encode(t, add_special_tokens=False).offsets:
            while s < e and t[s].isspace():  # algunos tokens incluyen el espacio previo
                s += 1
            if s < e:
                spans.append((s, e))
        return spans

    return tokenizar


def recortar(t, s, e):
    """Quita espacios en los bordes de [s, e) sin tocar el interior."""
    while s < e and t[s].isspace():
        s += 1
    while e > s and t[e - 1].isspace():
        e -= 1
    return s, e


def ajustar_a_palabras(t, s, e):
    """Amplía [s, e) para no partir una palabra ('Bogo|tá')."""
    while 0 < s < len(t) and t[s - 1].isalnum() and t[s].isalnum():
        s -= 1
    while 0 < e < len(t) and t[e - 1].isalnum() and t[e].isalnum():
        e += 1
    return s, e


FIN = re.compile(r"[.!?…]+[\"'»”’)\]]*|\n")


def oraciones(t):
    """Oraciones y líneas. No parte '3.5', 'eltiempo.com' ni 'aprox. diez'."""
    cortes = [0]
    for m in FIN.finditer(t):
        e = m.end()
        if m.group() != "\n":
            if e < len(t) and not t[e].isspace():
                continue
            siguiente = t[e:].lstrip()[:1]
            if siguiente.islower():
                continue
        cortes.append(e)
    cortes.append(len(t))
    return [r for a, b in zip(cortes, cortes[1:]) if (r := recortar(t, a, b))[0] < r[1]]


def ventanas(t, tok, i0, i1, W, O):
    """Ventanas de W tokens que avanzan W-O, sobre los tokens [i0, i1)."""
    salida, i = [], i0
    while True:
        j = min(i + W, i1)
        salida.append(recortar(t, *ajustar_a_palabras(t, tok.spans[i][0], tok.spans[j - 1][1])))
        if j >= i1:
            return salida
        i += W - O


def chunk_fixed(t, tok, W, O):
    """Estrategia A: ventana fija."""
    if len(tok.spans) <= W:
        return [recortar(t, 0, len(t))]
    return ventanas(t, tok, 0, len(tok.spans), W, O)


def chunk_struct(t, tok, W, O):
    """Estrategia B: oraciones completas hasta W tokens; solapamiento con las
    últimas oraciones del chunk anterior (máx. O tokens)."""
    if len(tok.spans) <= W:
        return [recortar(t, 0, len(t))]
    unidades = []
    for s, e in oraciones(t):
        if tok.contar(s, e) > W:  # oración gigante: se parte en ventanas
            unidades += ventanas(t, tok, *tok.indices(s, e), W, O)
        elif tok.contar(s, e) > 0:
            unidades.append((s, e))
    chunks, actual, n_actual = [], [], 0
    for u in unidades:
        n = tok.contar(*u)
        if actual and n_actual + n > W:  # no cabe: se cierra el chunk
            chunks.append((actual[0][0], actual[-1][1]))
            solape, n_solape = [], 0
            for v in reversed(actual):  # oraciones finales que quepan en O
                if n_solape + tok.contar(*v) > O:
                    break
                solape.insert(0, v)
                n_solape += tok.contar(*v)
            actual, n_actual = (solape, n_solape) if n_solape + n <= W else ([], 0)
        actual.append(u)
        n_actual += n
    if actual:
        chunks.append((actual[0][0], actual[-1][1]))
    return chunks


ESTRATEGIAS = {"fixed": chunk_fixed, "struct": chunk_struct}


def id_estable(doc_id, estrategia, s, e):
    """Hash de documento + configuración + offsets: reindexar no duplica."""
    return hashlib.sha256(f"{doc_id}|{estrategia}|{s}|{e}".encode()).hexdigest()[:32]


def hash_contenido(texto):
    """Mismo contenido aunque cambien espacios o saltos de línea."""
    return hashlib.sha256(" ".join(texto.split()).encode()).hexdigest()


def nombre_config(estrategia: str, max_tokens: int, overlap: int) -> str:
    """'struct', 256, 32 -> 'struct256o32' (mismo nombre que en el notebook)."""
    return f"{estrategia}{max_tokens}o{overlap}"
