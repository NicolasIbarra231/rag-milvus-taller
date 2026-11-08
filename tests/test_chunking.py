"""Propiedades del chunking (las mismas 'pruebas rápidas' del notebook, Fase 2)."""
from app.chunking import (Tokens, chunk_fixed, chunk_struct, crear_tokenizer, hash_contenido,
                          id_estable, nombre_config, oraciones)

TOKENIZAR = crear_tokenizer("regex")
TEXTO = " ".join(f"Oración número {i} sobre el precio del dólar en Bogotá." for i in range(80))


def test_offsets_bordes_y_cobertura():
    tok = Tokens(TEXTO, TOKENIZAR)
    for nombre, fn in {"fixed": chunk_fixed, "struct": chunk_struct}.items():
        spans = fn(TEXTO, tok, 40, 8)
        assert len(spans) > 1, nombre
        for s, e in spans:
            assert TEXTO[s:e] == TEXTO[s:e].strip(), f"{nombre}: espacios en el borde"
            assert not (s > 0 and TEXTO[s - 1].isalnum() and TEXTO[s].isalnum()), f"{nombre}: parte una palabra"
        assert all(any(s <= a and b <= e for s, e in spans) for a, b in tok.spans), f"{nombre}: texto sin cubrir"


def test_struct_termina_en_oracion_y_respeta_w():
    tok = Tokens(TEXTO, TOKENIZAR)
    spans = chunk_struct(TEXTO, tok, 40, 8)
    assert all(TEXTO[e - 1] == "." for _, e in spans)
    assert all(tok.contar(s, e) <= 40 for s, e in spans)


def test_documento_corto_es_un_solo_chunk():
    corto = "Post corto. Nada más."
    assert chunk_fixed(corto, Tokens(corto, TOKENIZAR), 40, 8) == [(0, len(corto))]
    assert chunk_struct(corto, Tokens(corto, TOKENIZAR), 40, 8) == [(0, len(corto))]


def test_oraciones_no_parte_decimales_ni_dominios():
    o = "Subió 3.5 hoy. Ver eltiempo.com ya. Fueron aprox. diez."
    assert [o[s:e] for s, e in oraciones(o)] == ["Subió 3.5 hoy.", "Ver eltiempo.com ya.", "Fueron aprox. diez."]


def test_ids_estables_y_hash_de_contenido():
    cid = nombre_config("struct", 256, 32)
    assert cid == "struct256o32"
    assert id_estable("doc1", cid, 0, 10) == id_estable("doc1", cid, 0, 10)
    assert id_estable("doc1", cid, 0, 10) != id_estable("doc1", cid, 0, 11)
    assert hash_contenido("hola   mundo\n") == hash_contenido("hola mundo")
