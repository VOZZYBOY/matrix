import json
import os

import pytest

import sys, os, pathlib
# Убеждаемся, что корень проекта в sys.path для импорта rag_setup
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import rag_setup


class DummyChromaClient:
    """Простой объект-заглушка для имитации chromadb.ClientAPI."""

    def __init__(self):
        self.deleted_collection = None

    def delete_collection(self, name: str):
        """Запоминает, какую коллекцию "удалили"."""
        self.deleted_collection = name


class DummyEmbeddings:
    """Заглушка для объекта эмбеддингов GigaChatEmbeddings."""

    pass


def test_load_service_details(tmp_path):
    """Проверяем корректность загрузки и нормализации ключей."""
    data = [
        {
            "serviceName": "МРТ коленного сустава",
            "categoryName": "МРТ",
            "indications": ["Боль", "Травма"],
            "contraindications": ["Металлические импланты"],
        }
    ]
    file_path = tmp_path / "services.json"
    file_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    details_map = rag_setup.load_service_details(str(file_path))

    key = (
        rag_setup.normalize_text("МРТ коленного сустава", keep_spaces=True),
        rag_setup.normalize_text("МРТ", keep_spaces=True),
    )
    assert key in details_map
    assert details_map[key]["indications"] == ["Боль", "Травма"]


def test_preprocess_for_rag_v2_enrich(tmp_path):
    """Убеждаемся, что preprocess_for_rag_v2 добавляет показания/противопоказания."""
    service_key = (
        rag_setup.normalize_text("МРТ коленного сустава", keep_spaces=True),
        rag_setup.normalize_text("МРТ", keep_spaces=True),
    )
    service_details_map = {
        service_key: {
            "indications": ["Боль"],
            "contraindications": ["Металл"],
        }
    }

    raw_data = [
        {
            "serviceId": "srv1",
            "serviceName": "МРТ коленного сустава",
            "categoryName": "МРТ",
            "serviceDescription": "",
        }
    ]

    docs = rag_setup.preprocess_for_rag_v2(raw_data, service_details_map)

    assert len(docs) == 1
    page = docs[0].page_content
    assert "Показания:" in page
    assert "Противопоказания:" in page


def test_build_collection_name_edge_cases():
    """Проверяем разные варианты формирования названий коллекций."""
    assert rag_setup.build_collection_name("tenant1_chain1") == "tenant_tenant1_chain1"
    assert rag_setup.build_collection_name("tenant-1") == "tenant_tenant-1_default"

    long_name = rag_setup.build_collection_name("t" * 80)
    assert len(long_name) <= 63

    special = rag_setup.build_collection_name("tenant@1_chain#")
    assert special == "tenant_tenant_1_chain"


def test_reindex_tenant_no_docs(monkeypatch, tmp_path):
    """Сценарий отсутствия документов: коллекция удаляется, карты очищаются."""
    # Отключаем tenant_config_manager, чтобы не пытаться загружать clinic_info
    monkeypatch.setattr(rag_setup, "tenant_config_manager", None)

    client = DummyChromaClient()
    embeddings = DummyEmbeddings()

    bm25_map = {"tenantx": "old"}
    docs_map = {"tenantx": ["dummy"]}
    raw_data_map = {"tenantx": ["raw"]}

    success = rag_setup.reindex_tenant_specific_data(
        tenant_id="tenantx",
        chroma_client=client,
        embeddings_object=embeddings,
        bm25_retrievers_map=bm25_map,
        tenant_documents_map=docs_map,
        tenant_raw_data_map=raw_data_map,
        service_details_map={},
        base_data_dir=str(tmp_path),  # каталог, где явно нет файла tenantx.json
        chunk_size=1000,
        chunk_overlap=0,
        search_k=5,
    )

    assert success is True
    assert "tenantx" not in docs_map
    assert client.deleted_collection == rag_setup.build_collection_name("tenantx")
