"""核心逻辑单元测试：切块、向量化降级、引用校验。"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from rag.chunker import Chunk, Paragraph, build_chunks
from rag.citations import extract_citations, verify_answer
from rag.embeddings import NgramTfidfEmbedder
from rag.textutil import containment


class TestChunker(unittest.TestCase):
    def setUp(self):
        self.paragraphs = [
            Paragraph("p1", "doc", "标题甲", "第一段内容，用于测试切块逻辑。" * 3),
            Paragraph("p2", "doc", "标题甲", "第二段内容，同样用于测试。" * 3),
            Paragraph("p3", "doc", "标题乙", "第三段属于另一个小节。" * 3),
        ]

    def test_chunk_size_respected(self):
        chunks = build_chunks(self.paragraphs, chunk_size=120, overlap=20)
        self.assertTrue(chunks)
        for chunk in chunks:
            # 允许单个段落自身超长时按句切分，此处每段约 50 字，不应超限
            self.assertLessEqual(len(chunk.text), 120)

    def test_heading_boundary(self):
        chunks = build_chunks(self.paragraphs, chunk_size=400, overlap=40)
        for chunk in chunks:
            headings = {self.paragraphs[i].heading for i in range(chunk.start_para, chunk.end_para + 1)}
            self.assertEqual(len(headings), 1)

    def test_overlap_carries_context(self):
        chunks = build_chunks(self.paragraphs, chunk_size=110, overlap=60)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(set(chunks[0].para_ids) & set(chunks[1].para_ids))

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            build_chunks(self.paragraphs, chunk_size=100, overlap=100)


class TestFallbackEmbedder(unittest.TestCase):
    def test_similar_text_scores_higher(self):
        texts = [
            "快速排序通过基准分区递归处理左右子数组。",
            "红烧肉需要先用小火慢炖两个小时。",
            "归并排序把数组对半切分后合并两个有序序列。",
        ]
        embedder = NgramTfidfEmbedder(dim=512)
        embedder.fit(texts)
        matrix = embedder.encode(texts)
        query = embedder.encode(["排序算法的分治思路"], is_query=True)[0]
        scores = matrix @ query
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[2], scores[1])

    def test_deterministic_across_instances(self):
        texts = ["中文哈希向量需要稳定", "不同进程的哈希种子可能不同"]
        first = NgramTfidfEmbedder(dim=256)
        first.fit(texts)
        second = NgramTfidfEmbedder(dim=256)
        second.fit(texts)
        self.assertTrue(np.allclose(first.encode(texts), second.encode(texts)))


class TestCitationVerification(unittest.TestCase):
    CHUNKS = {
        "c0001": "快速排序平均 O(n log n)，最坏 O(n^2)，最坏情况出现在基准选择极不均衡时。",
        "c0002": "归并排序在三种情况下都是 O(n log n)，但需要 O(n) 额外空间。",
    }

    def test_supported_claim(self):
        answer = "- 结论1：快速排序最坏情况的时间复杂度是 O(n^2) [c0001]"
        result = verify_answer(answer, self.CHUNKS, threshold=0.55)
        self.assertEqual(result.total, 1)
        self.assertTrue(result.claims[0].supported)

    def test_unsupported_claim_detected(self):
        """负例：引用编号正确但内容与引用无关，必须被判为未支撑。"""
        answer = "- 结论1：快速排序在最好情况下只需要一次比较 [c0001]"
        result = verify_answer(answer, self.CHUNKS, threshold=0.55)
        self.assertFalse(result.claims[0].supported)

    def test_unknown_citation_id(self):
        answer = "- 结论1：归并排序需要 O(n) 额外空间 [c9999]"
        result = verify_answer(answer, self.CHUNKS, threshold=0.55)
        self.assertEqual(result.unknown_ids, ["c9999"])
        self.assertFalse(result.claims[0].supported)

    def test_extract_citations_dedup(self):
        self.assertEqual(extract_citations("[c0001] 与 [c0001] 和 [c0002]"), ["c0001", "c0002"])

    def test_containment_metric(self):
        self.assertGreater(containment("归并排序需要额外空间", "归并排序需要 O(n) 额外空间。"), 0.8)


if __name__ == "__main__":
    unittest.main()
