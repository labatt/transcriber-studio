# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
import unittest

from transcriber_studio.glossary import merge_glossaries, merge_speakers, merge_terms


class GlossaryMergeTests(unittest.TestCase):
    def test_merge_terms_unions_variants_and_prefers_common_canonical(self):
        merged = merge_terms(
            [
                [
                    {
                        "canonical": "GrowthMark",
                        "variants": ["growth mark"],
                        "type": "product",
                    }
                ],
                [
                    {
                        "canonical": "growth mark",
                        "variants": ["growth market"],
                        "type": "product",
                    },
                    {
                        "canonical": "GrowthMark",
                        "variants": ["growth-market"],
                        "type": "product",
                    },
                ],
            ]
        )
        by_name = {item["canonical"].lower(): item for item in merged}
        self.assertIn("growthmark", by_name)
        entry = by_name["growthmark"]
        self.assertEqual(entry["canonical"], "GrowthMark")
        self.assertEqual(
            sorted(entry["variants"]),
            ["growth mark", "growth market", "growth-market"],
        )

    def test_merge_speakers_prefers_named_high_confidence_and_keeps_raw_intro(self):
        merged = merge_speakers(
            [
                [
                    {
                        "label": "Clinton",
                        "name": None,
                        "role": "",
                        "confidence": "low",
                        "raw_intro": "Flynn Justin",
                    }
                ],
                [
                    {
                        "label": "Clinton",
                        "name": "Clinton Smith",
                        "role": "PM",
                        "confidence": "high",
                        "raw_intro": "",
                    }
                ],
            ]
        )
        self.assertEqual(len(merged), 1)
        clinton = merged[0]
        self.assertEqual(clinton["name"], "Clinton Smith")
        self.assertEqual(clinton["confidence"], "high")
        self.assertEqual(clinton["role"], "PM")
        self.assertEqual(clinton["raw_intro"], "Flynn Justin")

    def test_merge_glossaries_combines_both_lists(self):
        merged = merge_glossaries(
            [
                {
                    "speakers": [
                        {
                            "label": "Greg",
                            "name": "Gregory Jackson",
                            "role": "",
                            "confidence": "high",
                            "raw_intro": "",
                        }
                    ],
                    "terms": [
                        {
                            "canonical": "Silver Tsunami",
                            "variants": ["Solar Tsunami"],
                            "type": "concept",
                        }
                    ],
                },
                {
                    "speakers": [],
                    "terms": [
                        {
                            "canonical": "silver tsunami",
                            "variants": ["solar tsunami"],
                            "type": "concept",
                        }
                    ],
                },
            ]
        )
        self.assertEqual(len(merged["speakers"]), 1)
        self.assertEqual(merged["speakers"][0]["name"], "Gregory Jackson")
        self.assertEqual(len(merged["terms"]), 1)
        self.assertEqual(merged["terms"][0]["canonical"], "Silver Tsunami")
        self.assertIn("Solar Tsunami", merged["terms"][0]["variants"])


    def test_glossary_paths_differ_per_recording(self):
        """Two meetings with the same name and date must not share a glossary."""
        from transcriber_studio.config import Settings
        from transcriber_studio.glossary import glossary_path, legacy_glossary_path
        from transcriber_studio.models import Recording, Source, TranscriptResult

        settings = Settings()
        first = TranscriptResult(
            Recording(Source.PLAUD, "plaud-id-aaa", "Strategy Meeting", date="2026-05-29")
        )
        second = TranscriptResult(
            Recording(Source.PLAUD, "plaud-id-bbb", "Strategy Meeting", date="2026-05-29")
        )

        self.assertNotEqual(glossary_path(settings, first), glossary_path(settings, second))
        # The old scheme keyed on the rendered name alone, which is exactly how
        # they used to collide.
        self.assertEqual(
            legacy_glossary_path(settings, first), legacy_glossary_path(settings, second)
        )
        # Same recording, same file, every time.
        self.assertEqual(glossary_path(settings, first), glossary_path(settings, first))

if __name__ == "__main__":
    unittest.main()
