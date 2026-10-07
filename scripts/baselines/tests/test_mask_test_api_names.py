import contextlib
import io
import sys
import unittest

from pathlib import Path

BASELINES_DIR = Path(__file__).resolve().parents[1]
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

import mask_test_api_names as masker


class OfficialApiNameMaskingTest(unittest.TestCase):
    def build_index(self, records):
        entries, warnings = masker.build_api_name_entries(records)
        return masker.build_first_token_index(entries), entries, warnings

    def test_normalization_contract(self):
        self.assertEqual(masker.normalize_text(" A&amp;Ｂ—c!  42 "), "a b c 42")
        self.assertEqual(masker.normalize_text("Google\tMaps\nAPI"), "google maps api")

    def test_does_not_match_partial_tokens(self):
        index, _, _ = self.build_index([masker.CatalogRecord("Box")])
        text = "sandbox checkbox Box boxer"
        matches = masker.locate_matches(text, index, placeholder=masker.PLACEHOLDER)
        self.assertEqual([match.matched_surface_form for match in matches], ["Box"])
        self.assertEqual(
            masker.apply_matches(text, matches, placeholder=masker.PLACEHOLDER),
            "sandbox checkbox <API_NAME> boxer",
        )

    def test_html_surface_form_and_source_span_are_logged(self):
        index, _, _ = self.build_index([masker.CatalogRecord("Sales Invoices")])
        text = "Sales &amp; Invoices are here"
        matches = masker.locate_matches(text, index, placeholder=masker.PLACEHOLDER)
        self.assertEqual(len(matches), 1)
        match = matches[0]
        self.assertEqual(match.matched_surface_form, "Sales &amp; Invoices")
        self.assertEqual(match.normalized_matched_surface_form, "sales invoices")
        self.assertEqual(match.start_char, 0)
        self.assertEqual(match.end_char, len("Sales &amp; Invoices"))

    def test_no_semantic_completion_or_api_suffix_removal(self):
        index, _, _ = self.build_index(
            [
                masker.CatalogRecord("Amazon Product Advertising"),
                masker.CatalogRecord("Twitter API"),
            ]
        )
        text = "Amazon API and Twitter are mentioned. Twitter API is exact."
        matches = masker.locate_matches(text, index, placeholder=masker.PLACEHOLDER)
        self.assertEqual([match.matched_surface_form for match in matches], ["Twitter API"])

    def test_duplicate_normalized_api_names_are_one_mask_target(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            index, entries, warnings = self.build_index(
                [
                    masker.CatalogRecord("Box.com", "api-1"),
                    masker.CatalogRecord("Box com", "api-2"),
                ]
            )
        self.assertEqual(len(entries), 1)
        self.assertEqual(len(warnings), 1)
        self.assertIn("duplicate normalized official API names", stderr.getvalue())

        text = "Use Box.com today"
        matches = masker.locate_matches(text, index, placeholder=masker.PLACEHOLDER)
        self.assertEqual(len(matches), 1)
        row = masker.match_to_log_row(
            masker.TextRecord(
                row_index=3,
                text_id="sample-3",
                text=text,
                location_kind="json_list_dict",
                key=3,
                text_field="description",
            ),
            matches[0],
        )
        self.assertEqual(row["official_api_id"], "api-1;api-2")
        self.assertEqual(row["official_api_name"], "Box.com;Box com")
        self.assertEqual(row["normalized_api_name"], "box com")

    def test_placeholder_is_protected_on_rerun(self):
        index, _, _ = self.build_index([masker.CatalogRecord("API Name")])
        text = "Already <API_NAME> should remain, but API Name here is masked."
        matches = masker.locate_matches(text, index, placeholder=masker.PLACEHOLDER)
        self.assertEqual([match.matched_surface_form for match in matches], ["API Name"])


if __name__ == "__main__":
    unittest.main()
