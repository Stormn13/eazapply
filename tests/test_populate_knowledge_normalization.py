import json
import unittest
from unittest.mock import patch

import populate_knowledge as ingestion


def extraction_payload(record_type: str, status: str = "draft", title: str = "Example") -> dict:
    return {
        "records": [
            {
                "record_type": record_type,
                "record_id": "example-record",
                "title": title,
                "status": status,
            }
        ]
    }


class ExtractionNormalizationTests(unittest.TestCase):
    def parse(self, payload: dict) -> ingestion.CareerRecord:
        return ingestion.parse_json_payload(json.dumps(payload)).records[0]

    def test_confirmed_status_maps_to_active_and_preserves_raw_value(self) -> None:
        record = self.parse(extraction_payload("experience", "  CONFIRMED  "))

        self.assertEqual(record.status, "active")
        self.assertEqual(record.normalization_original_values["status"], ["  CONFIRMED  "])
        self.assertIn('"  CONFIRMED  "', ingestion.render_markdown(record))

    def test_competition_maps_to_achievement(self) -> None:
        record = self.parse(extraction_payload(" Competition "))

        self.assertEqual(record.record_type, "achievement")
        self.assertEqual(record.normalization_original_values["record_type"], [" Competition "])

    def test_award_maps_to_achievement(self) -> None:
        record = self.parse(extraction_payload("AWARD"))

        self.assertEqual(record.record_type, "achievement")

    def test_leadership_maps_to_experience(self) -> None:
        record = self.parse(extraction_payload("Leadership"))

        self.assertEqual(record.record_type, "experience")

    def test_position_of_responsibility_maps_to_experience(self) -> None:
        record = self.parse(extraction_payload("POSITION_OF_RESPONSIBILITY"))

        self.assertEqual(record.record_type, "experience")

    def test_club_leadership_position_maps_to_experience(self) -> None:
        payload = extraction_payload("position", title="Social Media Head")
        payload["records"][0]["fields"] = {"organization": "Robotics Club"}

        record = self.parse(payload)

        self.assertEqual(record.record_type, "experience")

    def test_non_leadership_volunteer_position_maps_to_activity(self) -> None:
        record = self.parse(extraction_payload("position", title="Volunteer Club Member"))

        self.assertEqual(record.record_type, "activity")

    def test_canonical_record_types_and_statuses_remain_canonical(self) -> None:
        record_types = (
            "profile", "education", "skill", "project", "experience",
            "achievement", "certification", "activity",
        )
        statuses = ("draft", "active", "needs_verification")

        for record_type in record_types:
            with self.subTest(record_type=record_type):
                record = self.parse(extraction_payload(record_type))
                self.assertEqual(record.record_type, record_type)
                self.assertEqual(record.normalization_original_values, {})

        for status in statuses:
            with self.subTest(status=status):
                record = self.parse(extraction_payload("experience", status))
                self.assertEqual(record.status, status)

    def test_unknown_record_type_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported record_type 'unknown_type'"):
            self.parse(extraction_payload("unknown_type"))

    def test_unknown_status_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported status 'unverified-ish'"):
            self.parse(extraction_payload("experience", "unverified-ish"))

    def test_normalization_occurs_before_pydantic_validation(self) -> None:
        payload = extraction_payload("competition", "confirmed")
        original_validate = ingestion.CareerExtractionBundle.model_validate.__func__

        def assert_normalized_before_validation(cls, value, *args, **kwargs):
            self.assertEqual(value["records"][0]["record_type"], "achievement")
            self.assertEqual(value["records"][0]["status"], "active")
            return original_validate(cls, value, *args, **kwargs)

        with patch.object(
            ingestion.CareerExtractionBundle,
            "model_validate",
            classmethod(assert_normalized_before_validation),
        ):
            record = self.parse(payload)

        self.assertEqual(record.record_type, "achievement")
        self.assertEqual(record.status, "active")


class PreviewAnalysisTests(unittest.TestCase):
    def entry(self, record: ingestion.CareerRecord, source_file: str) -> dict:
        return {"record": record, "source_file": source_file}

    def test_duplicate_group_reports_matching_projects_and_field_differences(self) -> None:
        first = ingestion.CareerRecord(
            record_type="project",
            record_id="project-a",
            title="Just Divide",
            fields={"description": "A React puzzle game"},
        )
        second = ingestion.CareerRecord(
            record_type="project",
            record_id="project-b",
            title="Just Divide",
            fields={"description": "A React and TypeScript puzzle game"},
        )

        groups = ingestion.find_duplicate_groups([
            self.entry(first, "resume-a.pdf"),
            self.entry(second, "resume-b.pdf"),
        ])

        self.assertEqual(len(groups), 1)
        differences = ingestion.duplicate_field_differences(groups[0])
        self.assertIn("description", {key for key, _ in differences})

    def test_skill_aliases_are_detected_across_documents(self) -> None:
        first = ingestion.CareerRecord(record_type="skill", record_id="s1", title="AI/ML")
        second = ingestion.CareerRecord(record_type="skill", record_id="s2", title="Machine Learning")

        groups = ingestion.find_duplicate_groups([
            self.entry(first, "resume-a.pdf"),
            self.entry(second, "resume-b.pdf"),
        ])

        self.assertEqual(len(groups), 1)

    def test_classification_warning_flags_award_framed_as_experience(self) -> None:
        record = ingestion.CareerRecord(
            record_type="experience",
            record_id="award-as-role",
            title="Innovation Award",
            fields={"description": "Won a regional competition"},
        )

        warnings = ingestion.classification_warnings([self.entry(record, "resume.pdf")])

        self.assertTrue(any("achievements" in warning for _, warning in warnings))


if __name__ == "__main__":
    unittest.main()