import json
from pathlib import Path
from types import SimpleNamespace
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

    def test_known_aliases_preserve_model_label(self) -> None:
        expected_types = {
            "person": "profile",
            "leadership": "experience",
            "position": "experience",
            "position_of_responsibility": "experience",
            "competition": "achievement",
            "award": "achievement",
        }
        for alias, expected_type in expected_types.items():
            with self.subTest(alias=alias):
                record = self.parse(extraction_payload(alias))
                self.assertEqual(record.record_type, expected_type)
                self.assertEqual(record.normalized_from, alias)

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

    def test_position_alias_always_maps_to_experience(self) -> None:
        record = self.parse(extraction_payload("position", title="Volunteer Club Member"))

        self.assertEqual(record.record_type, "experience")

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

    def test_summary_with_professional_identity_content_maps_to_profile(self) -> None:
        payload = extraction_payload("summary", title="Summary")
        payload["records"][0]["fields"] = {
            "description": "IoT Engineering student with professional internship experience in embedded systems."
        }

        record = self.parse(payload)

        self.assertEqual(record.record_type, "profile")
        self.assertEqual(record.normalized_from, "summary")

    def test_summary_with_award_content_maps_to_achievement(self) -> None:
        payload = extraction_payload("summary", title="Summary")
        payload["records"][0]["fields"] = {"description": "Won a regional innovation competition."}

        record = self.parse(payload)

        self.assertEqual(record.record_type, "achievement")
        self.assertEqual(record.normalized_from, "summary")

    def test_unclassifiable_summary_fails_clearly(self) -> None:
        payload = extraction_payload("summary", title="Summary")
        payload["records"][0]["fields"] = {"description": "A collection of personal notes."}

        with self.assertRaisesRegex(ValueError, "summary.*does not identify a canonical record type"):
            self.parse(payload)

    def test_profile_github_prefix_is_normalized_without_inventing_url(self) -> None:
        payload = extraction_payload("person", title="Akshat Sharma")
        payload["records"][0]["fields"] = {"github": "/githubgithub.com/Akshat"}

        record = self.parse(payload)

        self.assertEqual(record.fields["github"], "github.com/Akshat")

    def test_malformed_json_gets_one_json_only_retry(self) -> None:
        prompts: list[str] = []

        class FakeModel:
            def invoke(self, prompt: str) -> SimpleNamespace:
                prompts.append(prompt)
                content = "{malformed" if len(prompts) == 1 else '{"records": []}'
                return SimpleNamespace(content=content)

        with patch.object(ingestion, "llm_request_settings", return_value=(5.0, 0)), patch.object(
            ingestion, "create_ingestion_model", return_value=FakeModel()
        ):
            bundle = ingestion.llm_extract_structured_data(
                Path("qualcom.pdf"), "source-hash", "PRIVATE FULL DOCUMENT"
            )

        self.assertEqual(bundle.records, [])
        self.assertEqual(len(prompts), 2)
        self.assertIn("valid JSON only", prompts[1])
        self.assertNotIn("PRIVATE FULL DOCUMENT", prompts[1])

    def test_second_malformed_json_response_fails_after_one_retry(self) -> None:
        prompts: list[str] = []

        class FakeModel:
            def invoke(self, prompt: str) -> SimpleNamespace:
                prompts.append(prompt)
                return SimpleNamespace(content="{still malformed")

        with patch.object(ingestion, "llm_request_settings", return_value=(5.0, 0)), patch.object(
            ingestion, "create_ingestion_model", return_value=FakeModel()
        ):
            with self.assertRaises(json.JSONDecodeError):
                ingestion.llm_extract_structured_data(
                    Path("qualcom.pdf"), "source-hash", "PRIVATE FULL DOCUMENT"
                )

        self.assertEqual(len(prompts), 2)

    def test_provider_wrapped_server_error_is_retryable_but_client_error_is_not(self) -> None:
        self.assertTrue(ingestion.is_retryable_llm_error(SimpleNamespace(status_code=502)))
        self.assertFalse(ingestion.is_retryable_llm_error(SimpleNamespace(status_code=400)))
        self.assertTrue(ingestion.is_retryable_llm_error(SimpleNamespace(retryable=True)))

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

    def evidence(self, source_file: str, claim: str) -> ingestion.SourceEvidence:
        return ingestion.SourceEvidence(
            source_file=source_file,
            source_hash=f"hash-{source_file}",
            extracted_claim=claim,
        )

    def canonicalize(self, *items: tuple[ingestion.CareerRecord, str]) -> list[dict]:
        return ingestion.canonicalize_entries([
            {"record": record, "source_file": source_file}
            for record, source_file in items
        ])

    def test_duplicate_project_merges_compatible_fields_and_evidence(self) -> None:
        first = ingestion.CareerRecord(
            record_type="project", record_id="project-one", title="SCP-2949 Horror Project",
            fields={"technologies": ["Godot", "GDScript"], "start_year": "2023", "description": "A horror game."},
            evidence=[self.evidence("gamedev_resume.pdf", "A Godot horror game.")],
        )
        second = ingestion.CareerRecord(
            record_type="project", record_id="project-two", title="SCP-2949 Horror Project",
            fields={"technologies": ["Godot", "C#"], "end_year": "2024", "description": "A multiplayer horror project."},
            evidence=[self.evidence("main_resume (5).pdf", "A multiplayer horror project.")],
        )

        canonical = self.canonicalize((first, "gamedev_resume.pdf"), (second, "main_resume (5).pdf"))
        record = canonical[0]["record"]

        self.assertEqual(len(canonical), 1)
        self.assertEqual(record.fields["technologies"], ["Godot", "GDScript", "C#"])
        self.assertEqual(record.fields["start_year"], "2023")
        self.assertEqual(record.fields["end_year"], "2024")
        self.assertEqual(record.fields["description"], "A horror game.")
        self.assertEqual({item.source_file for item in record.evidence}, {"gamedev_resume.pdf", "main_resume (5).pdf"})
        self.assertEqual(canonical[0]["source_files"], ["gamedev_resume.pdf", "main_resume (5).pdf"])
        reversed_record = self.canonicalize((second, "main_resume (5).pdf"), (first, "gamedev_resume.pdf"))[0]["record"]
        self.assertEqual(record.record_id, reversed_record.record_id)

    def test_duplicate_experience_merges_by_exact_title_and_organization(self) -> None:
        first = ingestion.CareerRecord(
            record_type="experience", record_id="experience-one", title="Technical Lead at Game Veda",
            fields={"position": "Technical Lead", "organization": "Game Veda", "responsibilities": ["Led engineering"]},
            evidence=[self.evidence("resume-a.pdf", "Technical Lead at Game Veda.")],
        )
        second = ingestion.CareerRecord(
            record_type="experience", record_id="experience-two", title="Technical Lead - Game Veda",
            fields={"role": "Technical Lead", "organization": "Game Veda", "responsibilities": ["Coordinated releases"]},
            evidence=[self.evidence("resume-b.pdf", "Led and coordinated releases.")],
        )

        canonical = self.canonicalize((first, "resume-a.pdf"), (second, "resume-b.pdf"))

        self.assertEqual(len(canonical), 1)
        self.assertEqual(canonical[0]["record"].fields["responsibilities"], ["Led engineering", "Coordinated releases"])
        self.assertEqual(len(canonical[0]["record"].evidence), 2)

    def test_duplicate_profile_merges_contact_details(self) -> None:
        first = ingestion.CareerRecord(
            record_type="profile", record_id="profile-one", title="Akshat Sharma",
            fields={"email": "akshat@example.com", "github": "github.com/Akshat"},
            evidence=[self.evidence("resume-a.pdf", "Akshat Sharma profile and GitHub.")],
        )
        second = ingestion.CareerRecord(
            record_type="profile", record_id="profile-two", title="Akshat Sharma",
            fields={"phone": "555-0100", "linkedin": "linkedin.com/in/akshat"},
            evidence=[self.evidence("resume-b.pdf", "Akshat Sharma contact details.")],
        )

        canonical = self.canonicalize((first, "resume-a.pdf"), (second, "resume-b.pdf"))

        self.assertEqual(len(canonical), 1)
        self.assertEqual(canonical[0]["record"].fields["email"], "akshat@example.com")
        self.assertEqual(canonical[0]["record"].fields["phone"], "555-0100")
        self.assertEqual(len(canonical[0]["record"].evidence), 2)

    def test_skill_category_merge_unions_values_without_fuzzy_matching(self) -> None:
        first = ingestion.CareerRecord(
            record_type="skill", record_id="skill-one", title="Programming Languages",
            fields={"category": "Programming", "languages": ["Python", "GDScript", "C", "Bash", "HTML/CSS"]},
        )
        second = ingestion.CareerRecord(
            record_type="skill", record_id="skill-two", title="Programming Languages",
            fields={"category": "Programming", "skills": ["GDScript", "C#", "Python", "C"]},
        )
        third = ingestion.CareerRecord(
            record_type="skill", record_id="skill-three", title="Programming Languages: Python, GDScript, C, Bash, HTML/CSS",
            fields={"category": "Languages", "skills": ["Python", "GDScript", "C", "Bash", "HTML/CSS"]},
        )
        unrelated = ingestion.CareerRecord(
            record_type="skill", record_id="skill-four", title="Programming Paradigms",
            fields={"skills": ["Object-oriented"]},
        )
        data_tools = ingestion.CareerRecord(
            record_type="skill", record_id="skill-five", title="Data Tools",
            fields={"category": "Data Tools", "skills": ["Kaggle"]},
        )

        canonical = self.canonicalize(
            (first, "resume-a.pdf"),
            (second, "resume-b.pdf"),
            (third, "resume-c.pdf"),
            (unrelated, "resume-d.pdf"),
            (data_tools, "resume-e.pdf"),
        )

        self.assertEqual(len(canonical), 3)
        languages = next(entry["record"] for entry in canonical if entry["record"].fields.get("category") == "Programming Languages")
        self.assertEqual(languages.fields["skills"], ["Python", "GDScript", "C", "Bash", "HTML/CSS", "C#"])

    def test_owner_prefixed_tools_bucket_merges_but_data_tools_stays_separate(self) -> None:
        first = ingestion.CareerRecord(
            record_type="skill", record_id="tools-one", title="Tools",
            fields={"skills": ["Git"]},
        )
        second = ingestion.CareerRecord(
            record_type="skill", record_id="tools-two", title="Akshat Sharma - Tools",
            fields={"name": "Akshat Sharma", "category": "Tools", "skills": ["GCC/GDB"]},
        )
        unrelated = ingestion.CareerRecord(
            record_type="skill", record_id="data-tools", title="Data Tools",
            fields={"category": "Data Tools", "skills": ["Kaggle"]},
        )

        canonical = self.canonicalize(
            (first, "resume-a.pdf"),
            (second, "resume-b.pdf"),
            (unrelated, "resume-c.pdf"),
        )

        self.assertEqual(len(canonical), 2)
        tools = next(entry["record"] for entry in canonical if entry["record"].fields.get("category") == "Tools")
        self.assertEqual(tools.fields["skills"], ["Git", "GCC/GDB"])

    def test_material_field_conflict_marks_record_for_verification(self) -> None:
        first = ingestion.CareerRecord(
            record_type="profile", record_id="profile-one", title="Akshat Sharma",
            fields={"email": "old@example.com"},
        )
        second = ingestion.CareerRecord(
            record_type="profile", record_id="profile-two", title="Akshat Sharma",
            fields={"email": "new@example.com"},
        )

        record = self.canonicalize((first, "resume-a.pdf"), (second, "resume-b.pdf"))[0]["record"]

        self.assertEqual(record.status, "needs_verification")
        self.assertTrue(any("email" in conflict for conflict in record.conflicts))

    def test_missing_field_value_does_not_conflict_with_populated_value(self) -> None:
        first = ingestion.CareerRecord(
            record_type="project", record_id="project-one", title="SCP-2949 Horror Project",
            fields={"end_date": None},
        )
        second = ingestion.CareerRecord(
            record_type="project", record_id="project-two", title="SCP-2949 Horror Project",
            fields={"end_date": "Ongoing"},
        )

        record = self.canonicalize((first, "resume-a.pdf"), (second, "resume-b.pdf"))[0]["record"]

        self.assertEqual(record.fields["end_date"], "Ongoing")
        self.assertEqual(record.status, "draft")

    def test_different_descriptions_remain_in_source_evidence_without_conflict(self) -> None:
        first = ingestion.CareerRecord(
            record_type="project", record_id="project-one", title="SCP-2949 Horror Project",
            fields={"description": "A horror game."},
            evidence=[self.evidence("resume-a.pdf", "A horror game.")],
        )
        second = ingestion.CareerRecord(
            record_type="project", record_id="project-two", title="SCP-2949 Horror Project",
            fields={"description": "A multiplayer horror game."},
            evidence=[self.evidence("resume-b.pdf", "A multiplayer horror game.")],
        )

        record = self.canonicalize((first, "resume-a.pdf"), (second, "resume-b.pdf"))[0]["record"]

        self.assertEqual(record.status, "draft")
        self.assertEqual({item.extracted_claim for item in record.evidence}, {"A horror game.", "A multiplayer horror game."})


if __name__ == "__main__":
    unittest.main()