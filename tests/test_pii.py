# tests/test_pii.py

import sys
from pathlib import Path

import pytest

# Add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from config import settings
from models.schemas import ExtractionInput
from pii.person_filter import (
    DocumentVocabulary,
    is_person_name,
    is_single_name_word,
    surname_keys,
    surname_spans,
    trim_person_span,
)
from pii.pipeline import PIIPipeline


def test_hard_pii_stress_test():
    pipeline = PIIPipeline()

    page1 = ExtractionInput(
        file_id="stress_doc_999",
        file_name="Q3_Security_Audit_and_Org_Roster.pptx",
        source_url="s3://secure-corp-bucket/docs/Q3_Audit.pptx",
        user_id="lead_auditor_07",
        page_number=1,
        raw_text="""
        =================== OPTIV CONFIDENTIAL ROSTER ===================
        Primary Escalation: Marguerite Vossberg (Direct:| Alt: 555.321.8890 x402)
        Email: M.VOSSBERG@cadencefg.example, Subdomain contact: alex.smith+privacy@dev.internal.co.uk
        
        International Ops Lead: Jean-Luc Picard-Smyth 
        Contractor Initial: A.-J. Miller (Email: a.j.miller-test@sub.corp.org)
        
        Infrastructure Nodes: Primary Gateway at 192.168.1.254, Fallback: 10.0.0.1
        Financial Gatekeeper: Credit Card Ref # 4532-7112-8901-2345 | SSN: 000-12-3456
        =================================================================
        """,
    )

    page2 = ExtractionInput(
        file_id="stress_doc_999",
        file_name="Q3_Security_Audit_and_Org_Roster.pptx",
        source_url="s3://secure-corp-bucket/docs/Q3_Audit.pptx",
        user_id="lead_auditor_07",
        page_number=2,
        raw_text="""
        RACI Matrix & Compliance Sign-off:
        -----------------------------------------------------------------
        1. RACI Approval: Signed off by M. Vossberg and A.-J. Miller.
        2. SLA & KPI Assessment: Verified against the quarterly SWOT framework.
        3. Escalation Contact: Reach out to Marguerite Vossberg via m.vossberg@cadencefg.example.
        4. Emergency Mobile: +1 (555) 0114 or direct internal line ext. 902.
        """,
    )

    page3 = ExtractionInput(
        file_id="stress_doc_999",
        file_name="Q3_Security_Audit_and_Org_Roster.pptx",
        source_url="s3://secure-corp-bucket/docs/Q3_Audit.pptx",
        user_id="lead_auditor_07",
        page_number=3,
        raw_text="""
        Roster Appendix (Raw Data Dump):
        * J.-P. van der Berg (j.vanderberg@global-corp.de)
        * Contact: +91 98765 43210
        * Final Sign-off: Tobias Lindqvist [Approved by M. Vossberg].
        """,
    )

    chunks = [page1, page2, page3]
    results = pipeline.process_document_chunks(chunks)

    # Console prints for local verification
    print("\n" + "=" * 80)
    print("STRESS TEST RESULTS")
    print("=" * 80)

    for idx, res in enumerate(results, start=1):
        print(f"\n--- PAGE {idx} SANITIZED TEXT ---")
        print(res.sanitized_text.strip())
        print(
            f"\nPage {idx} Verification Status:"
            f" {'PASSED ✅' if res.verification_passed else 'FLAGGED ❌'}"
        )
        print("-" * 80)

    print("\n--- CUMULATIVE TOKEN MAPPING LEDGER ---")
    final_ledger = results[-1].mapping_ledger
    for token, val in sorted(final_ledger.items()):
        print(f"  {token:<25} ==> {val}")

    # Assertions
    assert len(results) == 3, "Pipeline should return 3 processed output chunks"

    for idx, res in enumerate(results, start=1):
        assert (
            res.verification_passed is True
        ), f"Page {idx} failed verification gate"
        assert (
            "m.vossberg@cadencefg.example" not in res.sanitized_text
        ), f"Unmasked email found on page {idx}"
    assert len(final_ledger) > 0, "Mapping ledger should not be empty"


# --- recognizers (no spaCy needed) ------------------------------------------------------



def _found(recognizer, text):
    return [text[r.start:r.end] for r in recognizer.analyze(text, recognizer.supported_entities)]


@pytest.mark.parametrize(("recognizer", "text", "expected"), [
    ("PassportRecognizer", "Hiroshi Yamagata (EMP-41310, passport TK4820913)", ["TK4820913"]),
    ("PassportRecognizer", "Reference TK4820913 in the register", []),
    ("NationalIdRecognizer", "Reconciliation analyst PESEL 890412xxxxx +48 512", ["890412xxxxx"]),
    ("NationalIdRecognizer", "Ticket 890412 was closed", []),
    # The label must come directly before the value, not anywhere nearby.
    ("NationalIdRecognizer", "PESEL 890412xxxxx +48 512555 0180", ["890412xxxxx"]),
    ("PassportRecognizer", "passport holder since 2019, ref TK4820913", []),
    ("DateOfBirthRecognizer", "Priyanka Raghavan (EMP-40318, DOB 14 July 1991)", ["14 July 1991"]),
    ("DateOfBirthRecognizer", "your date of birth, 3 August 1996.", ["3 August 1996"]),
    ("DateOfBirthRecognizer", "Approved on 14 July 2026 by the Board", []),
    ("PanRecognizer", "Application support L2 PAN AQBPR7741L +91", ["AQBPR7741L"]),
    ("VendorIdRecognizer", "Rohan S. Deshmukh MER-IN-4471 Application support", ["MER-IN-4471"]),
    ("VendorIdRecognizer", "See GRP-POL-001 and CTL-IAM-04", []),
    ("StreetAddressRecognizer", "(home address 88 Willowbank Court, Hoboken, NJ 07030)",
     ["88 Willowbank Court", "Hoboken, NJ 07030"]),
    ("StreetAddressRecognizer", "your home at 2214 Palmetto Crescent, Coral Gables, FL 33134.",
     ["2214 Palmetto Crescent", "Coral Gables, FL 33134"]),
    ("CorpEmailRecognizer", "Naomi K. Osadebe (EMP-41077,\nn.osadebe @cadencefg.com, +1",
     ["n.osadebe @cadencefg.com"]),
])
def test_new_recognizers(recognizer, text, expected):
    assert _found(getattr(settings, recognizer), text) == expected


@pytest.mark.parametrize("phone", [
    "+33 1 5555 0144", "+91 98860 55107", "+65 6555 0172", "+1 (555) 0114", "+48 512 555 0180",
])
def test_international_phones(phone):
    assert phone in _found(settings.RobustPhoneRecognizer, f"Call {phone} today")


def test_phone_digit_count_rejects_short_numbers():
    assert _found(settings.RobustPhoneRecognizer, "Version +1 2 3 of the policy") == []


# --- person names: decided from the document, no word lists ------------------------------

# Ordinary words in this text appear in lowercase at least twice ("risk", "board",
# "committee", "head", "led", "data", "task", "name"). Names never do.
DOC = """
The board approves risk appetite. The risk committee and the audit committee meet monthly;
the board chair and the committee head review each task name. Work led by the data team
is led through the board. Each head of function owns data; the task name field is free text.
Contacts: n.osadebe@cadencefg.com, g.park@cadencefg.com, board.chair@cadencefg.com.
There is car park access for the park team.
"""
VOCAB = DocumentVocabulary([DOC])


@pytest.mark.parametrize("value", [
    "Miriam T. Achebe", "Tobias Lindqvist", "Naomi K. Osadebe", "J.-P. van der Berg",
    "Hyun-Woo Park", "Sofia Marchetti-Lang",
])
def test_names_accepted(value):
    assert is_person_name(value, VOCAB)


@pytest.mark.parametrize("value", [
    "Board Risk Committee",   # every word is written in lowercase in the document
    "Head Data",
    "Task Name",
    "Q);FPlALA!AQ",           # not name-shaped
    "Jncontrolled",           # one word: not a full name
])
def test_ordinary_words_rejected(value):
    assert not is_person_name(value, VOCAB)


def test_single_lowercase_sighting_does_not_make_a_name_common():
    vocab = DocumentVocabulary(["Yevgeni A. Sorokina chairs. " * 5 + "OCR read A. sorokina once."])
    assert is_person_name("Yevgeni A. Sorokina", vocab)


def test_email_confirms_a_surname_that_is_also_a_word():
    # "park" is an ordinary word here, but g.park@... confirms Park as a surname.
    assert is_person_name("Grace Park", VOCAB)
    assert is_single_name_word("Osadebe", VOCAB)


def test_role_mailbox_confirms_nothing():
    # board.chair@... : both parts are ordinary words in the document.
    assert not VOCAB.is_email_confirmed("Board")
    assert not is_person_name("Board Chair", VOCAB, model_backed=False)


def test_pattern_only_name_needs_every_word_backed():
    known = {"Eleanor", "Vasquez-Holt"}
    assert is_person_name("Eleanor R. Vasquez-Holt", VOCAB, model_backed=False, known_words=known)
    assert not is_person_name("Directors Eleanor", VOCAB, model_backed=False, known_words=known)


@pytest.mark.parametrize(("text", "expected"), [
    ("Marguerite Vossberg Led", "Marguerite Vossberg"),       # "led" is ordinary here
    ("H. Park VP Data", "H. Park"),                           # capitals, ordinary word
    ("DAYS OVERDUE Hiroshi Yamagata", "Hiroshi Yamagata"),
    ("Board CRO Miriam T. Achebe", "Miriam T. Achebe"),
    ("by Tobias Lindqvist,", "Tobias Lindqvist"),
    ("Dir. Rafael Quintanilla", "Rafael Quintanilla"),        # abbreviation
])
def test_trim_person_span(text, expected):
    start, end = trim_person_span(text, 0, len(text), VOCAB)
    assert text[start:end] == expected


def test_surname_anchoring_stops_at_labels():
    text = "Medium Devansh R. Kulkarni and Officer Kulkarni"
    keys = surname_keys(["Devansh R. Kulkarni"], DocumentVocabulary([text]))
    assert [text[a:b] for a, b in surname_spans(text, keys, {"Devansh"})] == [
        "Devansh R. Kulkarni", "Kulkarni"]


def test_repeated_first_name_before_a_confirmed_surname_is_taken():
    from pii.person_filter import repeated_first_names

    text = ("Led by Priyanka Raghunathan\n"
            "Priyanka Raghunathan | p.raghunathan@x.example\n"
            "Officer Raghunathan signs once.")
    vocab = DocumentVocabulary([text])
    keys = surname_keys([], vocab)
    pairs = repeated_first_names([text], keys, vocab)
    assert pairs == {("Priyanka", "raghunathan")}   # "Officer" appears only once
    found = [text[a:b] for a, b in surname_spans(text, keys, set(), vocab, pairs)]
    assert found == ["Priyanka Raghunathan", "Priyanka Raghunathan", "Raghunathan"]


def test_canonical_names_drop_label_words():
    from pii.person_filter import canonical_names

    names = {"Medium Helena Vorstermann", "Helena Vorstermann", "Miriam T. Achebe"}
    assert canonical_names(names) == {"Helena Vorstermann", "Miriam T. Achebe"}


def test_surname_anchoring_takes_initials_and_first_name():
    text = "Owners: P. S. Raghavan, M.Achebe and Ifeoma C. Adeyemi; the board reviews Raghavan."
    vocab = DocumentVocabulary([text, DOC, "i.adeyemi@cadencefg.com"])
    keys = surname_keys(["Priyanka S. Raghavan", "Miriam T. Achebe"], vocab)
    found = [text[a:b] for a, b in surname_spans(text, keys, {"Priyanka", "Miriam"})]
    assert found == ["P. S. Raghavan", "M.Achebe", "Ifeoma C. Adeyemi", "Raghavan"]


# --- pipeline behaviour -------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline():
    return PIIPipeline()


def _page(n, text):
    return ExtractionInput(file_id="f", file_name="x.pdf", page_number=n, raw_text=text)


def test_full_name_found_once_is_redacted_on_every_page(pipeline):
    results = pipeline.process_document_chunks([
        _page(1, "Policy owner: Naomi K. Osadebe, Head of IAM."),
        _page(2, "IAM Engineer II\nNaomi K. Osadebe EMP-41077\nCaleb Whitfield"),
    ])
    page2 = [(e.entity_type, e.original_value) for e in results[1].detected_entities]
    assert ("PERSON", "Naomi K. Osadebe") in page2


def test_organisations_places_and_policy_numbers_are_not_redacted(pipeline):
    text = ("The Board Risk Committee of Cadence Financial Group in New York approves "
            "GRP-POL-001 and exception EXC-2026-041.")
    found = pipeline.process_chunk(_page(1, text)).detected_entities
    assert found == []


if __name__ == "__main__":


    test_hard_pii_stress_test()