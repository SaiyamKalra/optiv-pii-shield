# tests/test_pii.py

import sys
from pathlib import Path

# Add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from models.schemas import ExtractionInput
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
        Primary Escalation: Marguerite Vossberg (Direct: +1 (555) 0114 | Alt: 555.321.8890 x402)
        Email: M.VOSSBERG@cadencefg.example, Subdomain contact: alex.smith+privacy@dev.internal.co.uk
        
        International Ops Lead: Jean-Luc Picard-Smyth (+44 (0) 20 7946 0958)
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
        * Contact: +91 98765 43210 / +1 (800) 555-0199
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
        assert (
            "+1 (555) 0114" not in res.sanitized_text
        ), f"Unmasked phone number found on page {idx}"

    assert len(final_ledger) > 0, "Mapping ledger should not be empty"


if __name__ == "__main__":


    test_hard_pii_stress_test()