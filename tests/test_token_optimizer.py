from models.schemas import PIIOutput
from redaction.token_redaction.token_optimizer import TokenOptimizer


def test_token_optimizer_multi_page_suite():
    # Page 1 Chunk
    p1 = PIIOutput(
        file_id="doc_enterprise_test_001",
        file_name="q3_financial_compliance_audit.pdf",
        source_url="",
        user_id="audit_admin",
        page_number=1,
        status="SUCCESS",
        sanitized_text=(
            "OPTIV GLOBAL SECURITY AUDIT REPORT   \n\n"
            "Page 1 of 3\n\n"
            "Executive Summary:\n"
            "Chief Compliance Officer [PERSON_001] approved the quarterly audit. "
            "Primary escalation point is reach-able at [EMAIL_ADDRESS_001] or [PHONE_NUMBER_001].\n\n"
            "Confidential - Internal Use Only"
        ),
        mapping_ledger={
            "[PERSON_001]": "Sarah Jenkins",
            "[EMAIL_ADDRESS_001]": "s.jenkins@optiv.com",
            "[PHONE_NUMBER_001]": "+1 (555) 019-2834",
        },
        detected_entities=[],
        verification_passed=True,
        detection_counts={"PERSON": 1, "EMAIL_ADDRESS": 1, "PHONE_NUMBER": 1},
    )

    # Page 2 Chunk
    p2 = PIIOutput(
        file_id="doc_enterprise_test_001",
        file_name="q3_financial_compliance_audit.pdf",
        source_url="",
        user_id="audit_admin",
        page_number=2,
        status="SUCCESS",
        sanitized_text=(
            "OPTIV GLOBAL SECURITY AUDIT REPORT   \n\n"
            "Page 2 of 3\n\n"
            "Transaction Verification Ledger:\n"
            "1. Corporate Account Manager [PERSON_002] initiated wire transfer of $150,000.\n"
            "2. Destination SSN record logged as [US_SSN_001].\n"
            "3. Settlement Card: [CREDIT_CARD_001].\n\n"
            "All Rights Reserved."
        ),
        mapping_ledger={
            "[PERSON_002]": "Michael Chang",
            "[US_SSN_001]": "999-00-1234",
            "[CREDIT_CARD_001]": "4532-7112-8901-2345",
        },
        detected_entities=[],
        verification_passed=True,
        detection_counts={"PERSON": 1, "US_SSN": 1, "CREDIT_CARD": 1},
    )

    # Page 3 Chunk
    p3 = PIIOutput(
        file_id="doc_enterprise_test_001",
        file_name="q3_financial_compliance_audit.pdf",
        source_url="",
        user_id="audit_admin",
        page_number=3,
        status="SUCCESS",
        sanitized_text=(
            "OPTIV GLOBAL SECURITY AUDIT REPORT   \n\n"
            "Page 3 of 3\n\n"
            "Infrastructure Nodes:\n"
            "Gateway Router IP: [IP_ADDRESS_001]   \n"
            "Secondary DNS Node: [IP_ADDRESS_002]\n\n"
            "Sign-off confirmed by Lead Auditor [PERSON_003] ([EMAIL_ADDRESS_002]).\n\n"
            "Confidential - Internal Use Only"
        ),
        mapping_ledger={
            "[IP_ADDRESS_001]": "192.168.1.1",
            "[IP_ADDRESS_002]": "10.0.0.1",
            "[PERSON_003]": "David Ross",
            "[EMAIL_ADDRESS_002]": "d.ross@optiv.com",
        },
        detected_entities=[],
        verification_passed=True,
        detection_counts={
            "IP_ADDRESS": 2,
            "PERSON": 1,
            "EMAIL_ADDRESS": 1,
        },
    )

    raw_original_unoptimized = (
        "OPTIV GLOBAL SECURITY AUDIT REPORT Page 1 of 3 Executive Summary: "
        "Chief Compliance Officer Sarah Jenkins approved the quarterly audit. Primary escalation point is reach-able at s.jenkins@optiv.com or +1 (555) 019-2834. Confidential - Internal Use Only "
        "OPTIV GLOBAL SECURITY AUDIT REPORT Page 2 of 3 Transaction Verification Ledger: "
        "1. Corporate Account Manager Michael Chang initiated wire transfer of $150,000. 2. Destination SSN record logged as 999-00-1234. 3. Settlement Card: 4532-7112-8901-2345. All Rights Reserved. "
        "OPTIV GLOBAL SECURITY AUDIT REPORT Page 3 of 3 Infrastructure Nodes: Gateway Router IP: 192.168.1.1 Secondary DNS Node: 10.0.0.1 Sign-off confirmed by Lead Auditor David Ross (d.ross@optiv.com). Confidential - Internal Use Only"
    )

    optimizer = TokenOptimizer()
    md_content = optimizer.optimize_output(
        [p1, p2, p3], raw_original_text=raw_original_unoptimized
    )

    # Terminal output visualization
    print("\n" + "=" * 70)
    print("      OPTIV PII SHIELD - COMPREHENSIVE TOKEN OPTIMIZER TEST OUTPUT      ")
    print("=" * 70)
    print(md_content)
    print("=" * 70 + "\n")

    # Metadata assertions
    assert "title: q3_financial_compliance_audit.pdf" in md_content
    assert "file_id: doc_enterprise_test_001" in md_content
    assert "total_pages: 3" in md_content

    # Page markers
    assert "<!-- Page 1 -->" in md_content or "<!--Page 1-->" in md_content
    assert "<!-- Page 2 -->" in md_content or "<!--Page 2-->" in md_content
    assert "<!-- Page 3 -->" in md_content or "<!--Page 3-->" in md_content

    # Boilerplate removal verification
    assert "Page 1 of 3" not in md_content
    assert "Page 2 of 3" not in md_content
    assert "Page 3 of 3" not in md_content
    assert "Confidential - Internal Use Only" not in md_content
    assert "All Rights Reserved." not in md_content

    # Content integrity
    assert "[PERSON_001]" in md_content
    assert "[CREDIT_CARD_001]" in md_content
    assert "[IP_ADDRESS_001]" in md_content