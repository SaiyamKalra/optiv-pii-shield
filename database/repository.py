from database.db import DocumentSummary,PIIDetectionLog, SessionLocal
from models.schemas import PIIOutput

def save_pipeline_output(pii_output:PIIOutput):
    db=SessionLocal();
    try:
        summary_record=DocumentSummary(
            file_id=pii_output.file_id,
            file_name=pii_output.file_name,
            page_number=pii_output.page_number,
            status=pii_output.status,
            verification_passed=pii_output.verification_passed,
            total_entities_detected=len(pii_output.detected_entities),
        )
        db.add(summary_record)

        for entity in pii_output.detected_entities:
            entity_record = PIIDetectionLog(
                file_id=pii_output.file_id,
                token=entity.token,
                entity_type=entity.entity_type,
                confidence_score=entity.score,
                page_number=entity.page_number,
            )
            db.add(entity_record)
        db.commit()

    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()
