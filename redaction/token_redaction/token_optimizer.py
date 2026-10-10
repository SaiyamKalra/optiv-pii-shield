import re
from datetime import datetime, timezone
from typing import Any

from models.schemas import PIIOutput


class TokenOptimizer:
    def __init__(
        self,
        strip_headers_footers: bool=True,
        normalize_whitespace: bool=True,
        include_page_markers: bool=True,
        include_metadata_header: bool=True,
    ):
        self.strip_headers_footers=strip_headers_footers
        self.normalize_whitespace=normalize_whitespace
        self.include_page_markers=include_page_markers
        self.include_metadata_header=include_metadata_header

    def optimize_output(self,pii_outputs:list[PIIOutput],raw_original_text: str="")->str:
        if not pii_outputs:
            return ""

        optimized_pages=[]
        
        for pii in pii_outputs:
            text=pii.sanitized_text or ""

            if not text.strip():
                continue
            
            if self.strip_headers_footers:
                text=re.sub(r"(?i)page\s+\d+\s+(of|/)\s+\d+","",text)
                text = re.sub(
                    r"(?i)confidential\s*(-|\|)?\s*internal\s+use\s+only",
                    "",
                    text,
                )
                text = re.sub(r"(?i)all\s+rights\s+reserved\.?", "", text)

            if self.normalize_whitespace:
                text=re.sub(r"[ \t]+"," ",text)
                text=re.sub(r"\n{3,}","\n\n",text)
                
            text=text.strip()

            if self.include_page_markers:
                page_block=f"<!-- Page {pii.page_number} -->\n{text}"
            else:
                page_block=text
                
            optimized_pages.append(page_block)
                
                
        body_markdown = "\n\n---\n\n".join(optimized_pages)
        
        if self.include_metadata_header and pii_outputs:
            first_doc=pii_outputs[0]
            metrics=self.calculate_metrics(raw_original_text,body_markdown)
            header = (
                f"---\n"
                f"title: {first_doc.file_name}\n"
                f"file_id: {first_doc.file_id}\n"
                f"processed_at: {datetime.now(timezone.utc).isoformat()}\n"
                f"total_pages: {len(pii_outputs)}\n"
                f"estimated_tokens_saved: {metrics['tokens_saved']} ({metrics['savings_percentage']}%)\n"
                f"---\n\n"
            )
            return header + body_markdown
        return body_markdown

    @staticmethod
    def calculate_metrics(raw_text: str,optimized_text:str)->dict[str,Any]:
        raw_chars=len(raw_text) if raw_text else 1
        opt_chars=len(optimized_text)
        est_raw_tokens = raw_chars 
        est_opt_tokens = opt_chars 
        tokens_saved = max(0, est_raw_tokens - est_opt_tokens)
        savings_pct = (
            round((tokens_saved / est_raw_tokens) * 100, 2)
            if est_raw_tokens > 0
            else 0.0
        )

        return {
            "raw_characters": raw_chars,
            "optimized_characters": opt_chars,
            "estimated_raw_tokens": est_raw_tokens,
            "estimated_optimized_tokens": est_opt_tokens,
            "tokens_saved": tokens_saved,
            "savings_percentage": max(0.0, savings_pct),
        }

