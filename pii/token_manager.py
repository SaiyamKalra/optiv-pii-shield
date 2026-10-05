class TokenManager:
    
    def __init__(self):
        self.mapping_ledger: dict[str,str]={}
        self.value_to_token: dict[str,str]={}
        self.entity_counters: dict[str,int]={}
    
    def get_or_create_token(self, original_val: str, entity_type: str) -> str:
        
        clean_val=original_val.strip()

        if clean_val in self.value_to_token:
            return self.value_to_token[clean_val]
        
        self.entity_counters[entity_type]=(
            self.entity_counters.get(entity_type,0)+1
        )

        token=f"[{entity_type}_{self.entity_counters[entity_type]:03d}]"

        self.value_to_token[clean_val]=token
        self.mapping_ledger[token]=clean_val
        return token
    
    def reset(self):
        self.mapping_ledger.clear()
        self.value_to_token.clear()
        self.entity_counters.clear()
        
