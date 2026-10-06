import re


class TokenManager:
    
    def __init__(self):
        self.mapping_ledger: dict[str,str]={}
        self.value_to_token: dict[str,str]={}
        self.entity_counters: dict[str,int]={}
        self.person_canonical_map: dict[str,str]={}
    
    def _normalize_key(self, value: str, entity_type: str) -> str:
        
        clean_val=value.strip().lower()

        if entity_type=="EMAIL_ADDRESS":
            return clean_val

        if entity_type=="PERSON":
            
            clean_val=re.sub(r"\b([a-z])\.\s?",r"\1 ",clean_val)
            tokens=clean_val.split()

            if len(tokens)>=2:
                first_initial=tokens[0][0]
                last_name=tokens[-1]
                fuzzy_key=f"person_{first_initial}_{last_name}"

                if last_name in self.person_canonical_map:
                    existing_key = self.person_canonical_map[last_name]
                    if existing_key.startswith(f"person_{first_initial}_"):
                        return existing_key

                if len(tokens[0])>2:
                    self.person_canonical_map[last_name] = fuzzy_key
                
                return fuzzy_key
        
        return f"{entity_type.lower()}_{clean_val}"
    
    def get_or_create_token(self,original_value:str,entity_type:str)->str:
        norm_key=self._normalize_key(original_value,entity_type)

        if norm_key in self.value_to_token:
            return self.value_to_token[norm_key]
        
        count = self.entity_counters.get(entity_type, 0) + 1
        self.entity_counters[entity_type] = count
        token = f"[{entity_type}_{count:03d}]"

        self.value_to_token[norm_key] = token
        self.mapping_ledger[token] = original_value.strip()

        return token

    def reset(self):
        self.mapping_ledger.clear()
        self.value_to_token.clear()
        self.entity_counters.clear()
        self.person_canonical_map.clear()
        
