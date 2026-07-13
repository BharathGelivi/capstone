"""
Test script for Claim Representation Framework (Phase 2.1).
Ensures functionality without utilizing an LLM.
"""

import os
from src.claims import ClaimSet, ClaimFactory, ClaimType, VerificationComplexity

def run_test():
    print("=== Starting Claim Representation Test ===")
    
    # 1. Initialize ClaimFactory for a mock trace
    mock_trace_id = "TRACE_9999"
    factory = ClaimFactory(trace_id=mock_trace_id)
    
    # 2. Create three manual Claim objects
    claim1 = factory.create_claim(
        claim_text="The punishment for murder is death or life imprisonment.",
        source_sentence="According to the context, the punishment for murder is death or imprisonment for life, as stated in section 103(1).",
        sentence_id="S001",
        character_start=0,
        character_end=58,
        # Testing optional fields
        claim_type=ClaimType.PROCEDURAL,
        verification_complexity=VerificationComplexity.VCL1,
        metadata={"confidence": "High"}
    )
    
    claim2 = factory.create_claim(
        claim_text="Murder is defined in section 103(1).",
        source_sentence="According to the context, the punishment for murder is death or imprisonment for life, as stated in section 103(1).",
        sentence_id="S001",
        character_start=59,
        character_end=95
    )
    
    claim3 = factory.create_claim(
        claim_text="The offender is liable to fine.",
        source_sentence="The offender shall also be liable to fine.",
        sentence_id="S002",
        character_start=0,
        character_end=31,
        normalized_text="offender liable to fine",
        claim_hash="hash_12345"
    )
    
    # 3. Insert into ClaimSet
    claim_set = ClaimSet(trace_id=mock_trace_id)
    claim_set.add_claim(claim1)
    claim_set.add_claim(claim2)
    claim_set.add_claim(claim3)
    
    print(f"Created ClaimSet with {claim_set.total_claims} claims.")
    
    # 4. Save to JSON
    save_path = f"artifacts/claim_sets/{mock_trace_id}.json"
    claim_set.to_json(save_path)
    print(f"Saved to {save_path}.")
    
    # 5. Reload and Verify Equality
    print("Reloading ClaimSet...")
    reloaded_set = ClaimSet.from_json(save_path)
    
    # Equality check
    # We compare the dictionary representation (excluding timestamp which might slightly vary if parsed differently, 
    # though dataclass from_json keeps it exact here)
    if claim_set == reloaded_set:
        print("✅ SUCCESS: Reloaded ClaimSet exactly matches the original.")
    else:
        print("❌ FAILURE: Reloaded ClaimSet does NOT match the original.")
        
    print("\n--- Original ---")
    print(claim_set)
    print("\n--- Reloaded ---")
    print(reloaded_set)

if __name__ == "__main__":
    run_test()
