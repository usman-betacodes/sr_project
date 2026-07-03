from collections import defaultdict
from qdrant_client import QdrantClient

from config import (
    FALLBACK_MIN_SCORE, FALLBACK_ENABLED, QDRANT_URL, COLLECTION_NAME,
    TOP_K, MIN_HITS_REQUIRED, REJECT_THRESHOLD
)


def create_qdrant_client() -> QdrantClient:
    """Creates and returns a Qdrant client. Called once at startup."""
    return QdrantClient(url=QDRANT_URL)


def identify_speaker(vector: list[float], client: QdrantClient) -> dict:
    """
    Core identification logic using Score-Weighted Sum + Top-3 reject gate.

    Steps:
    1. Query top TOP_K results from Qdrant
    2. Group hits by speaker name
    3. Sum scores per speaker (rewards frequency + accuracy)
    4. Winning speaker = highest sum
    5. Reject gate: needs MIN_HITS_REQUIRED hits AND top-3 avg >= REJECT_THRESHOLD
    6. Return result dict

    Always returns a dict — never raises. Caller decides HTTP response.
    """
    # query_points() is the correct method in qdrant-client v1.7+
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=TOP_K
    )

    search_result = response.points
    
    if not search_result:
        return {
            "speaker": "Unknown",
            "reason": "Database empty or no results found."
        }

    # --- INJECTED DIAGNOSTIC BLOCK ---
    print("\n--- RAW API SCORES ---")
    for hit in search_result[:3]:
        print(f"Speaker: {hit.payload.get('speaker', 'Unknown')} -> Score: {hit.score:.4f}")
    print("----------------------\n")
    # ------------------------------------

    # Step 2: Group scores by speaker
    speaker_scores = defaultdict(list)
    for hit in search_result:
        speaker_scores[hit.payload.get("speaker", "Unknown")].append(hit.score)

    # Step 3: Score-weighted sum
    speaker_sums = {
        speaker: sum(scores)
        for speaker, scores in speaker_scores.items()
    }

    # Step 4: Winner = highest sum
    winning_speaker = max(speaker_sums, key=speaker_sums.get)
    winning_sum     = speaker_sums[winning_speaker]

    # Step 5: Reject gate — top-3 average of winner
    winner_scores = sorted(speaker_scores[winning_speaker], reverse=True)
    top_3_scores  = winner_scores[:3]
    top_3_avg     = sum(top_3_scores) / len(top_3_scores)

    

    if len(winner_scores) < MIN_HITS_REQUIRED or top_3_avg < REJECT_THRESHOLD:

                      # NEW: Fallback check
        if FALLBACK_ENABLED and winner_scores[0] >= FALLBACK_MIN_SCORE:
           return {
            "speaker": winning_speaker,
            "match_type": "fallback",
            "confidence": "low",
            "best_single_score": round(winner_scores[0], 4),
            "top_3_avg": round(top_3_avg, 4),
            "hits_in_top_20": len(winner_scores),
            "reason": "Strict gate failed, but closest match exceeded fallback threshold"
        }
        return {
            "speaker"          : "Unknown",
            "reason"           : "Insufficient confidence — voice not in DB or audio too noisy",
            "top_3_avg"        : round(top_3_avg, 4),
            "hits_for_winner"  : len(winner_scores),
            "closest_match"    : winning_speaker,       # diagnostic — helps in QA
            "best_single_score": round(winner_scores[0], 4)
        }
        

    


    # Step 6: Confident identification
    return {
        "speaker"          : winning_speaker,
        "confidence_sum"   : round(winning_sum, 4),
        "top_3_avg"        : round(top_3_avg, 4),
        "hits_in_top_20"   : len(winner_scores),
        "best_single_score": round(winner_scores[0], 4)
    }