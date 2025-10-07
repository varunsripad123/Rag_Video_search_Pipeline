"""Query expansion for improved search accuracy.

Expands user queries with synonyms and related terms to improve recall.
"""

from typing import List, Dict


# Action-specific expansions
ACTION_EXPANSIONS: Dict[str, List[str]] = {
    "basketball": [
        "basketball",
        "playing basketball",
        "basketball game",
        "shooting basketball",
        "dribbling basketball"
    ],
    "running": [
        "running",
        "person running",
        "jogging",
        "sprint",
        "runner"
    ],
    "walking": [
        "walking",
        "person walking",
        "strolling",
        "pedestrian",
        "walk"
    ],
    "cycling": [
        "cycling",
        "riding bicycle",
        "biking",
        "cyclist",
        "bike riding"
    ],
    "swimming": [
        "swimming",
        "swimmer",
        "swim",
        "swimming pool",
        "freestyle swimming"
    ],
    "dancing": [
        "dancing",
        "dancer",
        "dance performance",
        "choreography",
        "dance moves"
    ],
    "playing guitar": [
        "playing guitar",
        "guitarist",
        "guitar performance",
        "acoustic guitar",
        "electric guitar"
    ],
    "playing piano": [
        "playing piano",
        "pianist",
        "piano performance",
        "keyboard playing",
        "piano keys"
    ],
    "cooking": [
        "cooking",
        "chef cooking",
        "preparing food",
        "kitchen cooking",
        "culinary"
    ],
    "driving": [
        "driving",
        "driver",
        "driving car",
        "vehicle driving",
        "steering wheel"
    ]
}


def expand_query(query: str, max_expansions: int = 5) -> List[str]:
    """
    Expand query with synonyms and related terms.
    
    Args:
        query: Original user query
        max_expansions: Maximum number of expanded queries
    
    Returns:
        List of expanded queries (including original)
    """
    query_lower = query.lower().strip()
    
    # Check if we have predefined expansions
    if query_lower in ACTION_EXPANSIONS:
        expansions = ACTION_EXPANSIONS[query_lower][:max_expansions]
        return expansions
    
    # Check for partial matches
    for action, expansions in ACTION_EXPANSIONS.items():
        if action in query_lower or query_lower in action:
            return expansions[:max_expansions]
    
    # Generic expansion (add context)
    generic_expansions = [
        query,
        f"person {query}",
        f"{query} activity",
        f"{query} action",
        f"someone {query}"
    ]
    
    return generic_expansions[:max_expansions]


def get_expanded_embedding(query: str, embedder, average: bool = True):
    """
    Get embedding for expanded query.
    
    Args:
        query: Original query
        embedder: Query embedder instance
        average: If True, average all expansions. If False, return all.
    
    Returns:
        Averaged embedding or list of embeddings
    """
    import numpy as np
    
    # Expand query
    expanded_queries = expand_query(query)
    
    # Get embeddings for all expansions
    embeddings = []
    for exp_query in expanded_queries:
        emb = embedder.embed(exp_query)
        embeddings.append(emb)
    
    if average:
        # Average all embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        # Renormalize
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
        return avg_embedding
    else:
        return embeddings


# Add more action expansions as needed
def add_custom_expansion(action: str, expansions: List[str]):
    """Add custom expansion for a specific action."""
    ACTION_EXPANSIONS[action.lower()] = expansions


__all__ = ["expand_query", "get_expanded_embedding", "add_custom_expansion"]
