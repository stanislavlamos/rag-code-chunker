def calculate_token_metrics(retrieved_chunks: List[str], golden_excerpt: str) -> Dict[str, float]:
    """
    Calculate token-wise precision and recall
    :param retrieved_chunks: list of retrieved text chunks
    :param golden_excerpt: golden excerpt containing relevant tokens
    :return: dictionary of metrics
    """
    # Get sets of tokens
    te = tokenize(golden_excerpt)  # tokens in golden excerpt
    tr = set()  # tokens in retrieved chunks
    
    # Combine tokens from all retrieved chunks
    for chunk in retrieved_chunks:
        tr.update(tokenize(chunk))
    
    # Calculate intersection
    intersection = te.intersection(tr)
    
    # Calculate metrics
    if len(tr) == 0:
        precision = 0.0
    else:
        precision = len(intersection) / len(tr)
    
    if len(te) == 0:
        recall = 0.0
    else:
        recall = len(intersection) / len(te)
    
    # Calculate IoU
    union = len(te) + len(tr) - len(intersection)
    if union == 0:
        iou = 0.0
    else:
        iou = len(intersection) / union
    
    return {
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "num_relevant_tokens": len(te),
        "num_retrieved_tokens": len(tr),
        "num_intersection_tokens": len(intersection)
    }