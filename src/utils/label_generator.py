from rouge_score import rouge_scorer

def create_labels(article_sentences, summary):
    """
    Generates ground-truth labels for extractive summarization.

    Args:
        article_sentences (list of str): The sentences of the article.
        summary (str): The ground-truth summary.

    Returns:
        list of int: A list of binary labels (0 or 1) for each sentence.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    
    # Calculate ROUGE scores for each sentence against the summary
    scores = [scorer.score(summary, sent)['rouge1'].fmeasure for sent in article_sentences]
    
    # Greedily select sentences to maximize ROUGE score
    selected_indices = []
    best_rouge = 0.0
    
    # Sort sentences by score to start with the best candidates
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    current_summary_sentences = []
    for i in sorted_indices:
        # Add the next best sentence and see if it improves the overall ROUGE score
        temp_summary = " ".join(current_summary_sentences + [article_sentences[i]])
        new_rouge = scorer.score(summary, temp_summary)['rouge1'].fmeasure
        
        if new_rouge > best_rouge:
            best_rouge = new_rouge
            current_summary_sentences.append(article_sentences[i])
            selected_indices.append(i)

    # Create binary labels
    labels = [0] * len(article_sentences)
    for i in selected_indices:
        labels[i] = 1
        
    return labels
