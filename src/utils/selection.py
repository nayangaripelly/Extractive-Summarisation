import math

def greedy_selection_with_trigram_blocking(sentences, scores):
    """
    Selects the top ceil(n/4) sentences based on scores, with trigram blocking.
    """
    num_sentences = len(sentences)
    num_to_select = math.ceil(num_sentences / 4)
    
    # Combine sentences and scores and sort
    ranked_sentences = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    
    selected_summary_sentences = []
    
    for sentence, score in ranked_sentences:
        if len(selected_summary_sentences) >= num_to_select:
            break
            
        if not trigram_blocking(selected_summary_sentences, sentence):
            selected_summary_sentences.append(sentence)
            
    return " ".join(selected_summary_sentences)

def trigram_blocking(summary_sentences, candidate_sentence):
    """
    Prevents redundant sentences from being added to the summary.
    """
    summary_trigrams = set()
    for sent in summary_sentences:
        words = sent.lower().split()
        for i in range(len(words) - 2):
            summary_trigrams.add(tuple(words[i:i+3]))
            
    candidate_words = candidate_sentence.lower().split()
    for i in range(len(candidate_words) - 2):
        if tuple(candidate_words[i:i+3]) in summary_trigrams:
            return True  # Block this sentence
            
    return False
