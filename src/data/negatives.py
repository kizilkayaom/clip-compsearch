import spacy

nlp = spacy.load("en_core_web_sm")

def swap_objects(nlp, caption):
    doc = nlp(caption)
    
    noun_tokens = []
    for token in doc:
        if token.pos_ == "NOUN":
            noun_tokens.append(token)
            
    words = [token.text for token in doc]
    
    if len(noun_tokens) >= 2:
        index_1 = noun_tokens[0].i
        index_2 = noun_tokens[-1].i
        
        word_1_orig = words[index_1]
        word_2_orig = words[index_2]
        
        is_index_1_title = word_1_orig.istitle()
        is_index_2_title = word_2_orig.istitle()
        
        words[index_1] = word_2_orig
        words[index_2] = word_1_orig
        
        if is_index_1_title:
            words[index_1] = words[index_1].capitalize()
        else:
            words[index_1] = words[index_1].lower()
            
        if is_index_2_title:
            words[index_2] = words[index_2].capitalize()
        else:
            words[index_2] = words[index_2].lower()
        
    return " ".join(words)

def swap_attributes(nlp, caption):
    doc = nlp(caption)
    
    adj_tokens = []
    for token in doc:
        if token.pos_ == "ADJ":
            adj_tokens.append(token)
            
    words = [token.text for token in doc]
    
    if len(adj_tokens) >= 2:
        index_1 = adj_tokens[0].i
        index_2 = adj_tokens[-1].i
        
        word_1_orig = words[index_1]
        word_2_orig = words[index_2]
        
        is_index_1_title = word_1_orig.istitle()
        is_index_2_title = word_2_orig.istitle()
        
        words[index_1] = word_2_orig
        words[index_2] = word_1_orig
        
        if is_index_1_title:
            words[index_1] = words[index_1].capitalize()
        else:
            words[index_1] = words[index_1].lower()
            
        if is_index_2_title:
            words[index_2] = words[index_2].capitalize()
        else:
            words[index_2] = words[index_2].lower()
        
    return " ".join(words)