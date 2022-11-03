import numpy as np
import random
import spacy

### ProfessorBob format dataset reader
class PBDataset():
    def __init__(self, data):
        self.data = data
        self.cum_sum = np.cumsum([len(document['qas']) for document in self.data])
        
    def __len__(self):
        return sum([len(document['qas']) for document in self.data])
    
    def __getitem__(self, index):
        doc_index, _ = list(filter(lambda d : index < d[1], zip(range(len(self.cum_sum)), self.cum_sum[::-1])))[-1]
        doc_index  = len(self.data) -1 - doc_index
        qas_index = index  - (0 if(doc_index == 0) else self.cum_sum[doc_index - 1])
        
        item = self.data[doc_index]['qas'][qas_index] 
        return {'paragraphs' : self.data[doc_index]['paragraphs'], **item}



def all_subtree_flatened(root):
    children = list(root.children)
    total = []
    if(len(children) == 0):
        return []
    for child in children:
        total += all_subtree_flatened(child)
    return [root] + children + total

def get_right_verb(sentence):
    object_list = []
    set_idx = set()
    for token in sentence:
        if token.pos_ == "VERB" and token.idx not in set_idx:
            list_sub = all_subtree_flatened(token)
            #print("icic", list_sub)
            if(len(list_sub) == 0):
                continue
            set_idx = set_idx.union({t.idx for t in list_sub if(t.idx >= token.idx)})
            idx_start = [t.idx for t in list_sub if(t.idx >= token.idx)]
            idx_end = [t.idx + len(t.text) for t in list_sub if(t.idx >= token.idx)]
            start_at = min(idx_start)
            end_at = max(idx_end)
            object_list.append((start_at, end_at))
    return object_list

def get_cobj(sentence):
    cobj_list = []
    #print(f"sent : {sentence}")
    for token in sentence:
        #print(token.dep_)
        with_head = {} # {"obj", "iobj", "nsubj"}
        if('obj' in token.dep_ or "iobj" in token.dep_): # or "nsubj" in token.dep_):
            #print(token.head.text)
            start_at = token.idx
            # find all subtree        
            idx_start = [t.idx for t in all_subtree_flatened(token)]  +  [token.head.idx, token.idx] if(token.dep_ in with_head ) else [token.idx] # + [token.head.idx]
            idx_end = [t.idx + len(t.text) for t in all_subtree_flatened(token.head if(token.dep_ in with_head ) else token)] + [token.idx + len(token.text)]
            #                                                                        token.head.idx + len(token.head.text)]
            start_at = min(idx_start)
            end_at = max(idx_end)
            cobj_list.append((start_at, end_at))
    return cobj_list





class FQAGPBDataset():
    nlp = spacy.load("fr_core_news_lg")
    @staticmethod
    def rverb(context, start_pos, end_pos):
        response = []
        selected_sentences = []
        
        t = 0
        for sent in nlp_parse(context).sents:
            t += len(sent.text_with_ws)  
            if t > start_pos:
                selected_sentences.append(sent)
            if t > end_pos:
                break

        for sent in selected_sentences:
            list_cobj = get_right_verb(sent)
            response += [f"{context[:s]}<hl>{context[s:e]}<hl>{context[e:]}" for s,e in list_cobj]
        if len(res) == 0 :
            return [f"{context[:start_pos]}<hl>{context[start_pos: end_pos]}<hl>{context[end_pos:]}"]
        return response
    @staticmethod
    def cobject(context, start_pos, end_pos):
        response = []
        selected_sentences = []
        
        t = 0
        for sent in nlp_parse(context).sents:
            t += len(sent.text_with_ws)  
            if t > start_pos:
                selected_sentences.append(sent)
            if t > end_pos:
                break

        for sent in selected_sentences:
            list_cobj = get_cobj(sent)
            response += [f"{context[:s]}<hl>{context[s:e]}<hl>{context[e:]}" for s,e in list_cobj]
        if len(response) == 0 :
            return [f"{context[:start_pos]}<hl>{context[start_pos: end_pos]}<hl>{context[end_pos:]}"], True
        return response, False
    @staticmethod
    def default_processing(context, start_pos, end_pos):
        return [f"{context[:start_pos]}<hl>{context[start_pos: end_pos]}<hl>{context[end_pos:]}"], True
    @staticmethod
    def sentences(context, start_pos, end_pos):
        #print(['-> '+i.text for i in nlp_parse(context).sents])
        selected_sentences = []
        t = 0
        for sent in nlp_parse(context).sents:
            t += len(sent.text_with_ws)  
            if t > start_pos:
                selected_sentences.append(sent)
            if t > end_pos:
                break
        return [f"{context[:s.start_char]}<hl>{s.text_with_ws}<hl>{context[s.end_char:]}" for s in selected_sentences], False

    @staticmethod
    def entities(context, start_pos, end_pos):
        selected_sentences = []
        t = 0
        for sent in nlp(context).sents:
            t += len(sent.text_with_ws)  
            if t > start_pos:
                selected_sentences.append(sent)
            if t > end_pos:
                break
        entity_list = sum([ list(sent.ents) for sent in selected_sentences], [])

        if len(entity_list) == 0 :
            return [f"{context[:start_pos]}<hl>{context[start_pos: end_pos]}<hl>{context[end_pos:]}"], True
        return [f"{context[:e.start_char]}<hl>{str(e)}<hl>{context[e.end_char:]}" for e in entity_list], False
    
    @staticmethod
    def nouns_phrase(context, start_pos, end_pos):
        selected_sentences = []
        t = 0

        for sent in nlp(context).sents:
            t += len(sent.text_with_ws)  
            if t > start_pos:
                selected_sentences.append(sent)
            if t > end_pos:
                break

        entitity_list = sum([ list(sent.ents) for sent in selected_sentences], [])
        noun_phrase_list = sum([ list(sent.noun_chunks) for sent in selected_sentences], [])
        noun_phrase_filtered_list = []
        for nph in noun_phrase_list:
            insert_in = True
            for ent in entitity_list:

                se, ee = ent.start_char, ent.end_char
                sn, en = nph.start_char, nph.end_char
                if(not (ee < sn or en < se)):
                    insert_in = False
            if(insert_in):
                noun_phrase_filtered_list.append(nph)
        if len(noun_phrase_filtered_list) == 0 :
            return [f"{context[:start_pos]}<hl>{context[start_pos: end_pos]}<hl>{context[end_pos:]}"], True
        return [f"{context[:e.start_char]}<hl>{str(e)}<hl>{context[e.end_char:]}" for e in noun_phrase_filtered_list], False

    def __init__(self, data, 
        higlight_token="<hl>", 
        input_processing=None,
        sampler = lambda x : x,
        input_lang="fr",
        output_lang="fr"
        ):
        self.dataset = PBDataset(data)
        self.higlight_token = higlight_token
        self.input_processing =\
            input_processing if(input_processing is not None) else FQAGPBDataset.default_processing
        self.sampler = sampler
        self.input_lang = input_lang
        self.output_lang = output_lang

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        #print(data)
        paragraph = data["paragraphs"]
        question = data["question"]
        question_type = data["question_type"] if("question_type" in data) else "NONE"
        items = []

        for sample in self.sampler(data['answers']):
            #print(sample)
            for part in sample:
                context = paragraph[part['paragraph_id']]['content']
                start_pos = part["answer_start"]
                end_pos = start_pos + len(part["text"])
                answer = part['text']
                context, is_default = self.input_processing(context, start_pos, end_pos)
                
                items += [{"is_default": is_default, "id": index, "question": question, "context": ctx, "question_type": question_type} for ctx in context]
        return {**items[0], 'input_lang':self.input_lang, 'output_lang': self.output_lang}
                
            
        
        


