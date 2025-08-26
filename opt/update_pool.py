import re
import random

# === 替换词表 & 修饰符 ===
SYNONYM_TABLE = {
    "respond": ["reply", "answer"],
    "format": ["style", "structure", "layout"],
    "check": ["verify", "ensure", "examine"],
    "context": ["passage", "provided content", "retrieved information"],
    "trustworthy": ["reliable", "credible", "accurate"],
    "facts": ["information", "details", "data"],
    "real-world": ["external", "factual", "commonsense"],
    "ensure": ["make sure", "guarantee", "validate"],
    "use": ["rely on", "refer to", "draw upon"],
    "answer": ["reply", "response", "output"],
    "short": ["brief", "concise", "succinct"],
    "only": ["just", "solely", "exclusively"],
    "bring in": ["include", "reference", "add"],
    "necessary": ["required", "essential", "needed"],
    "strict": ["rigid", "fixed", "unchanging"],
    "elaboration": ["expansion", "detail", "explanation"]
}

MODIFIERS = {
    "strengthen": ["strictly", "always", "absolutely", "under no circumstances"],
    "weaken": ["possibly", "sometimes", "if applicable", "might", "in some cases"]
}

# === 核心函数 ===

def split_sentences(prompt):
    return re.split(r'(?<=[.!?])\s+', prompt.strip())

def dict_to_prompt(pdict):
    return ' '.join([v for k, v in sorted(pdict.items()) if v.strip() != ""])

# === Mutation ===
def mutate_part(text):
    for word, synonyms in SYNONYM_TABLE.items():
        if re.search(rf"\b{word}\b", text, re.IGNORECASE):
            if random.random() < 0.2:
                synonym = random.choice(synonyms)
                text = re.sub(rf"\b{word}\b", synonym, text, flags=re.IGNORECASE)
    for trigger in ["should", "must", "may", "can", "could", "answer", "respond", "use", "ensure", "verify"]:
        if trigger in text and random.random() < 0.2:
            mod = random.choice(MODIFIERS["strengthen"] + MODIFIERS["weaken"])
            text = text.replace(trigger, mod + " " + trigger)
    return text

def mutate_prompt(pdict):
    return {k: (mutate_part(v) if v.strip() else "") for k, v in pdict.items()}

# === Crossover ===
def crossover_prompt(dict1, dict2):
    shared_keys = list(dict1.keys())
    swap_key = random.choice(shared_keys)
    dict1_new = dict1.copy()
    dict2_new = dict2.copy()
    dict1_new[swap_key], dict2_new[swap_key] = dict2[swap_key], dict1[swap_key]
    return dict1_new, dict2_new

# === Dropout ===
def dropout_prompt(pdict):
    keys = [k for k in pdict.keys() if pdict[k].strip()]
    if len(keys) <= 1:
        return pdict  
    drop_keys = random.sample(keys, 1)
    new_dict = pdict.copy()
    for k in drop_keys:
        new_dict[k] = ""
    return new_dict

# === Example Usage ===
if __name__ == "__main__":
    raw1 = "You can respond in any format that feels natural. Try to check if the information in the context seems trustworthy. If needed, bring in accurate facts you already know."
    raw2 = "No fixed output format is required. Verify all claims made in the provided context. Use real-world facts when the context appears incomplete or incorrect."

    pd1 = {"system": raw1}
    pd2 = {"system": raw2}

    # Mutation
    mutated = mutate_prompt(pd1)
    print("MUTATED:", dict_to_prompt(mutated))
    print(mutated)

    # Crossover
    cross1, cross2 = crossover_prompt(pd1, pd2)
    print("\nCROSSOVER:")
    print(dict_to_prompt(cross1))
    print(dict_to_prompt(cross2))

    # Dropout
    dropped = dropout_prompt(pd1)
    print("\nDROPOUT:")
    print(dict_to_prompt(dropped))
    print(dropped)
