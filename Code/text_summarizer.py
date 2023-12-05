import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained('t5-large')
tokenizer = T5Tokenizer.from_pretrained('t5-large')
device = torch.device('cuda')
model = model.to(device)

def summarize(text, ml):
    preprocess_text = text.strip().replace("\n", "")
    t5_prepared_Text = "summarize: " + preprocess_text
    print("Preprocessed and prepared text: \n", t5_prepared_Text)
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
    summary_ids = model.generate(tokenized_text,
                                 num_beams=4,
                                 no_repeat_ngram_size=2,
                                 min_length=30,
                                 max_length=ml,
                                 early_stopping=True)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output


text = """PATERSON, NJ—Family sources told reporters Tuesday that local mother Karen Burkhart came fairly close to using the term “streaming” correctly during a recent conversation. “She likes that Orange Is The New Black show and told us that she started ‘stream-watching’ a couple of episodes,” said daughter Melanie Burkhart, who was reportedly surprised by her mother’s nearly accurate usage of the technical jargon. “I thought, wow, she actually got really close to the actual meaning of the word. She almost nailed it. And this is the woman who asked if the internet was as good as the online. It’s certainly the most precise she’s been in a long time.” At press time, Burkhart’s children decided to give it to her, claiming that she got close enough."""
print("Originial Text:", text)
print("Number of characters:", len(text))
summary = summarize(text, 20)
print("\n\nSummarized text: \n", summary)