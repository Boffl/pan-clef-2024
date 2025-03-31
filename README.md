# pan-clef-2024 Oppositional Thinking Analysis
## Team auxR
This is the code for our submission to task one of the shared task on <a hrem="https://pan.webis.de/clef24/pan24-web/oppositional-thinking-analysis.html">Oppositional Thinking Analysis</a>. The Task involved training models to ditinguish conspiracy theories frome critical views on a bilingual (Spanish and English) dataset.

Our best performing approach was as follows: We augmented the training data by using a Machine translation system to translate the Spanish and English data into the other language. We fine-tuned twitter-XLM-RoBERTa-large on the bilingual training data, comprising the Spanish, English and translated datapoints. The run we submitted is an ensemble prediction from 3 models that were trained using different random initializations.

Our best performing models ranked 10th (out of 82) on the Enlgish dataset and 2nd (out of 77) on the Spanish dataset.
