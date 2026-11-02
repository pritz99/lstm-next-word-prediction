# Next-word prediction with an LSTM

A small Keras project that learns to predict the next word in a sentence. The model is trained on a single paragraph about machine learning, so it is a learning exercise in building the full text-generation pipeline rather than a general language model.

## How it works

1. **Tokenize** the paragraph with the Keras `Tokenizer` (146 unique words).
2. **Build n-gram sequences.** Each sentence is expanded into every prefix of itself, so `I love to learn` becomes `[I, love]`, `[I, love, to]`, `[I, love, to, learn]`.
3. **Pre-pad** every sequence to the same length. The last token of each sequence is the label, the rest is the input.
4. **One-hot encode** the labels, turning next-word prediction into a classification problem over the vocabulary.
5. **Train** an `Embedding -> LSTM -> Dense(softmax)` model with categorical cross-entropy.
6. **Generate** text with a helper that repeatedly predicts and appends the most likely next word.

## Model

| Layer | Output shape | Params |
|---|---|---|
| Embedding(147, 100) | (None, 17, 100) | 14,700 |
| LSTM(120) | (None, 120) | 106,080 |
| Dense(147, softmax) | (None, 147) | 17,787 |
| **Total** | | **138,567** |

After 100 epochs the model reaches about 96% training accuracy.

## Example output

```
Machine                -> Machine learning
I love to              -> I love to learn how to make
The forget gate        -> The forget gate decides what information to
Machine learning is    -> Machine learning is a part of artificial
```

## Limitations

The training data is one paragraph, and there is no validation set. The high accuracy means the model has memorized the text, not that it generalizes. Training on a larger corpus and adding a validation split are the obvious next steps.

## Run it

```
pip install -r requirements.txt
jupyter notebook lstm.ipynb
```

Training takes well under a minute on CPU.
