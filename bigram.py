from collections import Counter

def preprocess(sentence):
    sentence = sentence.lower().strip()
    sentence = sentence.replace(".", "")
    return sentence.split()

def generate_bigrams(tokens):
    return list(zip(tokens, tokens[1:]))

with open("NLP_Bigram_Corpus.txt", "r", encoding="utf-8") as file:
    corpus = [line.strip() for line in file if line.strip()]

bigram_counts = Counter()
word_counts = Counter()
vocabulary = set()

for sentence in corpus:
    tokens = preprocess(sentence)
    tokens = ["<s>"] + tokens + ["</s>"]

    vocabulary.update(tokens)

    for word in tokens:
        word_counts[word] += 1

    bigrams = generate_bigrams(tokens)

    for bigram in bigrams:
        bigram_counts[bigram] += 1

vocabulary_size = len(vocabulary)

print("\n========== BIGRAM FREQUENCY ==========\n")

for bigram, count in bigram_counts.items():
    print(f"{bigram[0]:15} {bigram[1]:15} {count}")

print("\n========== BIGRAM PROBABILITIES ==========\n")

bigram_probabilities = {}

for bigram, count in bigram_counts.items():
    previous_word = bigram[0]
    probability = count / word_counts[previous_word]
    bigram_probabilities[bigram] = probability

    print(
        f"P({bigram[1]} | {bigram[0]}) = "
        f"{count}/{word_counts[previous_word]} = "
        f"{probability:.6f}"
    )

def sentence_probability(sentence):
    tokens = preprocess(sentence)
    tokens = ["<s>"] + tokens + ["</s>"]

    probability = 1.0
    zero_found = False

    for bigram in generate_bigrams(tokens):
        count = bigram_counts.get(bigram, 0)
        previous_count = word_counts.get(bigram[0], 0)

        if count == 0 or previous_count == 0:
            zero_found = True
            probability = 0
            break

        probability *= count / previous_count

    return probability, zero_found

def smoothed_sentence_probability(sentence):
    tokens = preprocess(sentence)

    probability = 1.0

    for bigram in generate_bigrams(["<s>"] + tokens + ["</s>"]):
        count = bigram_counts.get(bigram, 0)
        previous_count = word_counts.get(bigram[0], 0)

        probability *= (count + 1) / (previous_count + vocabulary_size)

    return probability

with open("NLP_Bigram_Test_Sentences.txt", "r", encoding="utf-8") as file:
    test_sentences = [line.strip() for line in file if line.strip()]

print("\n========== TEST SENTENCE PROBABILITIES ==========\n")

results = []

for i, sentence in enumerate(test_sentences, 1):
    unsmoothed, zero_found = sentence_probability(sentence)
    smoothed = smoothed_sentence_probability(sentence)

    results.append((sentence, unsmoothed, smoothed, zero_found))

    print(f"Sentence {i}: {sentence}")
    print(f"Unsmoothed Probability : {unsmoothed:.12f}")
    print(f"Laplace Probability    : {smoothed:.12f}")

    if zero_found:
        print("Zero-frequency bigram  : YES")
    else:
        print("Zero-frequency bigram  : NO")

    print()

print("\n========== FINAL COMPARISON ==========\n")

print(f"{'Sentence':<8} {'Unsmoothed':<20} {'Laplace Smoothed':<20}")
print("-" * 50)

for i, result in enumerate(results, 1):
    print(
        f"{i:<8} "
        f"{result[1]:<20.12f} "
        f"{result[2]:<20.12f}"
    )

print("\nVocabulary Size:", vocabulary_size)