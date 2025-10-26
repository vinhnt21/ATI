---
title: Natural Language Processing (NLP)
markmap:
  maxWidth: 450
  initialExpandLevel: 2
---

# Natural Language Processing (NLP)

## Introduction
- **Natural Language Processing (NLP)**
  - Branch of Artificial Intelligence
  - Concerned with providing computers the ability to understand texts and human language
- NLP Subfields
  - **Natural Language Understanding (NLU)**
    - Focuses on semantic analysis
    - Determining the intended meaning of text
  - **Natural Language Generation (NLG)**
    - Focuses on text generation by a machine
- **Speech Recognition**
  - Separate from but often used with NLP
  - Parses spoken language into words
  - Turns sound into text and vice versa

---

## Why Does NLP Matter?
- **Everyday Life Integration**
  - Integral part of daily life
  - Applied across diverse fields
  - Retail (customer service chatbots)
  - Medicine (interpreting electronic health records)
- **Conversational Agents**
  - [Amazon Alexa](https://www.amazon.science/blog/alexa-ais-natural-language-understanding-papers-at-icassp-2022)
    - Listens to user queries
    - Finds answers
  - [Apple Siri](https://machinelearning.apple.com/research/hey-siri)
    - Utilizes NLP for voice interaction
- **Advanced Systems**
  - **ChatGPT**
    - Recently opened for [commercial applications](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models?tabs=python-secure%2Cglobal-standard%2Cstandard-chat-completions)
    - Generates sophisticated prose
    - Powers chatbots capable of coherent conversation
- **Industry Applications**
  - **Google**
    - Uses NLP to [improve its search results](https://blog.google/products/search/introducing-mum/)
  - **Facebook**
    - Employs it to [detect and filter hate speech](https://ai.facebook.com/blog/how-facebook-uses-super-efficient-ai-models-to-detect-hate-speech/)
- **Current Challenges**
  - Bias in NLP systems
  - Incoherence issues
  - Unpredictable behavior
  - Still central to modern AI applications

---

## Common Tasks in NLP
- **Text Classification**
  - Assigning class labels
  - Sentiment analysis
  - Spam detection
  - Abusive content detection
- **Text Summarization / Reading Comprehension**
  - Generating concise summaries of documents
- **Speech Recognition**
  - Converting spoken language to text
- **Machine Translation**
  - Translating text between languages
- **Part-of-Speech (PoS) Tagging**
  - Marking words as nouns, verbs, etc.
- **Question Answering**
  - Producing answers to natural-language questions
- **Dialog Generation**
  - Generating conversational responses
- **Text Generation**
  - Completing sentences or paragraphs

---

# Applications of NLP

## Sentiment Analysis
- **Definition**
  - Classifies the emotional intent of text
  - Usually as positive, negative, or neutral
- **Methods**
  - Uses features like n-grams
  - TF-IDF
  - Deep learning models
- **Applications**
  - Customer review classification
  - [Detecting signs of mental illness](https://deeplearning.ai/the-batch/online-clues-to-mental-illness/)
- **Visual Example**
  - ![Sentiment Analysis](https://lms.fit.hanu.vn/pluginfile.php/1/core_h5p/content/3069/images/image-uukf4HJ7.png)

---

## Toxicity Classification
- **Definition**
  - Goes beyond sentiment to detect hostility
  - Detects specific toxic categories
- **Toxic Categories**
  - Threats
  - Insults
  - Obscenities
  - Hate speech
- **Applications**
  - Helps moderate online spaces
  - [Silencing offensive comments](https://deeplearning.ai/the-batch/news-haters-gonna-mute/)
  - [Detecting hate speech](https://deeplearning.ai/the-batch/outing-hidden-hatred/)
  - [Scanning for defamation](https://deeplearning.ai/the-batch/double-check-for-defamation/)

---

## Machine Translation
- **Definition**
  - Automatically translates text between languages
- **Famous Example**
  - [Google Translate](https://ai.googleblog.com/2020/06/recent-advances-in-google-translate.html)
- **Applications**
  - Improves cross-language communication
  - Used on platforms like Facebook and Skype
  - Can [distinguish between similar words](https://deeplearning.ai/the-batch/choosing-words-carefully/)

---

## Named Entity Recognition (NER)
- **Definition**
  - Extracts entities from text
- **Entity Types**
  - Names
  - Organizations
  - Locations
- **Applications**
  - [Summarizing news](https://deeplearning.ai/the-batch/ai-makes-headlines/)
  - [Combating disinformation](https://deeplearning.ai/the-batch/propaganda-watch/)
- **Visual Example**
  - ![NER Example](https://lms.fit.hanu.vn/pluginfile.php/1/core_h5p/content/3069/images/image-V4GfMLaA.png)

---

## Spam Detection
- **Definition**
  - Classifies emails as spam or not spam
- **Benefits**
  - Improves user experience
  - Filters unwanted content
- **Examples**
  - Used in Gmail

---

## Grammatical Error Correction
- **Definition**
  - Models transform ungrammatical text into correct sentences
  - "Sequence-to-sequence" task
- **Applications**
  - [Grammarly](https://www.grammarly.com/blog/how-grammarly-uses-ai/)
  - [Microsoft Word](https://support.microsoft.com/en-us/office/microsoft-editor-checks-grammar-and-more-in-documents-mail-and-the-web-91ecbe1b-d021-4e9e-a82e-abc4cd7163d7)
  - [Grading essays](https://deeplearning.ai/the-batch/smart-students-dumb-algorithms/)

---

## Topic Modeling
- **Definition**
  - Unsupervised task for discovering abstract topics within a corpus
- **Methods**
  - _Latent Dirichlet Allocation (LDA)_
- **Applications**
  - Used commercially for [legal document discovery](https://deeplearning.ai/the-batch/order-in-the-court/)

---

## Text Generation
- **Definition**
  - **Text generation (NLG)** produces human-like text
- **Models Used**
  - [Markov processes](https://mathworld.wolfram.com/MarkovProcess.html)
  - [LSTMs](https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/)
  - [BERT](https://arxiv.org/abs/1810.04805)
  - [GPT-2](https://openai.com/blog/tags/gpt-2/)
  - [LaMDA](https://blog.google/technology/ai/lamda/)
- **Applications**
  - **Autocomplete**
    - Example: Google search
  - **Chatbots**
    - Query databases
    - Generate open-domain conversations
    - Google's LaMDA famously led a developer to [believe it had feelings](https://deeplearning.ai/the-batch/lamda-comes-alive/)

---

## Information Retrieval
- **Definition**
  - Retrieves documents relevant to a query
  - Foundation for search and recommendation systems
- **Processes**
  - **Indexing** (vector-based)
  - **Matching** (similarity-based)
- **Modern Systems**
  - Google Search employs [multimodal retrieval models](https://deeplearning.ai/the-batch/search-goes-multimodal/)
- **Visual Example**
  - ![Information Retrieval](https://lms.fit.hanu.vn/pluginfile.php/1/core_h5p/content/3069/images/image-TiOBVf6H.png)

---

## Summarization
- **Definition**
  - Shortens text while preserving key information
- **Development**
  - Salesforce developed a summarizer that [ensures factual consistency](https://deeplearning.ai/the-batch/keeping-the-facts-straight/)
- **Methods**
  - **Extractive summarization**
    - Selects and combines key sentences
  - **Abstractive summarization**
    - Paraphrases and rewrites content in new words

---

## Question Answering
- **Definition**
  - **Question answering (QA)** generates answers to natural-language queries
- **Famous Example**
  - IBM's [Watson](https://www.ibm.com/ibm/history/ibm100/us/en/icons/watson/) famously won _Jeopardy!_ in 2011
- **Types**
  - **Multiple choice**
    - Select the correct answer from given options
  - **Open domain**
    - Generate answers freely
    - Often using large text databases

---

# NLP Pipeline Overview
- **Visual Overview**
  - ![NLP Pipeline](https://lms.fit.hanu.vn/pluginfile.php/1/core_h5p/content/3069/images/image-T6Tnrmkg.jpg)
- **Three Stages**
  1. **Text Processing**
     - Clean, normalize, and prepare raw text
  2. **Feature Extraction**
     - Create numerical features suitable for modeling
  3. **Modeling**
     - Train and optimize models to make predictions

---

## Preprocessing Text Data
- **Purpose**
  - Text must be converted into numerical form for ML models
- **Typical Steps**
  - **Standardization**
    - Remove punctuation
    - Lowercase text
  - **Tokenization**
    - Split into words, characters, or n-grams
  - **Indexing**
    - Assign numerical indices or embeddings

---

### Text Standardization
- **Standardization Steps**
  - Removing punctuation/non-alphabetic characters
  - Lowercasing text
  - Optionally: spelling correction, stop-word removal, stemming/lemmatization
- **Benefits**
  - Reduces data redundancy
- **Drawbacks**
  - Can remove useful nuances

---

### Tokenization
- **Definition**
  - **Tokenization** divides text into **tokens** (words, subwords, or characters)
- **Types**
  - **Character-level**
    - Each letter is a token
  - **Word-level**
    - Each word is a token (most common)
  - **Subword-level**
    - Split complex words into meaningful parts
  - **N-gram**
    - Group consecutive words (e.g., bigrams, trigrams)
- **Example Illustration**
  - ![Tokenization Example](https://lms.fit.hanu.vn/pluginfile.php/1/core_h5p/content/3069/images/image-oJVDhLSH.png)

---

# Text Preprocessing with Keras Tokenizer
- **Overview**
  - Keras provides a convenient `Tokenizer` for text standardization, tokenization, and indexing
- **Main Arguments**
  - `num_words`: max number of words to keep
  - `filters`: characters to remove
  - `lower`: lowercase text
  - `split`: separator (default = space)
  - `char_level`: tokenization at character level (default = False)
  - `oov_token`: handles Out-of-Vocabulary words

---

### Character-Level Tokenization
- **Code Example**
  ```python
  from keras.preprocessing.text import Tokenizer
  
  sentence = ['TensorFlow is a Machine Learning framework']
  tokenizer = Tokenizer(num_words=1000, char_level=True)
  tokenizer.fit_on_texts(sentence)
  ```
- **Word Index Output**
  ```python
  print(tokenizer.word_index)
  # {' ': 1, 'e': 2, 'n': 3, 'r': 4, ...}
  ```
- **Sequence Output**
  ```python
  print(tokenizer.texts_to_sequences(sentence))
  # [[13, 2, 3, 8, 6, 4, 9, 10, 6, 11, ...]]
  ```
- **Note**
  - Character-level tokenization is rarely used
  - It doesn't capture word-level meaning

---

### Word-Level Tokenization
- **Code Example**
  ```python
  sentences = [
    'TensorFlow is a Machine Learning framework.',
    'Keras is a well designed deep learning API!',
    'Keras is built on top of TensorFlow!'
  ]
  
  tokenizer = Tokenizer(num_words=1000)
  tokenizer.fit_on_texts(sentences)
  print(tokenizer.word_index)
  ```
- **Word Index Output**
  ```python
  # {'is': 1, 'tensorflow': 2, 'a': 3, 'learning': 4, ...}
  ```
- **Result**
  - Converts each word into an index
  - Ignores punctuation

---

### Handling Out-of-Vocabulary Words
- **Code Example**
  ```python
  tokenizer = Tokenizer(num_words=1000, oov_token='Word Out of Vocab')
  tokenizer.fit_on_texts(sentences)
  ```
- **Testing with New Sentences**
  ```python
  new_sentences = ['I like TensorFlow', 'Keras is a superb deep learning API']
  print(tokenizer.texts_to_sequences(new_sentences))
  # [[1, 1, 3], [6, 2, 4, 1, 11, 5, 12]]
  ```
- **Result**
  - Unknown words are replaced by the `oov_token`

---

### Padding Sequences
- **Purpose**
  - Neural networks require inputs of uniform length
  - Use Keras' `pad_sequences()` to pad shorter sequences with zeros
- **Code Example**
  ```python
  from keras.preprocessing.sequence import pad_sequences
  
  tokenized = tokenizer.texts_to_sequences(sentences)
  padded = pad_sequences(tokenized, maxlen=10)
  print(padded)
  ```

---

# Word Representation Models
- **Two Major Categories**
  1. **Set models**
     - Treat text as an unordered bag of words
     - Example: _Bag-of-Words_
  2. **Sequence models**
     - Preserve word order
     - Examples: RNNs, Transformers
- **Modern Approach**
  - Modern NLP relies on **sequence models**
  - Especially **Transformers**

---

## Bag-of-Words Models
- **Definition**
  - **Bag-of-Words** models ignore word order
  - Treat text as a multiset of words or N-grams
- **Example**
  - Spam vs. Non-spam classification using frequency of words
  - Words like "cheap," "buy," "stock" more frequent in spam
- **Visual Example**
  - ![Bag-of-Words](https://lms.fit.hanu.vn/pluginfile.php/1/core_h5p/content/3069/images/image-58keebEF.png)

---

# Sequence Models
- **Definition**
  - **Sequence models** process entire sequences while preserving order
- **Word Representation**
  - Each word is mapped to a **vector representation**
- **Types of Vector Representations**
  - **One-hot vectors**
  - **Word embeddings**

---

## One-Hot Word Vectors
- **Definition**
  - Each word is represented as a binary vector
  - One element is "hot" (1) and the rest are "cold" (0)
- **Visual Example**
  - ![One-Hot Example](https://lms.fit.hanu.vn/pluginfile.php/1/core_h5p/content/3069/images/image-USbNRGb3.png)
- **Limitations**
  - Inefficient for large vocabularies
  - Vectors become huge and sparse

---

## Word Embeddings
- **Definition**
  - **Word embeddings** represent each word as a dense vector
  - Similar words have similar spatial positions
- **Technical Details**
  - Typical dimensions: **256–1024**
  - Similarity between words measured via **cosine similarity**
- **Visual Example**
  - ![Word Embedding Vector](https://lms.fit.hanu.vn/pluginfile.php/1/core_h5p/content/3069/images/image-7T5M9SHP.png)
- **Importance**
  - Embeddings capture semantic meaning
  - Form the foundation of modern NLP
