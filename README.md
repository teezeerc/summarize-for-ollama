# summarize-for-ollama

Script to summarize text document. Different chunking algorithms and summarizing algorithms are available.

# Installation

- conda create -n summarize-for-ollama python=3.10
- git clone https://github.com/teezeerc/summarize-for-ollama
- cd summarize-for-ollama
- pip install -r requirements.txt

# Simple use case

Run Ollama and download models:
- ollama pull TeeZee/gemma-2-9b-it-abliterated
- ollama pull nomic-embed-text
- ollama serve

Download a sample book:
- curl https://www.gutenberg.org/cache/epub/24022/pg24022.txt > ./book.txt

Run summarizer with local Ollama model for embeddings and uncensored model to create summary:

- python summarizer.py --in-file book.txt --out-file book_summary.txt 

# Available options

- python summarizer.py -h
```
usage: summarizer.py [-h] --in-file IN_FILE --out-file OUT_FILE [--chunk {recursive,character,semantic_percentile,semantic_standard_deviation,semantic_interquartile,semantic_gradient}] [--algo {refine,map-reduce,map-reduce-custom}] [--chat-model-base-url CHAT_MODEL_BASE_URL] [--chat-model-name CHAT_MODEL_NAME]
                     [--chat-model-ctx CHAT_MODEL_CTX] [--chat-model-predict CHAT_MODEL_PREDICT] [--embedding-model-base-url EMBEDDING_MODEL_BASE_URL] [--embedding-model-name EMBEDDING_MODEL_NAME] [--chunk-size CHUNK_SIZE] [--chunk-overlap CHUNK_OVERLAP]

options:
  -h, --help            show this help message and exit
  --in-file IN_FILE     input text file path
  --out-file OUT_FILE   output text file path
  --chunk {recursive,character,semantic_percentile,semantic_standard_deviation,semantic_interquartile,semantic_gradient}
                        text chunking algorithm, default is 'semantic_gradient'
  --algo {refine,map-reduce,map-reduce-custom}
                        text summarization algorithm, default is 'refine'
  --chat-model-base-url CHAT_MODEL_BASE_URL
                        url for chat model, default is http://localhost:11434/
  --chat-model-name CHAT_MODEL_NAME
                        chat model name, default is 'TeeZee/gemma-2-9b-it-abliterated'
  --chat-model-ctx CHAT_MODEL_CTX
                        chat model context length, default is 4096 tokens
  --chat-model-predict CHAT_MODEL_PREDICT
                        chat model predict tokens number, default is 2500 tokens
  --embedding-model-base-url EMBEDDING_MODEL_BASE_URL
                        url for embedding model, default is http://localhost:11434/
  --embedding-model-name EMBEDDING_MODEL_NAME
                        embedding model name, default is 'nomic-embed-text'
  --chunk-size CHUNK_SIZE
                        chunk size in characters for 'recursive', 'character' chunking algorithms, default is 8000 characters
  --chunk-overlap CHUNK_OVERLAP
                        chunks overlap in characters for 'recursive', 'character' chunking algorithms, default is 200 characters
```

# Advanced use cases
- python summarizer.py --in-file book.txt --out-file book_summary1.txt --algo map-reduce --chunk semantic_interquartile
- python summarizer.py --in-file book.txt --out-file book_summary2.txt --algo map-reduce-custom --chunk character
