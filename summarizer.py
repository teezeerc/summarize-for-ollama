from langchain_community.document_loaders import TextLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OllamaEmbeddings
import argparse
from datetime import datetime


def get_character_splitter(chunk_size, chunk_overlap):
    return CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )


def get_recursive_character_splitter(chunk_size, chunk_overlap):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )


def get_semantic_splitter(breakpoint_threshold_type, embedding_model_base_url, embedding_model_name):
    return SemanticChunker(OllamaEmbeddings(model=embedding_model_name, base_url=embedding_model_base_url),
                           breakpoint_threshold_type=breakpoint_threshold_type)


def get_map_reduce_chain(llm, chat_model_ctx):
    map_prompt = hub.pull("rlm/map-prompt")
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    reduce_prompt = hub.pull("rlm/reduce-prompt")
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )

    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=chat_model_ctx,
    )

    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=True,
    )
    return map_reduce_chain


def get_map_reduce_chain_custom(llm, chat_model_ctx):
    map_template = """The following is a set of documents
    {docs}
    Based on this list of docs, please write a concise but detailed summary.
    Identify all characters and main themes. 
    CONCISE AND DETAILED SUMMARY:"""

    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    reduce_template = """The following is set of summaries for chapters of one book:
    {doc_summaries}
    Take these partial summaries and create unique, consolidated summary and a list of all the themes,
    list of all characters and write what kind of book genre this story is. 
    FINAL DETAILED SUMMARY:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )

    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=chat_model_ctx,
    )

    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=True,
    )
    return map_reduce_chain


def get_refine_chain(llm):
    prompt_template = """Write a concise summary of the following:
    {text}
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        "Your job is to produce a final summary\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary\n"
        "If the context isn't useful, return the original summary.\n"
        "Return ONLY the summary, don't provide any comments regarding summary quality.\n"
    )
    refine_prompt = PromptTemplate.from_template(refine_template)
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        output_key="output_text",
    )
    return chain


def get_splitter(chunk, chunk_size, chunk_overlap, embedding_model_base_url, embedding_model_name):
    match chunk:
        case "recursive":
            return get_recursive_character_splitter(chunk_size, chunk_overlap)
        case "character":
            return get_character_splitter(chunk_size, chunk_overlap)
        case "semantic_percentile":
            return get_semantic_splitter("percentile", embedding_model_base_url, embedding_model_name)
        case "semantic_standard_deviation":
            return get_semantic_splitter("standard_deviation", embedding_model_base_url, embedding_model_name)
        case "semantic_interquartile":
            return get_semantic_splitter("interquartile", embedding_model_base_url, embedding_model_name)
        case "semantic_gradient":
            return get_semantic_splitter("gradient", embedding_model_base_url, embedding_model_name)


def run_chain(algo, split_docs, llm, chat_model_ctx):
    match algo:
        case "map-reduce":
            return get_map_reduce_chain(llm, chat_model_ctx).invoke(split_docs)
        case "map-reduce-custom":
            return get_map_reduce_chain_custom(llm, chat_model_ctx).invoke(split_docs)
        case "refine":
            return get_refine_chain(llm).invoke(split_docs)


if __name__ == "__main__":
    start_time = datetime.now()
    print(f'Start {start_time}')
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True, help="input text file path")
    parser.add_argument("--out-file", type=str, required=True, help="output text file path")
    parser.add_argument("--chunk", type=str, required=False, default="semantic_gradient",
                        choices=['recursive', 'character', 'semantic_percentile', 'semantic_standard_deviation',
                                 'semantic_interquartile', 'semantic_gradient'],
                        help="text chunking algorithm, default is 'semantic_gradient'")
    parser.add_argument("--algo", type=str, required=False, default="refine",
                        choices=['refine', 'map-reduce', 'map-reduce-custom'],
                        help="text summarization algorithm, default is 'refine'")
    parser.add_argument("--chat-model-base-url", type=str, required=False, default="http://localhost:11434/",
                        help="url for chat model, default is http://localhost:11434/")
    parser.add_argument("--chat-model-name", type=str, required=False, default="TeeZee/gemma-2-9b-it-abliterated",
                        help="chat model name, default is 'TeeZee/gemma-2-9b-it-abliterated'")
    parser.add_argument("--chat-model-ctx", type=int, required=False, default=4096,
                        help="chat model context length, default is 4096 tokens")
    parser.add_argument("--chat-model-predict", type=int, required=False, default=2500,
                        help="chat model predict tokens number, default is 2500 tokens")
    parser.add_argument("--embedding-model-base-url", type=str, required=False, default="http://localhost:11434/",
                        help="url for embedding model, default is http://localhost:11434/")
    parser.add_argument("--embedding-model-name", type=str, required=False, default="nomic-embed-text",
                        help="embedding model name, default is 'nomic-embed-text'")
    parser.add_argument("--chunk-size", type=int, required=False, default=8000,
                        help="chunk size in characters for 'recursive', 'character' chunking algorithms, default is 8000 characters")
    parser.add_argument("--chunk-overlap", type=int, required=False, default=200,
                        help="chunks overlap in characters for 'recursive', 'character' chunking algorithms, default is 200 characters")

    args = parser.parse_args()
    in_file = args.in_file
    out_file = args.out_file
    chunk = args.chunk
    algo = args.algo
    chat_model_base_url = args.chat_model_base_url
    chat_model_name = args.chat_model_name
    chat_model_ctx = args.chat_model_ctx
    chat_model_predict = args.chat_model_predict

    embedding_model_base_url = args.embedding_model_base_url
    embedding_model_name = args.embedding_model_name

    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap

    loader = TextLoader(in_file)
    docs = loader.load()

    text_splitter = get_splitter(chunk, chunk_size, chunk_overlap, embedding_model_base_url, embedding_model_name)
    split_docs = text_splitter.split_documents(docs)
    print("No of chunks:", len(split_docs))

    llm = ChatOllama(temperature=0, model=chat_model_name, base_url=chat_model_base_url, num_ctx=chat_model_ctx,
                     num_predict=chat_model_predict)
    result = run_chain(algo, split_docs, llm, chat_model_ctx)

    with open(out_file, "w", encoding="utf-8") as output:
        output.write(result["output_text"])
    print(f'Duration: {datetime.now() - start_time}')
