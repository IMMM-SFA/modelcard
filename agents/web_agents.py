import logging
import os
import re
import shutil
import traceback

from typing import List

import git
from langchain_community.document_loaders import GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, RecursiveUrlLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


from .graph import GraphState


def fetch_github(
        state: GraphState,
        repository_clone_directory: str = "./data/temp_repo_clone",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        file_extensions: List[str] = None,
):
    """
    Fetch and process documents from a GitHub repository.

    This function attempts to clone or fetch updates from a specified GitHub
    repository, then loads and splits documents from it using the
    RecursiveCharacterTextSplitter.

    Args:
        state (GraphState): The current graph state containing input URLs and error messages.
        repository_clone_directory (str): Directory path to clone or fetch the repository into.
        chunk_size (int): The maximum number of characters per document chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.
        file_extensions (List[str], optional): List of file extensions to include (lowercase, with leading dot).
            Defaults to ['.txt', '.md', '.rst', '.py', '.png', '.jpg', '.svg', 
            '.cff', '.json', '.yaml', '.sh', '.cfg', '.config', '.ipynb'].

    Returns:
        dict: A dictionary with the processed GitHub documents under "github_docs"
              and any encountered error messages under "error_messages".
    """
    logging.info("--- Node: fetch_github ---")

    if file_extensions is None:
        file_extensions = [
            '.txt', '.md', '.rst', '.py', '.png', '.jpg', '.svg',
            '.cff', '.json', '.yaml', '.sh', '.cfg', '.config', '.ipynb'
        ]

    github_url = state['input_urls'].get('github')
    repo_path = repository_clone_directory
    errors = state.get('error_messages', []) or [] # Ensure errors list exists
    docs = None
    raw_docs = None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


    if not github_url:
        errors.append("Missing GitHub URL.")
        return {"github_docs": None, "error_messages": errors}

    try:
        if os.path.exists(repository_clone_directory):
            try:
                logging.info(f"Repository directory exists at {repository_clone_directory}. Attempting git fetch...")

                repo = git.Repo(repository_clone_directory)

                if not repo.remotes:
                     raise git.InvalidGitRepositoryError(f"{repository_clone_directory} exists but has no remotes.")
                
                logging.info("Fetching updates from all remotes (--all --prune)...")
                repo.git.fetch('--all', '--prune')

                logging.info("Git fetch complete.")
                loader = GitLoader(repo_path=repository_clone_directory)

                raw_docs = loader.load()

            except git.InvalidGitRepositoryError as e:
                 logging.error(f"Existing directory {repository_clone_directory} is not a valid Git repository: {e}. Re-cloning...")
                 errors.append(f"InvalidGitRepositoryError for existing dir: {str(e)}")
                 shutil.rmtree(repository_clone_directory)
                 raw_docs = None

            except git.GitCommandError as e:
                logging.error(f"Git fetch error: {e}")
                errors.append(f"Git fetch failed: {str(e)}. Check permissions/repo state.")
                logging.info("Attempting to load documents from current state despite fetch error...")

                try:
                    loader = GitLoader(repo_path=repository_clone_directory)
                    raw_docs = loader.load()

                except Exception as load_e:
                    logging.error(f"Failed to load docs after fetch error: {load_e}")
                    errors.append(f"Failed to load docs after GitCommandError: {str(load_e)}")
                    raw_docs = None

            except Exception as e:
                logging.error(f"Accessing existing repo failed: {e}. Attempting re-clone...")
                errors.append(f"Error accessing existing repo: {str(e)}")

                try:
                    shutil.rmtree(repository_clone_directory)

                except OSError as rm_e:
                    logging.error(f"Failed to remove problematic directory: {rm_e}")
                    errors.append(f"Failed to remove problematic directory: {str(rm_e)}")

                raw_docs = None

        # If the directory didn't exist OR we decided to re-clone
        if not os.path.exists(repository_clone_directory) or raw_docs is None and github_url:
             try:
                logging.info(f"Cloning repository {github_url} to {repository_clone_directory}...")
                loader = GitLoader(clone_url=github_url, repo_path=repository_clone_directory)
                raw_docs = loader.load()
                logging.info("Clone complete.")
                
             except Exception as e:
                 logging.error(f"Cloning GitHub repo failed: {e}")
                 errors.append(f"GitHub clone failed: {str(e)}")
                 raw_docs = None

        # Split documents if loading/cloning was successful
        if raw_docs:
            # Filter raw_docs by allowed file extensions
            filtered_docs = []
            for doc in raw_docs:
                source = doc.metadata.get('source', '')
                _, ext = os.path.splitext(source)
                if ext.casefold() in file_extensions:
                    filtered_docs.append(doc)
            docs = text_splitter.split_documents(filtered_docs)
            logging.info(f"Processed {len(docs)} docs from GitHub.")

        else:
             logging.warning("No documents loaded or processed from GitHub due to errors.")
             docs = None

    except Exception as e:

        tb_str = traceback.format_exc()
        logging.critical(f"CRITICAL ERROR in fetch_github: {e}\n{tb_str}")
        errors.append(f"Critical fetch_github error: {str(e)}")
        docs = None

    return {"github_docs": docs, "error_messages": errors}


def fetch_documentation(
        state: GraphState,
        timeout: int = 30,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        max_depth: int = 3,
    ):
    """
    Fetch and process documents from a documentation URL.

    This function loads HTML documents recursively from the specified Docs URL,
    then splits them into smaller chunks using the RecursiveCharacterTextSplitter.

    Args:
        state (GraphState): The current graph state containing input URLs and error messages.
        timeout (int): Timeout in seconds for loading the URL. Defaults to 30.
        chunk_size (int): Maximum number of characters per chunk. Defaults to 1000.
        chunk_overlap (int): Number of overlapping characters between chunks. Defaults to 150.
        max_depth (int): Maximum depth for recursive URL loading. Defaults to 3.

    Returns:
        dict: A dictionary with the processed documents under "docs_docs" and any
              encountered error messages under "error_messages".
    """
    logging.info("--- Node: fetch_documentation ---")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    docs_url = state['input_urls'].get('docs')
    errors = state.get('error_messages', []) or []
    docs = None

    if docs_url:
        try:
            loader = RecursiveUrlLoader(
                url=docs_url, 
                max_depth=max_depth, 
                prevent_outside=True,
                use_async=False, # Force synchronous to avoid event loop issues
                timeout=timeout,
            )
            raw_docs = loader.load()
            docs = text_splitter.split_documents(raw_docs)
            
            logging.info(f"Fetched and split {len(docs)} docs from Docs URL.")

        except Exception as e:
            logging.error(f"ERROR fetching Docs URL: {e}")
            tb_str = traceback.format_exc()
            errors.append(f"Docs URL fetch failed: {str(e)}\nTraceback: {tb_str}")
    else:
        errors.append("Missing Docs URL.")

    return {"docs_docs": docs, "error_messages": errors}


def fetch_publications(
        state: GraphState,
        max_context_length: int = 16000,
        max_chars_per_query = 2000,
    ):
    """
    Fetch key publications related to a software capability.

    This function performs a web search using predefined queries to identify key
    publications related to the specified software capability. The search results
    are passed through an LLM to extract a representative citation string. If a
    DOI is found within the citation, it is also extracted and stored.

    Args:
        state (GraphState): The current graph state containing extracted information
            and error messages.
        max_context_length (int): Maximum number of characters to pass into the LLM. Defaults to 16000.
        max_chars_per_query (int): Maximum number of characters to use from each search result. Defaults to 2000.

    Returns:
        dict: A dictionary with updated 'extracted_info' and a list of 'error_messages'.
    """
    logging.info("--- Node: fetch_publications ---")

    extracted = state.get('extracted_info', {})
    errors = state.get('error_messages', []) or []
    capability_name = extracted.get('capability_name')

    # Access global search tool (ensure initialized)
    global search_tool, llm
    if 'search_tool' not in globals() or search_tool is None:
        errors.append("CRITICAL: Search tool not initialized.")
        return {"extracted_info": extracted, "error_messages": errors}
    
    if 'llm' not in globals() or llm is None:
        errors.append("CRITICAL: LLM not initialized for publication parsing.")
        return {"extracted_info": extracted, "error_messages": errors}


    if not capability_name:
        logging.info("Skipping publication fetch: capability_name not available.")
        return {"extracted_info": extracted, "error_messages": errors}

    try:
        logging.info(f"Searching publications related to: {capability_name}")
        query1 = f"most important publication describing {capability_name} software"
        query2 = f"\"Journal of Open Source Software\" {capability_name}"
        query3 = f"citation OR paper {capability_name} model"
        search_queries = [query1, query2, query3]

        all_results_text = f"Search results for '{capability_name}':\n\n"

        try:
             for q in search_queries:
                  logging.info(f"  Executing search: {q}")
                  results = search_tool.run(q) # Assumes search_tool.run exists and returns string
                  all_results_text += f"--- Results for query: {q} ---\n{results[:max_chars_per_query]}\n\n"
        
        except Exception as search_e:
             logging.warning(f"Search query failed: {search_e}")
             errors.append(f"Search query failed: {str(search_e)}")

        if not all_results_text.strip():
             logging.warning("No search results obtained.")
             return {"extracted_info": extracted, "error_messages": errors}

        parser_prompt_template = """
        You are an expert academic researcher identifying key software publications.
        Based on the following software name and web search results, identify the single most important publication describing the software itself.
        Prioritize papers published in software journals (like JOSS) or papers explicitly introducing the model/software. Look for clear Title, Authors, Journal/Venue, Year, and DOI.

        Format the output as a single citation string suitable for a 'key_publications' field (e.g., "Title. Authors (First Author Last, Second Author Last). Journal Vol(Issue), Pages, Year. DOI: XXXXX").
        If multiple versions or papers exist, prefer the primary software description paper.
        If no single key publication clearly describes the software based *only* on the provided snippets, output the exact phrase "No key publication identified.".

        Software Name: {capability_name}

        Web Search Results Snippets:
        -------
        {search_results_text}
        -------

        Key Publication String:
        """
        parser_prompt = ChatPromptTemplate.from_template(parser_prompt_template)
        parser_chain = parser_prompt | llm | StrOutputParser()

        logging.info("Asking LLM to identify key publication from search results...")
        llm_input_text = all_results_text[:max_context_length]

        key_pub_string = parser_chain.invoke({
            "capability_name": capability_name, "search_results_text": llm_input_text
        })
        logging.info(f"LLM identified: {key_pub_string}")

        if key_pub_string and "No key publication identified." not in key_pub_string:
            extracted['key_publications'] = key_pub_string.strip()
            doi_match = re.search(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', key_pub_string, re.IGNORECASE)

            if doi_match:
                found_doi = doi_match.group(1)
                
                if not extracted.get('doi'):
                    extracted['doi'] = found_doi
                    logging.info(f"Extracted DOI ({found_doi}) and updated 'doi' field.")
        else:
            logging.info("No specific key publication identified by LLM.")

    except Exception as e:
        tb_str = traceback.format_exc()
        logging.error(f"ERROR during publication fetch node: {e}\n{tb_str}")
        errors.append(f"Publication fetch node error: {str(e)}")

    return {"extracted_info": extracted, "error_messages": errors}
