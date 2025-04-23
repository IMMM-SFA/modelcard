import os

import git
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from .graph import GraphState
from .card import ModelCard


VECTORSTORE_CACHE_DIR = "./data/vectorstore_cache" # Renamed from ./tmp for clarity
FAISS_INDEX_NAME = "faiss_index_mosartwmpy"

# Define allowed values globally for easier access in prompts
ALLOWED_COMPUTE_REQS = ["HPC", "Laptop", "None specified"]
ALLOWED_CATEGORIES = ["Atmosphere", "Physical Hydrology", "Water Management", "Wildfire", "Energy", "Multisectoral", "Land Use Land Cover", "Socioeconomics"]


# Revised Prompt Template Generation Logic
def get_field_extraction_prompt(field_name, field_info):
    """Generates a tailored prompt for extracting a specific field."""
    # Build a clear extraction prompt using the field's docstring description and type hint
    prompt_template = (
        "You are an expert scientific software auditor. Your task is to extract the value for the ModelCard field '{field_name}'.\n\n"
        "Field Definition:\n"
        "- Description: {description}\n"
        "- Expected type: {type_hint}\n\n"
        "Using *only* the context below, locate the value for this field. "
        "Return your answer as a JSON object with a single key '{field_name}'. "
        "If the field is not present or cannot be determined, set its value to null.\n\n"
        "Context:\n"
        "{context}\n\n"
        "JSON Output:"
    )
    # For enumeration or list fields, add explicit allowed values instructions
    if field_name == 'computational_requirements':
        prompt_template += f"\nIMPORTANT: The value must be one of {ALLOWED_COMPUTE_REQS} or null."
    elif field_name == 'category':
        prompt_template += f"\nIMPORTANT: The value must be a JSON list containing only strings from {ALLOWED_CATEGORIES}, or null."
    template = prompt_template
    return PromptTemplate(
        input_variables=["field_name", "description", "type_hint", "context"],
        template=template
    )



# Helper for RAG within the extraction node
def get_relevant_chunks(vectorstore, question, k=5):
    if not vectorstore: return []
    return vectorstore.similarity_search(question, k=k)



def information_extraction(state: GraphState):
    print("--- Node: information_extraction ---")
    # ... (initial setup: get docs, errors, embeddings check - remains the same) ...
    github_docs = state.get('github_docs') or []
    docs_docs = state.get('docs_docs') or []
    all_docs = github_docs + docs_docs
    errors = state.get('error_messages', []) or []
    extracted_info = {}
    vectorstore = None

    # --- Prepopulate basic fields from repository metadata ---
    # Capability name, license, key_contributors
    # Requires: import git, import os
    # (import toml if needed later)
    # Capability name from repo URL
    repo_url = state['input_urls'].get('github', '')
    repo_dir = "./data/temp_repo_clone"
    if repo_url:
        extracted_info['capability_name'] = repo_url.rstrip('/').rstrip('.git').split('/')[-1]
    # License from LICENSE file
    license_path = os.path.join(repo_dir, 'LICENSE')
    if os.path.exists(license_path):
        try:
            with open(license_path, 'r') as lf:
                first_line = lf.read().splitlines()[0].strip()
                extracted_info['license'] = first_line
        except Exception:
            pass
    # Key contributors from git history
    try:
        repo = git.Repo(repo_dir)
        authors = list({c.author.name for c in repo.iter_commits()})
        extracted_info['key_contributors'] = authors
    except Exception:
        extracted_info['key_contributors'] = []

    global embeddings
    if 'embeddings' not in globals() or embeddings is None:
         errors.append("CRITICAL: Embeddings model not initialized.")
         return {"extracted_info": {}, "error_messages": errors}

    if not all_docs:
        errors.append("No documents found. Skipping vector store and RAG.")
        return {"extracted_info": {}, "error_messages": errors}

    # --- Load or Create Vector Store ---
    faiss_index_folder_path = VECTORSTORE_CACHE_DIR
    faiss_file_path_check = os.path.join(faiss_index_folder_path, f"{FAISS_INDEX_NAME}.faiss")
    try:
        # Only create a new vector store if it doesn't already exist
        if os.path.exists(faiss_index_folder_path):
            print(f"Loading existing vector store: {faiss_index_folder_path}/{FAISS_INDEX_NAME}")
            vectorstore = FAISS.load_local(
                folder_path=faiss_index_folder_path,
                embeddings=embeddings,
                index_name=FAISS_INDEX_NAME,
                allow_dangerous_deserialization=True
            )
        else:
            print("Creating new vector store...")
            vectorstore = FAISS.from_documents(all_docs, embeddings)
            os.makedirs(faiss_index_folder_path, exist_ok=True)
            vectorstore.save_local(
                folder_path=faiss_index_folder_path,
                index_name=FAISS_INDEX_NAME
            )
            print(f"Vector store saved: {faiss_index_folder_path}/{FAISS_INDEX_NAME}")

        # --- Proceed with RAG ---
        if vectorstore:
            print("Performing RAG extraction using vector store...")
            retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
            json_parser = JsonOutputParser()

            for field_name, field_info in ModelCard.model_fields.items():
                print(f"  Extracting: {field_name}")
                # Use helper function to get tailored prompt
                prompt = get_field_extraction_prompt(field_name, field_info)
                question = f"Extract '{field_name}' for this scientific software." # Simplified question for retriever

                try:
                    relevant_docs = get_relevant_chunks(vectorstore, question, k=7)
                    if not relevant_docs:
                        print(f"    No relevant docs found for {field_name}.")
                        # Only populate if not already set by prepopulation
                        if field_name not in extracted_info:
                            extracted_info[field_name] = None
                        continue

                    context = "\n---\n".join([doc.page_content for doc in relevant_docs])

                    # Define chain for this field extraction
                    extract_chain = prompt | llm | json_parser

                    # === Debugging Start ===
                    if field_name in ['capability_name', 'computational_requirements', 'category']: # Debug critical/failing fields
                        print(f"    DEBUG: Context for {field_name} (first 500 chars): {context[:500]}")
                    # === Debugging End ===

                    try:
                        # Invoke LLM
                        field_result_raw = extract_chain.invoke({
                            "context": context,
                            "field_name": field_name,
                            "description": field_info.description or "",
                            "type_hint": str(field_info.annotation)
                        })

                        # === Debugging Start ===
                        if field_name in ['capability_name', 'computational_requirements', 'category']:
                             print(f"    DEBUG: Raw LLM Output for {field_name}: {field_result_raw}")
                        # === Debugging End ===

                        # --- Robust Parsing ---
                        field_value = None
                        if isinstance(field_result_raw, dict):
                            # Check if the exact field name key exists
                            if field_name in field_result_raw:
                                field_value = field_result_raw[field_name]
                            else:
                                # If key doesn't match, maybe LLM used a different key? Log it.
                                print(f"    WARNING: Expected key '{field_name}' not found in LLM output dict for {field_name}. Keys found: {list(field_result_raw.keys())}")
                                # Attempt to get the first value from the dict as a fallback? Risky.
                                # field_value = next(iter(field_result_raw.values()), None) if field_result_raw else None
                        else:
                             print(f"    WARNING: Unexpected output type from LLM parser for {field_name}: {type(field_result_raw)}. Expected dict.")
                        # --- End Robust Parsing ---

                        # Handle explicit null/None from LLM output
                        if field_value is None:
                            print(f"    LLM returned null/None for {field_name}.")
                            # Only populate if not already set by prepopulation
                            if field_name not in extracted_info:
                                extracted_info[field_name] = None
                        else:
                             print(f"    Raw value extracted for {field_name}: {field_value}")
                             extracted_info[field_name] = field_value

                    except Exception as invoke_e:

                        if "insufficient_quota" in err_msg.lower():
                            print(f"    WARNING: Insufficient quota for GPT-4 model. Falling back to GPT-3.5-turbo for field {field_name}.")
                            # Retry with fallback model
                            llm_fallback = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
                            extract_chain = prompt | llm_fallback | json_parser
                            try:
                                field_result_raw = extract_chain.invoke({
                                    "context": context,
                                    "field_name": field_name,
                                    "description": field_info.description or "",
                                    "type_hint": str(field_info.annotation)
                                })
                                # print(f"    DEBUG: Raw LLM Output (fallback) for {field_name}: {field_result_raw}")
                                if isinstance(field_result_raw, dict) and field_name in field_result_raw:
                                    extracted_info[field_name] = field_result_raw[field_name]
                                else:
                                    print(f"    WARNING: Fallback LLM output did not contain expected key for {field_name}.")
                                    if field_name not in extracted_info:
                                        extracted_info[field_name] = None
                            except Exception as fallback_e:
                                print(f"    ERROR retrying with fallback model for {field_name}: {fallback_e}")
                                errors.append(f"Fallback LLM invocation error for '{field_name}': {str(fallback_e)}")
                                if field_name not in extracted_info:
                                    extracted_info[field_name] = None
                        else:
                            print(f"    ERROR invoking LLM chain or parsing output for field {field_name}: {invoke_e}")
                            errors.append(f"LLM invocation/parsing error for '{field_name}': {err_msg}")
                            if field_name not in extracted_info:
                                extracted_info[field_name] = None

                except Exception as e:

                    if "insufficient_quota" in err_msg.lower():
                        print(f"    WARNING: Insufficient quota detected before LLM call for field {field_name}. Attempting fallback with GPT-3.5.")
                        llm_fallback = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
                        extract_chain_fallback = prompt | llm_fallback | json_parser
                        try:
                            field_result_raw = extract_chain_fallback.invoke({
                                "context": context,
                                "field_name": field_name,
                                "description": field_info.description or "",
                                "type_hint": str(field_info.annotation)
                            })
                            if isinstance(field_result_raw, dict) and field_name in field_result_raw:
                                extracted_info[field_name] = field_result_raw[field_name]
                            else:
                                print(f"    WARNING: Fallback output did not contain expected key for {field_name}.")
                                if field_name not in extracted_info:
                                    extracted_info[field_name] = None
                        except Exception as fb_e:
                            print(f"    ERROR fallback before LLM for field {field_name}: {fb_e}")
                            errors.append(f"Fallback before LLM error for '{field_name}': {str(fb_e)}")
                            if field_name not in extracted_info:
                                extracted_info[field_name] = None
                    else:
                        print(f"    ERROR processing field {field_name} before LLM call: {e}")
                        errors.append(f"Pre-LLM processing error for '{field_name}': {str(e)}")
                        if field_name not in extracted_info:
                            extracted_info[field_name] = None

            print("RAG extraction complete.")
        # ... (rest of the function: vector store error handling, return) ...
        else:
             print("ERROR: Vector store not available. Cannot perform RAG.")
             errors.append("Vector store initialization failed. RAG skipped.")

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        print(f"CRITICAL ERROR in information_extraction: {e}\n{tb_str}")
        errors.append(f"Critical information_extraction error: {str(e)}")
        extracted_info = {}

    return {"extracted_info": extracted_info, "error_messages": errors}

