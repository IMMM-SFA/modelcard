from typing import List, Dict, TypedDict, Optional

from langchain_core.documents import Document
from langgraph.graph import END


class GraphState(TypedDict):
    """
    Represents the state of the graph as it progresses through document
    processing and model card generation.

    Attributes:
        input_urls (Dict[str, str]): A dictionary of input source URLs, keyed by source name.
        github_docs (Optional[List[Document]]): Documents retrieved from a GitHub repository.
        docs_docs (Optional[List[Document]]): Documents retrieved from a Google Docs source.
        extracted_info (Dict): Information extracted from the documents.
        model_card (Optional[Dict]): A dictionary representing the generated model card.
        error_messages (List[str]): A list of error messages encountered during processing.
        validation_issues (Dict): Validation issues detected during processing.
    """
    input_urls: Dict[str, str]
    github_docs: Optional[List[Document]]
    docs_docs: Optional[List[Document]]
    extracted_info: Dict
    model_card: Optional[Dict]
    error_messages: List[str]
    validation_issues: Dict


def decide_next_step(state: GraphState):
    print("--- Decision ---")
    errors = state.get('error_messages', [])
    issues = state.get('validation_issues', {})

    # Check for critical fetch errors first
    if any("fetch failed" in msg.lower() for msg in errors):
         print("Routing to handle_error due to fetch failure.")
         return "error_handler"

    # Check for critical extraction errors (e.g., no vector store)
    if any("vector store initialization failed" in msg.lower() for msg in errors):
         print("Routing to handle_error due to vector store failure.")
         return "error_handler"

    # Check for validation issues from synthesize_validate
    if issues:
        print("Routing to handle_error due to validation issues.")
        return "error_handler"

    # Check if model card dictionary exists (might be None if validation failed badly)
    if not state.get('model_card'):
         print("Routing to handle_error due to missing final model card data.")
         return "error_handler"

    print("Routing to END (Success).")
    return END
