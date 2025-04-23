# Uses the most recent version of LangChain (v0.3)

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

from .agents.card import ModelCard
from .agents.graph import GraphState, decide_next_step
from .agents.extract import information_extraction
from .agents.web_agents import fetch_documentation, fetch_github, fetch_publications
from .agents.validate import synthesize_and_validate, handle_error


api_key = os.getenv("IM3_AZURE_OPENAI_API_KEY", default=None)
endpoint = os.getenv("IM3_AZURE_OPENAI_ENDPOINT", default=None)
deployment = "gpt-4o"
openai_api_version = "2024-02-01"

llm = AzureChatOpenAI(
    model_name="gpt-4o", 
    temperature=0.1, 
    api_key=api_key,
    openai_api_version=openai_api_version,
    azure_deployment=deployment,
    azure_endpoint=endpoint,
)

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    azure_endpoint=endpoint,
    api_key=api_key,
    openai_api_version=openai_api_version,
    chunk_size=512
)

# Initialize search tool (choose one)
search_tool = TavilySearchResults(max_results=5)


if __name__ == "__main__":

    # --- Build the Graph ---
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("fetch_github", fetch_github)
    workflow.add_node("fetch_documentation", fetch_documentation)
    workflow.add_node("extract_info", information_extraction)
    workflow.add_node("fetch_publications", fetch_publications) # Added publication node
    workflow.add_node("synthesize_validate", synthesize_and_validate)
    workflow.add_node("error_handler", handle_error)

    # Define edges
    workflow.set_entry_point("fetch_github")
    workflow.add_edge("fetch_github", "fetch_documentation")
    workflow.add_edge("fetch_documentation", "extract_info")
    workflow.add_edge("extract_info", "fetch_publications") # -> Fetch publications after initial extraction
    workflow.add_edge("fetch_publications", "synthesize_validate") # -> Synthesize after fetching publications

    # Conditional edge after synthesis/validation
    workflow.add_conditional_edges(
        "synthesize_validate",
        decide_next_step,
        {
            "error_handler": "error_handler",
            END: END
        }
    )
    workflow.add_edge("error_handler", END)

    # Compile the graph
    app = workflow.compile()

    # --- Run the Graph ---
    github_url = "https://github.com/IMMM-SFA/mosartwmpy"
    # Point to base URL, let RecursiveUrlLoader handle paths
    docs_url = "https://mosartwmpy.readthedocs.io/en/latest/"

    inputs = {
        "input_urls": {"github": github_url, "docs": docs_url},
        "extracted_info": {}, # Start with empty extracted info
        "error_messages": [],
        "validation_issues": {},
        "model_card": None, # Ensure model_card starts as None
        "github_docs": None, # Ensure docs start as None
        "docs_docs": None,   # Ensure docs start as None
    }

    print("Starting graph execution...")
    # For final result:
    final_state = app.invoke(inputs)

    print("\n--- Graph Execution Finished ---")

    # Check final state for success
    final_card_data = final_state.get("model_card")
    final_issues = final_state.get("validation_issues")
    final_errors = final_state.get("error_messages")

    # Define success more strictly: model_card exists AND no validation issues AND no critical errors?
    # For now, rely on the routing logic: if it reached END without errors/issues routing it away.
    # A simple check is if final_card_data exists and final_issues is empty.
    if final_card_data and not final_issues:
        print("Successfully generated Model Card:")
        try:
            # Create the final Pydantic object from the dictionary state
            final_card = ModelCard(**final_card_data)
            # Use model_dump_json for Pydantic v2
            print(final_card.model_dump_json(indent=2))
        except Exception as e:
            print(f"Error creating/printing final ModelCard object: {e}")
            print("Raw dictionary data:", final_card_data)
    else:
        print("Failed to generate valid Model Card.")
        print("Final State Details:")
        if final_errors:
            print("Errors:", final_errors)
        if final_issues:
            print("Validation Issues:", final_issues)
        
        print("Partial Card Data:", final_card_data) # Uncomment to see potentially invalid data

