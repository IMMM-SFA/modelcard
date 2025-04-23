from pydantic import ValidationError

from .card import ModelCard
from .graph import GraphState


def handle_error(state: GraphState):
    print("--- Node: handle_error ---")
    print("Errors occurred during processing:")
    for msg in state.get('error_messages', []):
        print(f"- {msg}")
    issues = state.get('validation_issues', {})
    if issues:
        print("Validation Issues:")
        for k, v in issues.items():
            print(f"- {k}: {v}")
    return {}


def synthesize_and_validate(
        state: GraphState
    ):
    """
    Synthesize and validate extracted information for a model card.

    This function performs type coercion, vocabulary checks, and Pydantic-based
    validation on extracted data to produce a validated model card dictionary.
    It handles list coercion, required field enforcement, allowed value checks,
    and applies default values for missing fields before validation.

    Args:
        state (GraphState): The current graph state containing extracted information
            and error messages.

    Returns:
        dict: A dictionary containing:
            - 'model_card': The validated model card as a dictionary (may be partially validated).
            - 'validation_issues': Any issues encountered during validation.
            - 'error_messages': Any error messages encountered during processing.
    """

    print("--- Node: synthesize_and_validate ---")

    extracted = state.get('extracted_info', {})
    issues = {}
    final_data_dict = {} # Start with empty dict
    errors = state.get('error_messages', []) or []

    if not extracted:
        errors.append("No extracted information to synthesize.")
        issues["synthesis"] = "No data available to synthesize or validate."
        return {"model_card": None, "validation_issues": issues, "error_messages": errors}

    # Create a working copy for coercion
    coerced_data = extracted.copy()

    # --- Type Coercion and Validation Logic ---
    # Iterate using Pydantic v2 style
    for field_name, field_info in ModelCard.model_fields.items():
        raw_value = coerced_data.get(field_name)
        # Use Pydantic v2 method to check if required
        is_required = field_info.is_required()

        # --- Type Coercion based on field_info.annotation ---
        # This section needs refinement for robust type handling

        # Example: Coercing list fields (more robust check)
        is_list_field = False
        field_type_origin = getattr(field_info.annotation, "__origin__", None)
        if field_type_origin is list or str(field_info.annotation).startswith("List["):
             is_list_field = True

        if is_list_field:

            if isinstance(raw_value, str):
                # Basic split, assuming comma separation. Needs refinement for other list formats.
                coerced_data[field_name] = [item.strip() for item in raw_value.split(',') if item.strip()]
                print(f"Coerced string to list for {field_name}: {coerced_data[field_name]}")
            # Check for missing required list after potential coercion attempt
            current_value = coerced_data.get(field_name)
            if current_value is None and is_required:
                issues[f"missing_{field_name}"] = f"Required list field '{field_name}' is missing."
            elif not isinstance(current_value, list) and current_value is not None:
                # If it should be a list but isn't after coercion, flag or attempt conversion
                issues[f"type_error_{field_name}"] = f"Expected list for '{field_name}', got {type(current_value)}. Using as single-element list."
                coerced_data[field_name] = [str(current_value)] # Fallback

        # Add specific coercion for other types if needed (e.g., float, specific Unions)
        # elif field_info.annotation is float:
        #     if isinstance(raw_value, str):
        #         try:
        #             coerced_data[field_name] = float(raw_value)
        #         except (ValueError, TypeError):
        #             issues[f"type_error_{field_name}"] = f"Cannot convert '{raw_value}' to float for {field_name}"
        #             coerced_data[field_name] = None # Or some default

        # --- General Check for Missing Required Fields ---
        final_value = coerced_data.get(field_name)
        if final_value is None and is_required:
             # Avoid double-flagging if already caught by list logic etc.
             if f"missing_{field_name}" not in issues:
                  issues[f"missing_{field_name}"] = f"Required field '{field_name}' is missing."

    # --- Vocabulary Validation ---
    allowed_compute = ["HPC", "Laptop", "None specified"]
    compute_req = coerced_data.get('computational_requirements')
    if compute_req is not None and compute_req not in allowed_compute:
        issues['invalid_compute'] = f"Value '{compute_req}' not in {allowed_compute}"
        coerced_data['computational_requirements'] = "None specified" # Fallback

    allowed_categories = ["Atmosphere", "Physical Hydrology", "Water Management", "Wildfire", "Energy", "Multisectoral", "Land Use Land Cover", "Socioeconomics"]
    category_list = coerced_data.get('category')
    if category_list is not None:
        if isinstance(category_list, list):

            valid_cats = [cat for cat in category_list if cat in allowed_categories]
            invalid_cats = [cat for cat in category_list if cat not in allowed_categories]

            if invalid_cats:
                issues['invalid_category'] = f"Invalid categories found: {invalid_cats}"
            coerced_data['category'] = valid_cats # Keep only valid ones

        else:
            issues['type_error_category'] = f"Field 'category' should be a list, got {type(category_list)}."
            # Remove invalid non-list entry for required list field if it exists
            if 'category' in coerced_data: del coerced_data['category']


    # --- Attempt Final Pydantic Validation ---
    try:
        # Ensure only fields defined in ModelCard are passed
        model_data_for_validation = {k: v for k, v in coerced_data.items() if k in ModelCard.model_fields}
        # Fill missing required fields with defaults to allow validation
        for field_name, field_info in ModelCard.model_fields.items():

            if model_data_for_validation.get(field_name) is None:

                # List types default to empty list, others to empty string
                origin = getattr(field_info.annotation, "__origin__", None)

                if origin is list or "List[" in str(field_info.annotation):
                    model_data_for_validation[field_name] = []
                else:
                    model_data_for_validation[field_name] = ""

        # Validate by creating the object
        final_model_card_obj = ModelCard(**model_data_for_validation)

        # If successful, store the validated data as a dictionary
        final_data_dict = final_model_card_obj.model_dump()
        print("Pydantic validation successful.")

    except ValidationError as e:
        print(f"ERROR: Pydantic validation failed: {e}")
        issues['pydantic_validation'] = str(e)
        final_data_dict = coerced_data # Keep coerced data for debugging
        errors.append(f"Final validation failed: {str(e)}")

    except Exception as e:
        print(f"ERROR: Unexpected error during final validation: {e}")
        issues['validation_unexpected_error'] = str(e)
        final_data_dict = coerced_data # Keep coerced data for debugging
        errors.append(f"Unexpected validation error: {str(e)}")

    # Return the potentially validated dict, issues, and errors
    return {"model_card": final_data_dict, "validation_issues": issues, "error_messages": errors}
