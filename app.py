from typing import List, Optional, Union

from pydantic import BaseModel


class ModelCard(BaseModel):
    """
    A model card specifying metadata and characteristics of a scientific capability.

    Attributes
    ----------
    capability_name : str
        The name of the model, dataset, or tool.
    current_version : str
        The current version of the capability.
    license : str
        The name of the license associated with the capability.
    contact_name : str
        Name of the contact person for the capability.
    contact_email : str
        Email of the contact person for the capability.
    key_contributors : List[str]
        List of contributors to the capability.
    doi : Optional[str]
        The DOI of the capability if available.
    computational_requirements : str
        Either "HPC" for high performance computing if required,
        "Laptop", or "None specified".
    category : List[str]
        Select from Atmosphere, Physical Hydrology, Water Management,
        Wildfire, Energy, Multisectoral, Land Use Land Cover,
        Socioeconomics.
    systems_covered : List[str]
        High-level list of systems.
    spatial_resolution : Union[List[str], str, float]
        Range of tested grid spacing or distance between resolved features.
    geographic_scope : Union[List[str], str]
        Where it is currently applied and where it could be applied.
    temporal_resolution : Union[List[str], str]
        E.g., seconds, hours, days, months, years.
    temporal_range : Union[List[str], str]
        E.g., historical, near-future, far-future, specific period.
    input_variables : Union[List[str], str]
        High-level list of input variable categories.
    output_variables : Union[List[str], str]
        High-level list of output variable categories.
    interdependencies : Optional[Union[List[str], str]]
        E.g., E3SM and GCAM, Demeter and Tethys.
    key_publications : Union[List[str], str]
        Most recent publication describing key capabilities.
    brief_description : str
        Typical uses and current users.
    figure : str
        Splash figure from documentation.
    figure_caption : str
        A caption for the splash figure.
    """
    capability_name: str
    current_version: str
    license: str
    contact_name: str
    contact_email: str
    key_contributors: List[str]
    doi: Optional[str]
    computational_requirements: str
    category: List[str]
    systems_covered: List[str]
    spatial_resolution: Union[List[str], str, float]
    geographic_scope: Union[List[str], str]
    temporal_resolution: Union[List[str], str]
    temporal_range: Union[List[str], str]
    input_variables: Union[List[str], str]
    output_variables: Union[List[str], str]
    interdependencies: Optional[Union[List[str], str]]
    key_publications: Union[List[str], str]
    brief_description: str
    figure: str
    figure_caption: str
