from typing import List, Optional, Union

from pydantic import BaseModel, Field


class ModelCard(BaseModel):
    """
    Pydantic model defining standardized metadata fields for a scientific software,
    model, dataset, or tool. Each field includes a default value and description
    to support automated or manual generation of model cards.

    Attributes:
        capability_name (str): Primary name of the software, model, dataset, or tool.
        brief_description (str): One- to two-sentence summary of purpose and usage.
        systems_covered (str): Real-world systems the model represents.
        contact_name (str): Name of the primary contact person.
        contact_email (str): Email address of the primary contact.
        key_contributors (Union[List[str], str]): Significant contributors or handles.
        doi (str): Digital Object Identifier for citation.
        computational_requirements (Union[List[str], str]): Required hardware resources.
        sponsoring_projects (Union[List[str], str]): Funding sources or associated projects.
        figure (str): Reference or URL to a representative figure.
        figure_caption (str): Caption for the associated figure.
        spatial_resolution (Union[List[str], str]): Supported spatial resolutions.
        geographic_scope (Union[List[str], str]): Geographic areas covered.
        temporal_resolution (Union[List[str], str]): Supported temporal intervals.
        temporal_range (Union[List[str], str]): Time periods the model addresses.
        input_variables (Union[List[str], str]): Required input data types.
        output_variables (Union[List[str], str]): Primary outputs generated.
        interdependencies (Union[List[str], str]): Related models or datasets.
        key_publications (Union[List[str], str]): Key supporting publications.
        category (Union[List[str], str]): Domain classification tags.
        license (str): Licensing terms.
        current_version (str): Current version identifier.
    """
    capability_name: str = Field(default="N/A", description="The primary name of the software, model, dataset, or tool.")
    brief_description: str = Field(default="N/A", description="A brief, one or two-sentence description of the software's main purpose and function.  What are the typical uses?  Who is currently using it?")
    systems_covered: str = Field(default="N/A", description="The real-world systems the model represents (e.g., energy systems, water resources, climate, ecosystems, land use, socioeconomics, machine learning models). List the main ones.")
    contact_name: str = Field(default="N/A", description="The name of the primary contact person for the software, if specified.")
    contact_email: str = Field(default="N/A", description="The email address of the primary contact person.")
    key_contributors: Union[List[str], str] = Field(default="N/A", description="Names or GitHub handles of other significant contributors mentioned.")
    doi: str = Field(default="N/A", description="The current Digital Object Identifier (DOI) for citing the software, if available (check CITATION.cff or README).")
    computational_requirements: Union[List[str], str] = Field(default="N/A", description="Hardware needed to run the software (e.g., Standard laptop, High-performance computing (HPC) cluster, GPU required).")
    sponsoring_projects: Union[List[str], str] = Field(default="N/A", description="Projects or funding agencies that sponsored the development, if mentioned.")
    figure: str = Field(default="N/A", description="Reference or URL to a key figure illustrating the software's structure or results, if described or linked in the text.")
    figure_caption: str = Field(default="N/A", description="The caption associated with the key figure, if available in the text.")
    spatial_resolution: Union[List[str], str] = Field(default="N/A", description="The spatial resolutions (range of tested grid spacing or distance between resolved features) at which the model can operate (e.g., 1 km, 5 arcmin, 0.5 degrees, N/A if not applicable).  Return all that apply.")
    geographic_scope: Union[List[str], str] = Field(default="N/A", description="The geographic area the model covers or is applied to (e.g., Global, CONUS, specific region, point location). Mention both potential and current applications if specified.")
    temporal_resolution: Union[List[str], str] = Field(default="N/A", description="The time step or frequency of the model's calculations.  Either: seconds, minutes, hourly, daily, monthly, annual, 5-year, or decadal).  Return all that apply.")
    temporal_range: Union[List[str], str] = Field(default="N/A", description="The time period the model simulates.  Either: historical, near-future, far-future, specific period.  Return all that apply.")
    input_variables: Union[List[str], str] = Field(default="N/A", description="Key input data or variables required by the software. List major examples.")
    output_variables: Union[List[str], str] = Field(default="N/A", description="Key output data or results generated by the software. List major examples.")
    interdependencies: Union[List[str], str] = Field(default="N/A", description="Other models, software libraries, or specific datasets the software relies on or commonly interacts with (e.g., E3SM, GCAM, specific weather data).")
    key_publications: Union[List[str], str] = Field(default="N/A", description="List of key publications...")
    category: Union[List[str], str] = Field(default="N/A", description="Select from:  Atmosphere, Physical Hydrology, Water Management, Wildfire, Energy, Multisectoral, Land Use Land Cover, Socioeconomics.  Return all that apply.")
    license: str = Field(default="N/A", description="The name of the license associated with the software, model, dataset, or tool.")
    current_version: str = Field(default="N/A", description="The current version of the software, model, dataset, or tool.")
