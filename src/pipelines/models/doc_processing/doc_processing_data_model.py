from typing import List
from pydantic import BaseModel, ConfigDict, Field


class TextBlock(BaseModel):
    text: str = Field(..., description="Content of the text block.")

class Table(BaseModel):
    headers: List[str] = Field(..., description="Column headers of the table.")
    rows: List[List[str]] = Field(
        ..., description="Rows of the table with extracted data."
    )
    insights: str = Field(..., description="Key takeaways and insights from the table.")
    markdown: str = Field(..., description="The table data formatted in markdown.")

class ChartDataPoint(BaseModel):
    label: str = Field(..., description="Category or label of the data point.")
    legend: str = Field(
        ...,
        description="Legend of the data point. What does that data point represent?",
    )
    value: str = Field(..., description="Numeric value of the data point.")

class Chart(BaseModel):
    chart_type: str = Field(
        ..., description="The type of chart (e.g., bar, line, pie, etc.)."
    )
    data: List[ChartDataPoint] = Field(
        ..., description="Data points extracted from the chart."
    )
    insights: str = Field(..., description="Key takeaways and insights from the chart.")

class Graphic(BaseModel):
    description: str = Field(
        ..., description="Description of the graphic, insights and its significance."
    )

class DiagramNode(BaseModel):
    id: str
    label: str

class DiagramConnection(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    label: str = Field(
        ..., description="Optional label or description for the connection."
    )

class FlowchartDiagram(BaseModel):
    title: str = Field(
        ..., description="Title or description of the flowchart/diagram."
    )
    nodes: List[DiagramNode] = Field(
        ..., description="Key components or nodes in the diagram."
    )
    connections: List[DiagramConnection] = Field(
        ..., description="Relationships or flows between the nodes."
    )
    insights: str = Field(
        ..., description="Key takeaways and insights from the flowchart/diagram."
    )

class Footer(BaseModel):
    source: str = Field(..., description="Source of the information.")
    notes: str = Field(
        ...,
        description="Additional notes or clarifications regarding the extracted data.",
    )

class DocOCRResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    extracted_text: str = Field(..., description="All text extracted from the image.")
    summary: str = Field(..., description="Summary of the image.")
    text_blocks: List[TextBlock] = Field(
        ..., description="Various blocks of text found in the image."
    )
    text_blocks_insights: str = Field(
        ..., description="Key takeaways and insights from all text blocks collectively."
    )
    tables: List[Table] = Field(
        ..., description="List of tables extracted from the image."
    )
    charts: List[Chart] = Field(
        ...,
        description="List of charts identified in the image along with extracted data.",
    )
    graphics: List[Graphic] = Field(
        ..., description="List of graphics detected with descriptions."
    )
    flowcharts_and_diagrams: List[FlowchartDiagram] = Field(
        ..., description="List of identified flowcharts or diagrams."
    )
    footers: List[Footer] = Field(
        ...,
        description="List of footers containing sources, notes, and additional information.",
    )
