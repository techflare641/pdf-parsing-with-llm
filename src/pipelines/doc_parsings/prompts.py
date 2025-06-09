doc_extraction_prompt = """
You will be given an image to analyze. Your task is to extract and structure all meaningful information, including text, tables, charts, graphics, and diagrams. Follow the steps below and present the results strictly according to the defined output schema.

1. Extract Full Text
    a. Extract all visible text thoroughly.
    b. Capture the entire text as a single string.
    c. Additionally, break it down into individual text blocks (like headings, bullet points, captions, labels, etc.).
    d. Summarize the entire text in a 2-3 sentence summary.

2. Extract and Analyze Content Types
    a. Charts
        - Identify the type of chart (e.g., bar, pie, line, etc.).
        - Extract all data points shown.
        - Summarize key insights, trends, or takeaways from the chart.

    b. Tables
        -  Extract headers and rows of any tables.
        - Present table data in a structured format.
        - Summarize key insights, trends, or takeaways from the table.

    c. Text Blocks
        - Present each individual text block clearly.
        - Where applicable, explain any relationships or flow between different pieces of text (e.g., a heading and its bullet points).
        - Summarize key insights, trends, or takeaways from the text blocks.
        - if the text block is a table, chart, graphic, flowchart, diagram or footer, do not include the text block in the text blocks section.

    d. Graphics
        - Describe non-logo graphics and illustrations in plain text.
        - Focus on meaningful visuals (e.g., process illustrations, concept diagrams).
        - Ignore and exclude logos, watermarks, backgrounds, or branding elements.
        - Summarize key insights, trends, or takeaways from the graphics.

    e. Flowcharts & Diagrams
        - Extract the title or purpose of the diagram.
        - List all nodes with their labels.
        - Describe how nodes are connected (e.g., flow from A to B).
        - Summarize the overall logic or process illustrated.

    f. Footers
        - Extract any notes or sources mentioned at the bottom.
        - Present them as clearly labeled items.

3. Output Format
    a. Present your results in a structured format matching the predefined output schema.
    b. Ensure proper field names, data types, and hierarchy.

4. Important Rules
    a. Do not reference or mention "image" or "extracted from image".
    b. Focus only on presenting the content and insights directly.
    c. Skip decorative or irrelevant elements like brand marks, logos, or visual design patterns.

Follow the instructions strictly and geneart the output in the format of the output schema.
"""
