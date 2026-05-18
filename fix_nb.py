import json

with open("notebooks/02_etl_to_streamlit_pipeline.ipynb", "r") as f:
    nb = json.load(f)

new_cells = []
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "---" in source and "Verificar Streamlit" in source:
            # We need to split this cell
            code_part = source.split("---\n")[0]
            markdown_part = "---\n" + source.split("---\n")[1]
            
            # Update code cell
            cell["source"] = [line + "\n" if i < len(code_part.split("\n"))-1 else line for i, line in enumerate(code_part.split("\n"))]
            # remove empty lines at the end
            while cell["source"] and cell["source"][-1].strip() == "":
                cell["source"].pop()
                
            new_cells.append(cell)
            
            # Create new markdown cell
            new_cells.append({
                "cell_type": "markdown",
                "id": "cdd2cea2",
                "metadata": {},
                "source": [line + "\n" if i < len(markdown_part.split("\n"))-1 else line for i, line in enumerate(markdown_part.split("\n"))]
            })
            continue
    new_cells.append(cell)

nb["cells"] = new_cells

with open("notebooks/02_etl_to_streamlit_pipeline.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

