"""
Business Opportunity Score MCP Server

This MCP server provides business opportunity scores based on US Census Bureau
County Business Patterns data. Scores are generated from a machine learning model
trained on establishment counts, payroll data, and employment statistics.

Deploy this server on Posit Connect to provide AI assistants with real-time
business intelligence scoring capabilities.
"""

import csv
import os
from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import HTMLResponse

# Initialize the MCP server
mcp = FastMCP(
    name="business-opportunity-score",
    instructions="""
    This MCP server provides business opportunity scores based on US Census Bureau data.

    Use the get_opportunity_score tool to retrieve scores for specific combinations of:
    - US State (e.g., "California", "Texas", "New York")
    - Corporation type: c-corp, s-corp, sole-proprietor, partnership, nonprofit, government, other
    - Employee size: 1-4, 5-9, 10-19, 20-49, 50-99, 100-249, 250-499, 500-999, 1000+

    Scores range from 0-100, where higher scores indicate better business opportunities
    based on factors like average salaries, establishment density, and economic momentum.

    Use list_states, list_corp_types, and list_emp_sizes to discover valid parameter values.
    """,
)

# Load the lookup table at startup
LOOKUP_DATA: dict[tuple[str, str, str], dict] = {}
STATES: set[str] = set()
CORP_TYPES: set[str] = set()
EMP_SIZES: set[str] = set()


def load_lookup_table():
    """Load the score lookup table from CSV."""
    global LOOKUP_DATA, STATES, CORP_TYPES, EMP_SIZES

    csv_path = Path(__file__).parent / "score_lookup.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Score lookup table not found at {csv_path}. "
            "Please run model.R first to generate the lookup table."
        )

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            state = row["state"]
            corp_type = row["corp_type"]
            emp_size = row["emp_size"]

            STATES.add(state)
            CORP_TYPES.add(corp_type)
            EMP_SIZES.add(emp_size)

            key = (state.lower(), corp_type.lower(), emp_size.lower())
            LOOKUP_DATA[key] = {
                "state": state,
                "corp_type": corp_type,
                "emp_size": emp_size,
                "score": float(row["score"]),
                "confidence": row["confidence"],
                "establishments": int(row["establishments"]),
                "employees": int(row["employees"]),
                "avg_salary_thousands": float(row["avg_salary_thousands"]),
            }


# Load data on module import
try:
    load_lookup_table()
except FileNotFoundError:
    # Will be loaded later or fail gracefully
    pass


@mcp.tool(
    description="""
    Get the business opportunity score for a specific state, corporation type, and employee size.

    The score (0-100) represents the relative business opportunity based on:
    - Average salary levels (40% weight)
    - Economic momentum/payroll growth (35% weight)
    - Establishment density (25% weight)

    Higher scores indicate more favorable business conditions.

    Returns the score along with supporting data including confidence level,
    number of establishments, total employees, and average salary.
    """
)
def get_opportunity_score(
    state: Annotated[str, "US State name (e.g., 'California', 'Texas', 'New York')"],
    corp_type: Annotated[str, "Corporation type: c-corp, s-corp, sole-proprietor, partnership, nonprofit, government, other"],
    emp_size: Annotated[str, "Employee size category: 1-4, 5-9, 10-19, 20-49, 50-99, 100-249, 250-499, 500-999, 1000+"],
) -> dict:
    """Get business opportunity score for the specified parameters."""

    if not LOOKUP_DATA:
        return {
            "error": "Lookup table not loaded. Please ensure score_lookup.csv exists.",
            "hint": "Run the R model script first: Rscript model.R"
        }

    # Normalize inputs
    key = (state.lower().strip(), corp_type.lower().strip(), emp_size.lower().strip())

    if key not in LOOKUP_DATA:
        # Try to provide helpful error message
        state_match = state.lower().strip() in [s.lower() for s in STATES]
        corp_match = corp_type.lower().strip() in [c.lower() for c in CORP_TYPES]
        emp_match = emp_size.lower().strip() in [e.lower() for e in EMP_SIZES]

        suggestions = []
        if not state_match:
            suggestions.append(f"Invalid state '{state}'. Use list_states to see valid options.")
        if not corp_match:
            suggestions.append(f"Invalid corp_type '{corp_type}'. Use list_corp_types to see valid options.")
        if not emp_match:
            suggestions.append(f"Invalid emp_size '{emp_size}'. Use list_emp_sizes to see valid options.")

        return {
            "error": "No data found for the specified combination",
            "suggestions": suggestions if suggestions else ["This combination may not exist in the census data"],
            "provided": {"state": state, "corp_type": corp_type, "emp_size": emp_size}
        }

    data = LOOKUP_DATA[key]

    return {
        "score": data["score"],
        "interpretation": _interpret_score(data["score"]),
        "confidence": data["confidence"],
        "details": {
            "state": data["state"],
            "corporation_type": data["corp_type"],
            "employee_size": data["emp_size"],
            "establishments": data["establishments"],
            "total_employees": data["employees"],
            "avg_salary_thousands": round(data["avg_salary_thousands"], 2),
        },
        "methodology": {
            "source": "US Census Bureau County Business Patterns (2022)",
            "model": "Random Forest regression on salary, momentum, and density features",
            "score_range": "0-100 (higher = better opportunity)"
        }
    }


def _interpret_score(score: float) -> str:
    """Provide a human-readable interpretation of the score."""
    if score >= 80:
        return "Excellent - Strong business opportunity indicators"
    elif score >= 60:
        return "Good - Above average business conditions"
    elif score >= 40:
        return "Moderate - Average business environment"
    elif score >= 20:
        return "Fair - Below average conditions"
    else:
        return "Limited - Challenging business environment"


@mcp.tool(
    description="List all valid US states available in the dataset. Use these exact values when calling get_opportunity_score."
)
def list_states() -> dict:
    """Get list of all available states."""
    if not STATES:
        return {"error": "Data not loaded"}

    sorted_states = sorted(STATES)
    return {
        "count": len(sorted_states),
        "states": sorted_states,
        "note": "Use these exact state names when calling get_opportunity_score"
    }


@mcp.tool(
    description="List all valid corporation types. Use these codes when calling get_opportunity_score."
)
def list_corp_types() -> dict:
    """Get list of all available corporation types."""
    if not CORP_TYPES:
        return {"error": "Data not loaded"}

    type_descriptions = {
        "c-corp": "C-Corporation (traditional corporation with double taxation)",
        "s-corp": "S-Corporation (pass-through taxation)",
        "sole-proprietor": "Individual/Sole Proprietorship",
        "partnership": "Partnership (general or limited)",
        "nonprofit": "Non-profit organization (501c3, etc.)",
        "government": "Government entity",
        "other": "Other non-corporate legal forms"
    }

    return {
        "count": len(CORP_TYPES),
        "corp_types": [
            {"code": ct, "description": type_descriptions.get(ct, ct)}
            for ct in sorted(CORP_TYPES)
        ]
    }


@mcp.tool(
    description="List all valid employee size categories. Use these codes when calling get_opportunity_score."
)
def list_emp_sizes() -> dict:
    """Get list of all available employee size categories."""
    if not EMP_SIZES:
        return {"error": "Data not loaded"}

    size_order = ["1-4", "5-9", "10-19", "20-49", "50-99", "100-249", "250-499", "500-999", "1000+"]
    sorted_sizes = sorted(EMP_SIZES, key=lambda x: size_order.index(x) if x in size_order else 99)

    return {
        "count": len(sorted_sizes),
        "emp_sizes": sorted_sizes,
        "note": "Ranges represent number of employees at establishment"
    }


@mcp.tool(
    description="""
    Compare opportunity scores across multiple states for a given corporation type and employee size.
    Useful for identifying the best locations for a specific business profile.
    """
)
def compare_states(
    states: Annotated[list[str], "List of US states to compare (e.g., ['California', 'Texas', 'New York'])"],
    corp_type: Annotated[str, "Corporation type: c-corp, s-corp, sole-proprietor, partnership, nonprofit, government, other"],
    emp_size: Annotated[str, "Employee size category: 1-4, 5-9, 10-19, 20-49, 50-99, 100-249, 250-499, 500-999, 1000+"],
) -> dict:
    """Compare opportunity scores across multiple states."""

    results = []
    errors = []

    for state in states:
        key = (state.lower().strip(), corp_type.lower().strip(), emp_size.lower().strip())
        if key in LOOKUP_DATA:
            data = LOOKUP_DATA[key]
            results.append({
                "state": data["state"],
                "score": data["score"],
                "confidence": data["confidence"],
                "establishments": data["establishments"]
            })
        else:
            errors.append(state)

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "comparison": {
            "corp_type": corp_type,
            "emp_size": emp_size,
            "results": results,
            "best_state": results[0]["state"] if results else None,
            "worst_state": results[-1]["state"] if results else None,
        },
        "errors": errors if errors else None,
        "summary": f"Compared {len(results)} states for {corp_type} businesses with {emp_size} employees"
    }


@mcp.tool(
    description="""
    Get the top N states by opportunity score for a specific corporation type and employee size.
    Useful for finding the best locations to establish or expand a business.
    """
)
def top_states(
    corp_type: Annotated[str, "Corporation type: c-corp, s-corp, sole-proprietor, partnership, nonprofit, government, other"],
    emp_size: Annotated[str, "Employee size category: 1-4, 5-9, 10-19, 20-49, 50-99, 100-249, 250-499, 500-999, 1000+"],
    n: Annotated[int, "Number of top states to return (default: 10)"] = 10,
) -> dict:
    """Get top N states by opportunity score."""

    matching = []
    corp_key = corp_type.lower().strip()
    emp_key = emp_size.lower().strip()

    for key, data in LOOKUP_DATA.items():
        if key[1] == corp_key and key[2] == emp_key:
            matching.append({
                "rank": 0,  # Will be set after sorting
                "state": data["state"],
                "score": data["score"],
                "confidence": data["confidence"],
                "establishments": data["establishments"],
                "avg_salary_thousands": round(data["avg_salary_thousands"], 2)
            })

    # Sort by score descending
    matching.sort(key=lambda x: x["score"], reverse=True)

    # Add ranks
    for i, item in enumerate(matching[:n], 1):
        item["rank"] = i

    return {
        "query": {
            "corp_type": corp_type,
            "emp_size": emp_size,
            "requested": n
        },
        "top_states": matching[:n],
        "total_available": len(matching)
    }


@mcp.custom_route("/", methods=["GET"])
async def landing_page(request: Request) -> HTMLResponse:
    """Serve a landing page with server info and setup instructions."""
    # List our known tools directly
    tool_info = [
        ("get_opportunity_score", "Get business opportunity score for a state/corp_type/emp_size"),
        ("list_states", "List all valid US states"),
        ("list_corp_types", "List all valid corporation types"),
        ("list_emp_sizes", "List all valid employee size categories"),
        ("compare_states", "Compare scores across multiple states"),
        ("top_states", "Get top N states by opportunity score"),
    ]

    tools_html = "<ul>"
    for name, desc in tool_info:
        tools_html += f"<li><code>{name}</code> - {desc}</li>"
    tools_html += "</ul>"

    # Build base URL accounting for proxy
    forwarded_proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    forwarded_host = request.headers.get("x-forwarded-host", request.url.netloc)
    current_path = str(request.url.path).rstrip("/")
    base_url = f"{forwarded_proto}://{forwarded_host}{current_path}"
    mcp_url = f"{base_url}/mcp"

    # Detect if running on Posit Connect
    is_posit_connect = bool(os.getenv("CONNECT_SERVER"))

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Business Opportunity Score MCP Server</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 900px;
                margin: 0 auto;
                padding: 2rem;
                line-height: 1.6;
                color: #333;
            }}
            h1 {{ color: #1a1a1a; border-bottom: 2px solid #0066cc; padding-bottom: 0.5rem; }}
            h2 {{ color: #444; margin-top: 2rem; }}
            h3 {{ color: #666; margin-top: 1.5rem; }}
            code {{
                background: #f4f4f4;
                padding: 0.2rem 0.4rem;
                border-radius: 3px;
                font-size: 0.9em;
            }}
            pre {{
                background: #f4f4f4;
                padding: 1rem;
                border-radius: 5px;
                overflow-x: auto;
            }}
            .endpoint {{
                background: #e7f3ff;
                padding: 1rem;
                border-radius: 5px;
                margin: 1rem 0;
                border-left: 4px solid #0066cc;
            }}
            .warning {{
                background: #fff3e0;
                padding: 1rem;
                border-radius: 5px;
                margin: 1rem 0;
                border-left: 4px solid #ff9800;
            }}
        </style>
    </head>
    <body>
        <h1>Business Opportunity Score MCP Server</h1>
        <p>This is a <a href="https://modelcontextprotocol.io">Model Context Protocol (MCP)</a> server
        that provides business opportunity scores based on US Census Bureau data.</p>

        <h2>MCP Endpoint</h2>
        <div class="endpoint">
            <strong>URL:</strong> <code>{mcp_url}</code>
        </div>

        {"<div class='warning'><strong>Authentication Required:</strong> This server is running on Posit Connect. You must include your Connect API key in requests.</div>" if is_posit_connect else ""}

        <h2>Setup Instructions</h2>

        {f'''<h3>Getting a Posit Connect API Key</h3>
        <ol>
            <li>Log in to Posit Connect</li>
            <li>Click your username in the top right corner</li>
            <li>Select "API Keys"</li>
            <li>Click "New API Key" and give it a name</li>
            <li>Copy the key (you won't be able to see it again)</li>
        </ol>''' if is_posit_connect else ""}

        <h3>Claude Code</h3>
        <pre>claude mcp add --transport http business-score {mcp_url}{f' \\{chr(10)}  --header "Authorization: Key YOUR_API_KEY"' if is_posit_connect else ""}</pre>

        <h3>Claude Desktop</h3>
        <p>Add to your Claude Desktop config file:</p>
        <pre>{{
  "mcpServers": {{
    "business-score": {{
      "url": "{mcp_url}"{f',{chr(10)}      "headers": {{{chr(10)}        "Authorization": "Key YOUR_API_KEY"{chr(10)}      }}' if is_posit_connect else ""}
    }}
  }}
}}</pre>

        <h2>Available Tools ({len(tool_info)} total)</h2>
        {tools_html}

        <h2>Data Source</h2>
        <p>Scores are generated from US Census Bureau County Business Patterns (2022) data,
        using a Random Forest model trained on salary, economic momentum, and establishment density features.</p>

        <p>Data loaded: <strong>{"Yes" if LOOKUP_DATA else "No"}</strong>
        {f" ({len(STATES)} states)" if STATES else ""}</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


# ASGI app for deployment
mcp_app = mcp.http_app(path="/mcp")
app = mcp_app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

