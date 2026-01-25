"""
Business Opportunity Score MCP Server

This MCP server provides business opportunity scores based on US Census Bureau
County Business Patterns data. Scores are generated from a machine learning model
trained on establishment counts, payroll data, and employment statistics.

Deploy this server on Posit Connect to provide AI assistants with real-time
business intelligence scoring capabilities.
"""

import csv
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI
from fastmcp import FastMCP

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


# Create MCP HTTP app and mount in FastAPI
mcp_app = mcp.http_app(path="/mcp")
app = FastAPI(
    title="Business Opportunity Score MCP Server",
    lifespan=mcp_app.lifespan,
)
app.mount("/", mcp_app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

