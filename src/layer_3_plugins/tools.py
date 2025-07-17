import inspect
import json
from typing import get_type_hints
import docstring_parser
import webbrowser
from youtube_search import YoutubeSearch

TOOLS = []
FUNCTION_MAP = {}

def create_tool_response(success: bool, message: str, data: dict = None, error: str = None) -> dict:
    """
    Create standardized tool response format.
    
    Args:
        success (bool): Whether the tool execution was successful
        message (str): Human-readable message for the user
        data (dict, optional): Additional data returned by the tool
        error (str, optional): Error message if failed
        
    Returns:
        dict: Standardized tool response format
    """
    response = {
        "success": success,
        "message": message
    }
    
    if data is not None:
        response["data"] = data
        
    if error is not None:
        response["error"] = error
        
    return response

def tool(func):
    """
    Decorator to register a function as a tool, auto parse docstring Google style.

    Google style docstring example:

    \"\"\"
    Function summary.

    Args:
        query (str): YouTube search query.
    Returns:
        dict: Standard tool response format with success, message, and optional data/error fields.
    \"\"\"
    """
    # parse docstring
    doc = docstring_parser.parse(func.__doc__ or "")
    # build schema from signature
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    properties = {}
    required = []
    for param in doc.params:
        # get type annotation
        ann = hints.get(param.arg_name, str)
        # simple Python type -> JSON type mapping
        t = "string"
        if ann in (int, float, bool):
            t = "number" if ann in (int, float) else "boolean"
        properties[param.arg_name] = {
            "type": t,
            "description": param.description
        }
        required.append(param.arg_name)

    # return description
    return_desc = doc.returns.description if doc.returns else ""

    schema = {
        "type": "object",
        "properties": properties,
        "required": required
    }

    # register tool
    TOOLS.append({
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc.short_description or "",
            "parameters": schema,
            "returns": {"description": return_desc}
        }
    })
    FUNCTION_MAP[func.__name__] = func
    return func

@tool
def youtube_search(query: str):
    """
    Search for videos on YouTube.

    Args:
        query (str): Search keyword.
    Returns:
        dict: Standard tool response format with success, message, and optional data/error fields.
    """
    try:
        results = YoutubeSearch(query).to_dict()
        if results:
            video_data = {
                "url": "https://www.youtube.com" + results[0]['url_suffix'],
                "title": results[0].get("title", ""),
                "duration": results[0].get("duration", ""),
                "views": results[0].get("views", "")
            }
            return create_tool_response(
                success=True,
                message=f"Found video: {video_data['title']}",
                data=video_data
            )
        else:
            return create_tool_response(
                success=False,
                message="No videos found for your search",
                error="No results found"
            )
    except Exception as e:
        return create_tool_response(
            success=False,
            message="Failed to search YouTube",
            error=str(e)
        )

@tool
def open_link(url: str):
    """
    Open link in browser.

    Args:
        url (str): URL to open.
    Returns:
        dict: Standard tool response format with success, message, and optional data/error fields.
    """
    try:
        ok = webbrowser.open(url)
        if ok:
            return create_tool_response(
                success=True,
                message="Successfully opened link in browser",
            )
        else:
            return create_tool_response(
                success=False,
                message="Failed to open link in browser",
                error="Browser failed to open URL"
            )
    except Exception as e:
        return create_tool_response(
            success=False,
            message="Error opening link",
            error=str(e)
        )

def format_tools_for_prompt(tools_list=None):
    """
    Format tools into a prompt string for models without native tool support.
    
    Args:
        tools_list: List of tools to format. If None, uses global TOOLS.
        
    Returns:
        str: Formatted tool context for prompt injection.
    """
    if tools_list is None:
        tools_list = TOOLS
        
    if not tools_list:
        return ""
    
    prompt = "\n## Available Tools\n"
    prompt += "You have access to the following tools. To use a tool, respond with a JSON object containing 'tool_calls' array:\n\n"
    
    for idx, tool in enumerate(tools_list):
        func_info = tool.get("function", {})
        name = func_info.get("name", "")
        description = func_info.get("description", "")
        parameters = func_info.get("parameters", {})
        
        prompt += f"{idx+1}.\nName: {name}\n"
        prompt += f"Description: {description}\n"
        
        # Format parameters
        if parameters.get("properties"):
            prompt += "Parameters:\n"
            for param_name, param_info in parameters["properties"].items():
                param_type = param_info.get("type", "string")
                param_desc = param_info.get("description", "")
                required = param_name in parameters.get("required", [])
                req_text = " (required)" if required else " (optional)"
                prompt += f"  - {param_name} ({param_type}){req_text}: {param_desc}\n"
        
        # Example usage
        # example_args = {}
        # if parameters.get("properties"):
        #     for param_name, param_info in parameters["properties"].items():
        #         if param_info.get("type") == "string":
        #             example_args[param_name] = f"example_{param_name}"
        #         elif param_info.get("type") == "number":
        #             example_args[param_name] = 123
        #         elif param_info.get("type") == "boolean":
        #             example_args[param_name] = True
                    
        # prompt += f"Example usage:\n"
        # prompt += f'```json\n{{\n  "tool_calls": [{{\n    "name": "{name}",\n    "arguments": {json.dumps(example_args, indent=6)}\n  }}]\n}}\n```\n\n'
    
    return prompt

# Export for compatibility
tools = TOOLS
function_map = FUNCTION_MAP
# print(tools)
# print(format_tools_for_prompt(tools))