import webbrowser
from youtube_search import YoutubeSearch

def youtube_search_tool(query: str):
    results = YoutubeSearch(query).to_dict()
    if results:
        url = 'https://www.youtube.com' + results[0]['url_suffix']
        return {"url": url, "title": results[0].get("title", "")}
    else:
        return {"error": "No results found."}

# def youtube_search_tool(query: str):
#     return {"error": "No results found."}

def open_link(url: str):
    try:
        result = webbrowser.open(url)
        if result:
            return {"status": "success"}
        else:
            return {"status": "error", "error": "Failed to open URL"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

tools = [
    {
        "type": "function",
        "function": {
            "name": "youtube_search",
        "description": "Search for url for a video on youtube.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_link",
            "description": "Open a link in browser. Only use this tool if you have a valid url.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"}
                },
                "required": ["url"]
            }
        }
    }
]

function_map = {
    "youtube_search": youtube_search_tool,
    "open_link": open_link
}
