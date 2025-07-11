import json
import time
from typing import List, Optional, Union, Iterator, Dict, Any, cast
from llama_cpp import llama, llama_types
from llama_cpp.llama_chat_format import (
    register_chat_completion_handler,
    _convert_completion_to_chat,
    _convert_text_completion_logprobs_to_chat
)
import llama_cpp.llama_grammar as llama_grammar
from jinja2 import Environment


def _stream_function_calls_to_chat(
    tool_names: List[str],
    completion_chunks: Iterator[llama_types.CreateCompletionStreamResponse],
    completion_id: str
) -> Iterator[llama_types.CreateChatCompletionStreamResponse]:
    """Convert streaming function call completions to streaming chat format"""
    
    # Generate call IDs for consistency
    call_ids = [f"call_{i}_{tool_name}_{completion_id}" for i, tool_name in enumerate(tool_names)]
    
    # First chunk with tool call start
    yield {
        "id": "chatcmpl-" + completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "unknown",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "index": i,
                            "id": call_ids[i],
                            "type": "function",
                            "function": {"name": tool_name, "arguments": ""}
                        }
                        for i, tool_name in enumerate(tool_names)
                    ]
                },
                "finish_reason": None,
            }
        ],
    }
    
    # Stream function arguments
    for chunk in completion_chunks:
        if "choices" in chunk and chunk["choices"]:
            text = chunk["choices"][0].get("text", "")
            if text:
                yield {
                    "id": "chatcmpl-" + completion_id,
                    "object": "chat.completion.chunk", 
                    "created": int(time.time()),
                    "model": "unknown",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,  # Assume single tool call for simplicity
                                        "id": call_ids[0],  # Include the ID for proper accumulation
                                        "function": {"arguments": text}
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
    
    # Final chunk
    yield {
        "id": "chatcmpl-" + completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "unknown", 
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "tool_calls",
            }
        ],
    }


@register_chat_completion_handler("chatml-function-calling-streaming")
def chatml_function_calling_streaming(
    llama: llama.Llama,
    messages: List[llama_types.ChatCompletionRequestMessage],
    functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
    function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
    tools: Optional[List[llama_types.ChatCompletionTool]] = None,
    tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 40,
    min_p: float = 0.05,
    typical_p: float = 1.0,
    stream: bool = False,
    stop: Optional[Union[str, List[str]]] = [],
    response_format: Optional[llama_types.ChatCompletionRequestResponseFormat] = None,
    max_tokens: Optional[int] = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repeat_penalty: float = 1.1,
    tfs_z: float = 1.0,
    mirostat_mode: int = 0,
    mirostat_tau: float = 5.0,
    mirostat_eta: float = 0.1,
    model: Optional[str] = None,
    logits_processor: Optional[llama.LogitsProcessorList] = None,
    grammar: Optional[llama.LlamaGrammar] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    **kwargs,  # type: ignore
) -> Union[
    llama_types.CreateChatCompletionResponse,
    Iterator[llama_types.CreateChatCompletionStreamResponse],
]:
    """
    Custom streaming-enabled chatml function calling handler.
    Based on the original but with proper streaming support.
    """
    
    # Template for function calling format
    function_calling_template = (
        "{% for message in messages %}"
        "<|im_start|>{{ message.role }}\n"
        # System message
        "{% if message.role == 'system' %}"
        "{{ message.content }}"
        "{% if tool_calls %}"
        "\n\nYou have access to the following functions:\n"
        "{% for tool in tools %}"
        "\nfunctions.{{ tool.function.name }}:\n"
        "{{ tool.function.parameters | tojson }}"
        "\n{% endfor %}"
        "\n\nYou can respond to users messages with either a single message or one or more function calls."
        "\n\nTo respond with a message begin the message with 'message:', use the following format:"
        "\n\nmessage:"
        "\n<message>"
        "\n\nTo respond with one or more function calls begin the message with 'functions.<function_name>:', use the following format:"
        "\n\nfunctions.<function_name>:"
        '\n{ "arg1": "value1", "arg2": "value2" }'
        "\nfunctions.<function_name>:"
        '\n{ "arg1": "value1", "arg2": "value2" }'
        "{% endif %}"
        "<|im_end|>\n"
        "{% endif %}"
        # User message
        "{% if message.role == 'user' %}"
        "{{ message.content }}"
        "<|im_end|>\n"
        "{% endif %}"
        # Tool response message
        "{% if message.role == 'tool' %}"
        "Tool {{ message.name }} returned: {{ message.content }}"
        "<|im_end|>\n"
        "{% endif %}"
        # Assistant message  
        "{% if message.role == 'assistant' %}"
        "{% if message.content and message.content | length > 0 %}"
        "{% if tool_calls %}"
        "message:\n"
        "{% endif %}"
        "{{ message.content }}"
        "<|im_end|>\n"
        "{% endif %}"
        "{% if 'tool_calls' in message %}"
        "{% for tool_call in message.tool_calls %}"
        "functions.{{ tool_call.function.name }}:\n"
        "{{ tool_call.function.arguments }}"
        "{% endfor %}"
        "<|im_end|>\n"
        "{% endif %}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    
    template_renderer = Environment().from_string(function_calling_template)

    # Convert legacy functions to tools
    if functions is not None:
        tools = [
            {
                "type": "function",
                "function": function,
            }
            for function in functions
        ]

    # Convert legacy function_call to tool_choice
    if function_call is not None:
        if isinstance(function_call, str) and (
            function_call == "none" or function_call == "auto"
        ):
            tool_choice = function_call
        if isinstance(function_call, dict) and "name" in function_call:
            tool_choice = {
                "type": "function",
                "function": {
                    "name": function_call["name"],
                },
            }

    stop = (
        [stop, "<|im_end|>"]
        if isinstance(stop, str)
        else stop + ["<|im_end|>"] if stop else ["<|im_end|>"]
    )

    # Case 1: No tool choice by user
    if (
        tool_choice is None
        or (isinstance(tool_choice, str) and tool_choice == "none")
        or tools is None
        or len(tools) == 0
    ):
        prompt = template_renderer.render(
            messages=messages,
            tools=[],
            tool_calls=None,
            add_generation_prompt=True,
        )

        return _convert_completion_to_chat(
            llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=stream,
                stop=stop,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                grammar=grammar,
                logprobs=top_logprobs if logprobs else None,
            ),
            stream=stream,
        )

    # Case 2 & 3: Tool choice (specific or auto)
    # First determine if it's a message or function call
    prompt = template_renderer.render(
        messages=messages,
        tools=tools,
        tool_calls=True,
        add_generation_prompt=True,
    )

    # Use two-stage approach for streaming:
    # 1. First determine if it's message or function
    # 2. Then stream the actual content
    
    function_names = [tool['function']['name'] for tool in tools] if tools else []
    initial_gbnf_tool_grammar = (
        """root   ::= functions | "message:"\n"""
        f"""functions ::= {' | '.join([f'"functions.{name}:"' for name in function_names])}\n"""
    )
    
    # Step 1: Determine response type (non-streaming)
    choice_completion = llama.create_completion(
        prompt=prompt,
        temperature=0,
        stream=False,
        stop=[":"],
        max_tokens=50,  # Just enough to determine choice
        grammar=llama_grammar.LlamaGrammar.from_string(
            initial_gbnf_tool_grammar, verbose=llama.verbose
        ),
    )
    
    choice_text = choice_completion["choices"][0]["text"]
    
    if "message" in choice_text:
        # Step 2a: Stream regular message
        return _convert_completion_to_chat(
            llama.create_completion(
                prompt=prompt + "message:\n",
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=stream,
                stop=["<|im_end|>"],
                logprobs=top_logprobs if logprobs else None,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
            ),
            stream=stream,
        )
    
    # Step 2b: Stream function calls
    tool_name = choice_text[len("functions."):] if choice_text.startswith("functions.") else function_names[0]
    tool = next((tool for tool in tools if tool["function"]["name"] == tool_name), None)
    
    if tool is None:
        # Fallback to first available tool
        tool = tools[0] if tools else None
        tool_name = tool["function"]["name"] if tool else "unknown"
    
    # Create grammar for function arguments
    try:
        grammar = llama_grammar.LlamaGrammar.from_json_schema(
            json.dumps(tool["function"]["parameters"]), verbose=llama.verbose
        )
    except Exception as e:
        grammar = llama_grammar.LlamaGrammar.from_string(
            llama_grammar.JSON_GBNF, verbose=llama.verbose
        )
        if llama.verbose:
            print(f"Failed to parse function body as JSON schema, falling back to default grammar: {e}")
    
    # Stream function arguments
    completion_chunks = llama.create_completion(
        prompt=prompt + f"functions.{tool_name}:\n",
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        typical_p=typical_p,
        stream=stream,
        stop=stop,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repeat_penalty=repeat_penalty,
        tfs_z=tfs_z,
        mirostat_mode=mirostat_mode,
        mirostat_tau=mirostat_tau,
        mirostat_eta=mirostat_eta,
        model=model,
        logits_processor=logits_processor,
        grammar=grammar,
    )
    
    if stream:
        return _stream_function_calls_to_chat(
            [tool_name], 
            completion_chunks,
            choice_completion["id"]
        )
    else:
        # Non-streaming function call response
        completion_chunks = cast(llama_types.CreateCompletionResponse, completion_chunks)
        return {
            "id": "chat" + choice_completion["id"],
            "object": "chat.completion",
            "created": choice_completion["created"],
            "model": choice_completion["model"],
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "index": 0,
                    "logprobs": _convert_text_completion_logprobs_to_chat(
                        completion_chunks["choices"][0]["logprobs"]
                    ),
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_0_{tool_name}_{choice_completion['id']}",
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": completion_chunks["choices"][0]["text"],
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "completion_tokens": (choice_completion["usage"]["completion_tokens"] + 
                                    completion_chunks["usage"]["completion_tokens"]),
                "prompt_tokens": choice_completion["usage"]["prompt_tokens"],
                "total_tokens": (choice_completion["usage"]["total_tokens"] + 
                               completion_chunks["usage"]["total_tokens"]),
            },
        }