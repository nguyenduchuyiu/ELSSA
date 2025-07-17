from src.layer_2_agentic_reasoning.llm_server import app

import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)