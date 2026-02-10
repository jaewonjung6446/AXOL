"""Run the Axol server: python -m axol.server"""

import uvicorn

from axol.server.app import app

uvicorn.run(app, host="0.0.0.0", port=8080)
