import uvicorn
from configs.api import HOST, PORT

if __name__ == "__main__":
    # Launch FastAPI application using Uvicorn
    uvicorn.run("src.api:app", host=HOST, port=PORT, reload=True)
