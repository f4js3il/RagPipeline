from fastapi import FastAPI
from app.routes import query

app = FastAPI(title="RagPipeline API", description="API for RagPipeline")
app.include_router(query.router)
