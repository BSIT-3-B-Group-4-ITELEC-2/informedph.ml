from fastapi import FastAPI
from api import news
from containers.container import Container

app = FastAPI(title="InformedPH API")

container = Container()

container.wire(modules=[news])

app.include_router(news.router)
