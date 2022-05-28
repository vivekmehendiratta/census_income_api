from fastapi import APIRouter
from starlette.responses import RedirectResponse
app_home = APIRouter()


@app_home.get('/home', tags=["Welcome to Census Income Prediction Page"])
async def hello():
    return RedirectResponse("/redoc")