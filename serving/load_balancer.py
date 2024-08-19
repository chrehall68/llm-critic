import uvicorn
import argparse
from argparse import Namespace
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import aiohttp

app = FastAPI()
app.state.remotes = []
app.state.i = 0


@app.get("/health")
async def health() -> Response:
    """Health check."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{app.state.remotes[0]}/health") as resp:
            return Response(status_code=resp.status)


@app.get("/v1/models")
async def show_available_models():
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{app.state.remotes[0]}/v1/models") as resp:
            return JSONResponse(content=await resp.json())


@app.get("/version")
async def show_version():
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{app.state.remotes[0]}/version") as resp:
            return JSONResponse(content=await resp.json())


@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    json = await request.json()
    async with aiohttp.ClientSession() as session:
        app.state.i += 1
        print(
            "sending",
            json,
            "to",
            app.state.remotes[app.state.i % len(app.state.remotes)],
        )
        r = session.post(
            f"{app.state.remotes[app.state.i % len(app.state.remotes)]}/v1/chat/completions",
            json=json,
        )
        async with r as resp:
            print(await resp.text())
            return JSONResponse(content=await resp.json())


@app.post("/v1/completions")
async def create_completion(request: Request):
    json = await request.json()
    async with aiohttp.ClientSession() as session:
        app.state.i += 1
        r = session.post(
            f"{app.state.remotes[app.state.i % len(app.state.remotes)]}/v1/completions",
            json=json,
        )
        async with r as resp:
            return JSONResponse(content=await resp.json())


def init_app(args: Namespace):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )
    app.state.remotes = args.remotes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--allowed-origins", type=str, default="*")
    parser.add_argument("--allow-credentials", action="store_true")
    parser.add_argument("--allowed-methods", type=str, default="*")
    parser.add_argument("--allowed-headers", type=str, default="*")
    parser.add_argument("--remotes", type=str, nargs="+")
    args = parser.parse_args()

    init_app(args)
    uvicorn.run(app, host=args.host, port=args.port)
