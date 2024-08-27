from typing import List
import uvicorn
import argparse
from argparse import Namespace
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import aiohttp
from dataclasses import dataclass


class RemoteHeap:
    """
    Custom binary heap. The functionality is the same as that of
    heapq, but also has a decrease_key method.
    """

    @dataclass
    class Remote:
        url: str
        load: int
        _idx: int

        def __hash__(self) -> int:
            return hash(self.url)

    def __init__(self):
        self.remotes: List[RemoteHeap.Remote] = []

    def get_min(self) -> "RemoteHeap.Remote":
        return self.remotes[0]

    def increment_min(self) -> None:
        """
        Increase the load of the min remote by 1. Update heap accordingly.
        """
        self.remotes[0].load += 1
        self.heapify_down(0)

    def decrease_key(self, remote: "RemoteHeap.Remote") -> None:
        """
        Decrease the load of `remote` by 1. Update heap accordingly.
        """
        self.remotes[remote._idx].load -= 1
        self.heapify_up(remote._idx)

    def heapify_down(self, idx: int) -> None:
        left_child = 2 * idx + 1
        right_child = 2 * idx + 2
        smallest = idx

        if (
            left_child < len(self.remotes)
            and self.remotes[left_child].load < self.remotes[smallest].load
        ):
            smallest = left_child
        if (
            right_child < len(self.remotes)
            and self.remotes[right_child].load < self.remotes[smallest].load
        ):
            smallest = right_child
        if smallest != idx:
            # swap idxs and values
            self.remotes[smallest], self.remotes[idx] = (
                self.remotes[idx],
                self.remotes[smallest],
            )
            self.remotes[smallest]._idx, self.remotes[idx]._idx = (
                self.remotes[idx]._idx,
                self.remotes[smallest]._idx,
            )
            # continue heapifying down
            self.heapify_down(smallest)

    def heapify_up(self, idx: int) -> None:
        parent = (idx - 1) // 2

        if parent >= 0 and self.remotes[parent].load > self.remotes[idx].load:
            # swap idxs and values
            self.remotes[parent], self.remotes[idx] = (
                self.remotes[idx],
                self.remotes[parent],
            )
            self.remotes[parent]._idx, self.remotes[idx]._idx = (
                self.remotes[idx]._idx,
                self.remotes[parent]._idx,
            )
            # continue heapifying up
            self.heapify_up(parent)

    def __str__(self) -> str:
        return str(self.remotes)


app = FastAPI()
app.state.remotes = RemoteHeap()  # min heap of (load, url)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{app.state.remotes.get_min().url}/health") as resp:
            return Response(status_code=resp.status)


@app.get("/v1/models")
async def show_available_models():
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{app.state.remotes.get_min().url}/v1/models") as resp:
            return JSONResponse(content=await resp.json())


@app.get("/version")
async def show_version():
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{app.state.remotes.get_min().url}/version") as resp:
            return JSONResponse(content=await resp.json())


@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    json = await request.json()
    async with aiohttp.ClientSession() as session:
        # get remote and increase load
        remote = app.state.remotes.get_min()
        app.state.remotes.increment_min()

        # send request
        r = session.post(
            f"{remote.url}/v1/chat/completions",
            json=json,
        )
        async with r as resp:
            # decrease load and return response
            response = JSONResponse(content=await resp.json())
            app.state.remotes.decrease_key(remote)
            return response


@app.post("/v1/completions")
async def create_completion(request: Request):
    json = await request.json()
    async with aiohttp.ClientSession() as session:
        # get remote and increase load
        remote = app.state.remotes.get_min()
        app.state.remotes.increment_min()

        # send request
        r = session.post(
            f"{remote.url}/v1/completions",
            json=json,
        )
        async with r as resp:
            # decrease load and return response
            response = JSONResponse(content=await resp.json())
            app.state.remotes.decrease_key(remote)
            return response


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
