# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import ipaddress
import logging
import os
import socket

import uvicorn
from fastapi import FastAPI

logger = logging.getLogger(__file__)


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def get_free_port(address: str) -> tuple[int, socket.socket]:
    family = socket.AF_INET
    if is_valid_ipv6_address(address):
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((address, 0))

    port = sock.getsockname()[1]
    return port, sock


async def run_unvicorn(app: FastAPI, server_args, server_address, max_retries=5) -> tuple[int, asyncio.Task]:
    server_port, server_task = None, None

    for i in range(max_retries):
        sock = None
        try:
            server_port, sock = get_free_port(server_address)
            app.server_args = server_args
            config = uvicorn.Config(app, host=server_address, port=server_port, log_level="warning")
            server = uvicorn.Server(config)

            # Keep the complete Uvicorn lifecycle in the background task.  The
            # previous implementation set should_exit before serve(), so newer
            # Uvicorn versions immediately shut the listener down and then ran
            # main_loop() against an already-closed server.
            server_task = asyncio.create_task(server.serve(sockets=[sock]))

            # Do not publish the port until Uvicorn has completed startup.
            while not server.started:
                if server_task.done():
                    # Propagate startup failures so the retry loop can select a
                    # fresh port. task.result() raises the original exception.
                    server_task.result()
                    raise RuntimeError(f"HTTP server exited during startup on port {server_port}")
                await asyncio.sleep(0.01)
            break
        except (OSError, RuntimeError, SystemExit) as e:
            logger.error(f"Failed to start HTTP server on port {server_port} at try {i}, error: {e}")
            if server_task is not None and not server_task.done():
                server_task.cancel()
                await asyncio.gather(server_task, return_exceptions=True)
            if sock is not None:
                sock.close()
    else:
        logger.error(f"Failed to start HTTP server after {max_retries} retries, exiting...")
        os._exit(-1)

    logger.info(f"HTTP server started on port {server_port}")
    return server_port, server_task
