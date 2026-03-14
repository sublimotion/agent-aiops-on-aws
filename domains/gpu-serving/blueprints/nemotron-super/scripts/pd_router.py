#!/usr/bin/env python3
"""PD disaggregation router for SGLang.
Sends the same request to BOTH prefill and decode workers.
Prefill computes KV and transfers to decode via NIXL.
Decode generates tokens — response is streamed from decode.
"""
import asyncio
import itertools
import json
import sys
from typing import List

import aiohttp
from aiohttp import web

class PDRouter:
    def __init__(self, prefill_urls: List[str], decode_urls: List[str],
                 prefill_hosts: List[str], bootstrap_port: int = 8998, port: int = 8000):
        self.prefill_urls = prefill_urls    # HTTP URLs for API requests
        self.decode_urls = decode_urls      # HTTP URLs for API requests
        self.prefill_hosts = prefill_hosts  # Pod IPs for bootstrap_host
        self.bootstrap_port = bootstrap_port
        self.port = port
        self.room_counter = itertools.count(1)
        self.prefill_cycle = itertools.cycle(range(len(prefill_urls)))
        self.decode_cycle = itertools.cycle(range(len(decode_urls)))
        self.session = None

    async def start(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300))
        app = web.Application()
        app.router.add_get('/health', self.health)
        app.router.add_get('/v1/models', self.models)
        app.router.add_post('/v1/chat/completions', self.chat_completions)
        app.router.add_post('/v1/completions', self.completions)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        print(f"PD Router listening on 0.0.0.0:{self.port}", flush=True)
        print(f"  Prefill: {self.prefill_urls}", flush=True)
        print(f"  Decode:  {self.decode_urls}", flush=True)
        print(f"  Bootstrap hosts (prefill IPs): {self.prefill_hosts}", flush=True)
        while True:
            await asyncio.sleep(3600)

    async def health(self, request):
        return web.Response(text="OK")

    async def models(self, request):
        idx = next(self.prefill_cycle)
        url = f"{self.prefill_urls[idx]}/v1/models"
        async with self.session.get(url) as resp:
            body = await resp.read()
            return web.Response(body=body, content_type='application/json', status=resp.status)

    async def chat_completions(self, request):
        return await self._proxy_generate(request, '/v1/chat/completions')

    async def completions(self, request):
        return await self._proxy_generate(request, '/v1/completions')

    async def _proxy_generate(self, request, path):
        body = await request.json()

        # Pick prefill and decode workers
        prefill_idx = next(self.prefill_cycle)
        decode_idx = next(self.decode_cycle)
        room_id = next(self.room_counter)

        # Bootstrap host = prefill worker's IP (where bootstrap server runs on port 8998)
        bootstrap_host = self.prefill_hosts[prefill_idx]

        # Add bootstrap fields
        body['bootstrap_host'] = bootstrap_host
        body['bootstrap_port'] = self.bootstrap_port
        body['bootstrap_room'] = room_id

        prefill_url = f"{self.prefill_urls[prefill_idx]}{path}"
        decode_url = f"{self.decode_urls[decode_idx]}{path}"

        stream = body.get('stream', False)

        print(f"[room={room_id}] prefill={prefill_url} decode={decode_url} bootstrap={bootstrap_host}:{self.bootstrap_port}", flush=True)

        try:
            if stream:
                # Send to both prefill and decode concurrently
                # Stream response from decode worker (it generates tokens)
                response = web.StreamResponse(
                    status=200,
                    headers={'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache'}
                )
                await response.prepare(request)

                async def send_to_prefill():
                    """Fire and forget to prefill - it does KV computation and transfer."""
                    try:
                        async with self.session.post(prefill_url, json=body) as resp:
                            await resp.read()
                    except Exception as e:
                        print(f"[room={room_id}] Prefill error: {e}", flush=True)

                # Start prefill in background
                prefill_task = asyncio.create_task(send_to_prefill())

                # Stream from decode (which waits for KV then generates)
                async with self.session.post(decode_url, json=body) as resp:
                    async for chunk in resp.content.iter_any():
                        await response.write(chunk)

                await response.write_eof()
                await prefill_task
                return response
            else:
                # Non-streaming: send to both, return decode response
                async def send_to_prefill():
                    try:
                        async with self.session.post(prefill_url, json=body) as resp:
                            return await resp.read()
                    except Exception as e:
                        print(f"[room={room_id}] Prefill error: {e}", flush=True)
                        return None

                async def send_to_decode():
                    async with self.session.post(decode_url, json=body) as resp:
                        return resp.status, await resp.read()

                # Send to both concurrently
                prefill_task = asyncio.create_task(send_to_prefill())
                decode_status, decode_body = await send_to_decode()
                await prefill_task

                return web.Response(body=decode_body, content_type='application/json', status=decode_status)

        except Exception as e:
            print(f"[room={room_id}] Error: {e}", flush=True)
            return web.Response(
                text=json.dumps({"error": str(e)}),
                content_type='application/json',
                status=502
            )


def main():
    if len(sys.argv) < 3:
        print("Usage: pd_router.py <prefill_ip1,...> <decode_ip1,...> [port] [bootstrap_port]")
        print("  All IPs are pod IPs. URLs constructed as http://IP:30000")
        sys.exit(1)

    prefill_ips = sys.argv[1].split(',')
    decode_ips = sys.argv[2].split(',')
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
    bootstrap_port = int(sys.argv[4]) if len(sys.argv) > 4 else 8998

    prefill_urls = [f"http://{ip}:30000" for ip in prefill_ips]
    decode_urls = [f"http://{ip}:30000" for ip in decode_ips]

    router = PDRouter(prefill_urls, decode_urls, prefill_ips, bootstrap_port, port)
    asyncio.run(router.start())


if __name__ == '__main__':
    main()
