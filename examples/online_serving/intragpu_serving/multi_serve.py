from vllm.benchmarks.lib.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS, OPENAI_COMPATIBLE_BACKENDS, RequestFuncInput,
    RequestFuncOutput)
from vllm.benchmarks.lib.ready_checker import wait_for_endpoint
import aiohttp
import argparse
import asyncio

def add_cli_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--endpoint-type",
        type=str,
        default="openai",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="The label (prefix) of the benchmark results. If not specified, "
        "the endpoint type will be used as the label.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    # Use 127.0.0.1 here instead of localhost to force the use of ipv4
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--ready-check-timeout-sec",
        type=int,
        default=600,
        help="Maximum time to wait for the endpoint to become ready "
        "in seconds (default: 600 seconds / 10 minutes).",
    )

def main():
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args()
    asyncio.run(main_async(args))

async def main_async(args: argparse.Namespace):
    print("here")
    print(args)
    request_func = ASYNC_REQUEST_FUNCS[args.endpoint_type]

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"


    connector = aiohttp.TCPConnector(
        limit= 0,
        limit_per_host= 0,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=60,
        enable_cleanup_closed=True,
        force_close=False,
        ssl=("https://" in api_url),
    )

    session = aiohttp.ClientSession(
        connector=connector,
        trust_env=True,
        timeout=aiohttp.ClientTimeout(total=6 * 60 * 60),
    )

    semaphore = None #asyncio.Semaphore(max_concurrency) if max_concurrency else None
    async def limited_request_func(request_func_input):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, session=session)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, session=session)

    test_prompts=[50*"San Francisco is a " for i in range(5)]
    tasks: list[asyncio.Task] = []

    for test_prompt in test_prompts:
        test_prompt_len = len(test_prompt)
        test_output_len = 10

        test_input = RequestFuncInput(
            model=args.model,
            model_name=args.model,
            prompt=test_prompt,
            api_url=api_url,
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            #logprobs=logprobs,
            #multi_modal_content=test_mm_content,
            #ignore_eos=ignore_eos,
            #extra_body=extra_body,
        )
        task = limited_request_func(request_func_input=test_input)
        tasks.append(asyncio.create_task(task))
    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)
    print(outputs)
    
    await session.close()

    #return test_output.generated_text

if __name__ == "__main__":
    main()
