import subprocess
import time
import requests
import openai
import asyncio
import aiohttp
import os
import shlex


class SGlangEngine:
    def __init__(
        self,
        model=os.getenv("MODEL_PATH"),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 30000)),
    ):
        self.model = model
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
        self.process = None

    def start_server(self):
        command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        # Dictionary of all possible options and their corresponding env var names
        options = {
            "MODEL_PATH": "--model-path",
            "TOKENIZER_PATH": "--tokenizer-path",
            "TOKENIZER_MODE": "--tokenizer-mode",
            "LOAD_FORMAT": "--load-format",
            "DTYPE": "--dtype",
            "CONTEXT_LENGTH": "--context-length",
            "QUANTIZATION": "--quantization",
            "SERVED_MODEL_NAME": "--served-model-name",
            "CHAT_TEMPLATE": "--chat-template",
            "MEM_FRACTION_STATIC": "--mem-fraction-static",
            "MAX_RUNNING_REQUESTS": "--max-running-requests",
            "MAX_TOTAL_TOKENS": "--max-total-tokens",
            "CHUNKED_PREFILL_SIZE": "--chunked-prefill-size",
            "MAX_PREFILL_TOKENS": "--max-prefill-tokens",
            "SCHEDULE_POLICY": "--schedule-policy",
            "SCHEDULE_CONSERVATIVENESS": "--schedule-conservativeness",
            "TENSOR_PARALLEL_SIZE": "--tensor-parallel-size",
            "STREAM_INTERVAL": "--stream-interval",
            "RANDOM_SEED": "--random-seed",
            "LOG_LEVEL": "--log-level",
            "LOG_LEVEL_HTTP": "--log-level-http",
            "API_KEY": "--api-key",
            "FILE_STORAGE_PATH": "--file-storage-path",
            "DATA_PARALLEL_SIZE": "--data-parallel-size",
            "LOAD_BALANCE_METHOD": "--load-balance-method",
            "ATTENTION_BACKEND": "--attention-backend",
            "KV_CACHE_DTYPE": "--kv-cache-dtype",
            "SAMPLING_BACKEND": "--sampling-backend",
            "TOOL_CALL_PARSER": "--tool-call-parser"
        }

        # Boolean flags
        boolean_flags = [
            "SKIP_TOKENIZER_INIT",
            "TRUST_REMOTE_CODE",
            "LOG_REQUESTS",
            "SHOW_TIME_COST",
            "DISABLE_RADIX_CACHE",
            "DISABLE_CUDA_GRAPH",
            "DISABLE_OUTLINES_DISK_CACHE",
            "ENABLE_TORCH_COMPILE",
            "ENABLE_P2P_CHECK",
            "ENABLE_FLASHINFER_MLA",
            "TRITON_ATTENTION_REDUCE_IN_FP32",
        ]

        # Add options from environment variables only if they are set
        for env_var, option in options.items():
            value = os.getenv(env_var)
            if value is not None and value != "":
                command.extend([option, value])

        # Add boolean flags only if they are set to true
        for flag in boolean_flags:
            if os.getenv(flag, "").lower() in ("true", "1", "yes"):
                command.append(f"--{flag.lower().replace('_', '-')}")

        # Generic escape hatch: append any extra SGLang launch flags verbatim, so a
        # future flag can be used without re-forking. Parsed safely with shlex.
        #
        # Precedence: flags already set above via their dedicated env vars WIN. A flag
        # in EXTRA_ARGS that duplicates an already-set flag is skipped (along with its
        # value), so EXTRA_ARGS can never clobber an explicit setting or crash argparse
        # with a duplicate. Flags not already set pass through unchanged.
        # Note: give EXTRA_ARGS values that begin with '-' in --flag=value form.
        extra_raw = os.getenv("EXTRA_ARGS", "")
        if extra_raw.strip():
            try:
                extra_tokens = shlex.split(extra_raw)
            except ValueError as exc:
                print(f"Warning: could not parse EXTRA_ARGS ({exc}); ignoring it.")
                extra_tokens = []

            # Long-option names already in the command (handles --flag and --flag=value).
            existing_flags = {
                tok.split("=", 1)[0] for tok in command if tok.startswith("-")
            }

            skipping_value = False  # currently dropping the value(s) of a skipped flag
            for tok in extra_tokens:
                if tok.startswith("-"):
                    name = tok.split("=", 1)[0]
                    if name in existing_flags:
                        # Drop this flag; if it's "--flag value" form, drop its value too.
                        skipping_value = "=" not in tok
                        print(
                            f"Warning: EXTRA_ARGS flag '{name}' is already set via a "
                            f"dedicated env var; ignoring the EXTRA_ARGS copy."
                        )
                        continue
                    skipping_value = False
                    existing_flags.add(name)
                    command.append(tok)
                elif not skipping_value:
                    command.append(tok)

        self.process = subprocess.Popen(command, stdout=None, stderr=None)
        print(f"Server started with PID: {self.process.pid}")

    def wait_for_server(self, timeout=900, interval=5):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/v1/models")
                if response.status_code == 200:
                    print("Server is ready!")
                    return True
            except requests.RequestException:
                pass
            time.sleep(interval)
        raise TimeoutError("Server failed to start within the timeout period.")

    def shutdown(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("Server shut down.")


class OpenAIRequest:
    def __init__(self, base_url="http://0.0.0.0:30000/v1", api_key="EMPTY"):
        self.client = openai.Client(base_url=base_url, api_key=api_key)

    async def request_chat_completions(
        self,
        model="default",
        messages=None,
        max_tokens=100,
        stream=False,
        frequency_penalty=0.0,
        n=1,
        stop=None,
        temperature=1.0,
        top_p=1.0,
    ):
        if messages is None:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": "List 3 countries and their capitals."},
            ]

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=stream,
            frequency_penalty=frequency_penalty,
            n=n,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
        )

        if stream:
            async for chunk in response:
                yield chunk.to_dict()
        else:
            yield response.to_dict()

    async def request_completions(
        self,
        model="default",
        prompt="The capital of France is",
        max_tokens=100,
        stream=False,
        frequency_penalty=0.0,
        n=1,
        stop=None,
        temperature=1.0,
        top_p=1.0,
    ):
        response = self.client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            stream=stream,
            frequency_penalty=frequency_penalty,
            n=n,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
        )

        if stream:
            async for chunk in response:
                yield chunk.to_dict()
        else:
            yield response.to_dict()

    async def get_models(self):
        response = await self.client.models.list()
        return response
