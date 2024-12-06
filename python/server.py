import json
import logging
from typing import Optional

import openai
from botmailroom import BotMailRoom, EmailPayload, verify_webhook_signature
from dotenv import load_dotenv
from exa_py import Exa
from fastapi import BackgroundTasks, Depends, FastAPI, Request, Response
from pydantic_settings import BaseSettings

# Load settings

load_dotenv()


class Settings(BaseSettings):
    botmailroom_webhook_secret: Optional[str] = None
    botmailroom_api_key: str
    openai_api_key: str
    exa_api_key: Optional[str] = None
    max_response_cycles: int = 10


settings = Settings()  # type: ignore

# Initialize logging


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger().setLevel(
        level=logging.INFO,
    )


setup_logging()

# Initialize clients
botmailroom_client = BotMailRoom(api_key=settings.botmailroom_api_key)
openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
exa_client = (
    Exa(api_key=settings.exa_api_key) if settings.exa_api_key else None
)

# Initialize agent

system_prompt = """
- Follow the user's instructions carefully
- If the task was sent via email, respond to that email
    - Email content should be formatted as email compliant html
- Always start by creating a set of steps to complete the task given by the user.
- Always respond with one of the following:
    - A tool call
    - `PLAN` followed by a description of the steps to complete the task
    - `WAIT` to wait for a response to an email
    - `DONE` to indicate that the task is complete
"""
chat = [
    {"role": "system", "content": system_prompt},
]
tools = botmailroom_client.get_tools(
    tools_to_include=["botmailroom_send_email"]
)
if exa_client:
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Perform a search query on the web, and retrieve the most relevant URLs/web data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to perform.",
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    )

# Initialize FastAPI app
app = FastAPI()


@app.get("/healthz", include_in_schema=False)
def health():
    return {"status": "ok"}


async def _validate_and_parse_email(request: Request) -> EmailPayload:
    body = await request.body()
    if settings.botmailroom_webhook_secret is None:
        logging.warning(
            "No botmailroom_webhook_secret found in settings, skipping signature verification"
        )
        payload = EmailPayload.model_validate_json(body)
    else:
        payload = verify_webhook_signature(
            request.headers["X-Signature"],
            body,
            settings.botmailroom_webhook_secret,
        )
    return payload


def exa_search(query: str) -> str:
    if not exa_client:
        raise ValueError("Exa client not initialized")
    response = exa_client.search_and_contents(
        query=query, type="auto", highlights=True
    )
    return "\n\n".join([str(result) for result in response.results])


async def handle_model_call():
    cycle_count = 1
    while cycle_count <= settings.max_response_cycles:
        output = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=chat,  # type: ignore
            tools=tools,  # type: ignore
        )

        # execute the tool call if it exists
        if output.choices[0].message.tool_calls:
            chat.append(output.choices[0].message.model_dump())
            tool_call = output.choices[0].message.tool_calls[0]
            logging.info(
                f"\033[93mTool call:\033[0m {tool_call.function.name} \033[93mwith args:\033[0m {tool_call.function.arguments}"
            )
            arguments = json.loads(tool_call.function.arguments)
            if tool_call.function.name.startswith("botmailroom_"):
                tool_output = botmailroom_client.execute_tool(
                    tool_call.function.name,
                    arguments,
                    enforce_str_output=True,
                )
            elif tool_call.function.name.startswith("web_search"):
                tool_output = exa_search(arguments["query"])
            else:
                raise ValueError(f"Unknown tool: {tool_call.function.name}")
            chat.append(
                {
                    "role": "tool",
                    "content": tool_output,
                    "tool_call_id": tool_call.id,
                }
            )
            logging.info(f"\033[93mTool output:\033[0m {tool_output}")
        else:
            content = output.choices[0].message.content
            chat.append({"role": "assistant", "content": content or ""})
            if content is None:
                logging.warning(
                    f"\033[93mInvalid response from model:\033[0m {content}"
                )
                chat.append(
                    {
                        "role": "user",
                        "content": "Please respond with either a tool call, PLAN, WAIT, or DONE",
                    }
                )
            elif content.startswith("PLAN"):
                logging.info(f"\033[93mPlan:\033[0m {content[4:]}")
                chat.append(
                    {
                        "role": "user",
                        "content": "Looks like a good plan, let's do it!",
                    }
                )
            elif content.startswith("WAIT"):
                logging.info("\033[93mWaiting for user response\033[0m")
                return
            elif content.startswith("DONE"):
                logging.info("\033[93mTask complete\033[0m")
                # clear chat
                global chat
                chat = chat[:1]
                return
            else:
                logging.warning(
                    f"\033[93mInvalid response from model:\033[0m {content}"
                )
                chat.append(
                    {
                        "role": "user",
                        "content": "Please respond with either a tool call, PLAN, WAIT, or DONE",
                    }
                )

        cycle_count += 1


async def handle_email(email_payload: EmailPayload):
    logging.info(
        f"\033[93mReceived email\033[0m from {email_payload.from_address.address}"
    )
    chat.append({"role": "user", "content": email_payload.thread_prompt})
    await handle_model_call()


@app.post("/receive-email", status_code=204)
async def receive_email(
    background_tasks: BackgroundTasks,
    email_payload: EmailPayload = Depends(_validate_and_parse_email),
):
    # move to background task to respond to webhook
    background_tasks.add_task(handle_email, email_payload)
    return Response(status_code=204)
