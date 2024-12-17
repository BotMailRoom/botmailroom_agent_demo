import json
import logging
from contextlib import asynccontextmanager
from typing import Optional, cast

import aiosqlite
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
    database_url: str = "sqlite+aiosqlite:///./sql_app.db"


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

valid_tool_names = ", ".join([tool["function"]["name"] for tool in tools])

system_prompt = f"""
- Respond to the user's instructions carefully
- The user is only able to respond to emails, so if you have a message to send, use the `botmailroom_send_email` tool.
- Email content should be formatted as email compliant html.
- When sending emails, prefer responding to an existing email thread over starting a new one.
- Only use one tool at a time
- Always respond with a tool call - the only valid tool names are {valid_tool_names}
"""

# Initialize FastAPI app

db: Optional[aiosqlite.Connection] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db
    db = await aiosqlite.connect(settings.database_url)
    # create a chats table with id (varchar, pkey), chat_thread (json) if it doesn't exist
    await db.execute(
        "CREATE TABLE IF NOT EXISTS chats (id VARCHAR PRIMARY KEY, chat_thread JSON)"
    )
    await db.commit()
    yield
    await db.close()


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


async def handle_model_call(chat_id: str, chat_thread: list[dict]):
    cycle_count = 0
    end = False
    while cycle_count <= settings.max_response_cycles:
        cycle_count += 1
        output = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=chat_thread,  # type: ignore
            tools=tools,  # type: ignore
        )

        # execute the tool call if it exists
        if output.choices[0].message.tool_calls:
            chat_thread.append(output.choices[0].message.model_dump())
            for tool_call in output.choices[0].message.tool_calls:
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
                    end = True
                elif tool_call.function.name.startswith("web_search"):
                    tool_output = exa_search(arguments["query"])
                else:
                    raise ValueError(
                        f"Unknown tool: {tool_call.function.name}"
                    )
                chat_thread.append(
                    {
                        "role": "tool",
                        "content": tool_output,
                        "name": tool_call.function.name,
                        "tool_call_id": tool_call.id,
                    }
                )
                logging.info(f"\033[93mTool output:\033[0m {tool_output}")
        else:
            content = output.choices[0].message.content
            logging.warning(
                f"\033[93mInvalid response from model:\033[0m {content}"
            )
            chat_thread.append(
                {
                    "role": "user",
                    "content": "Please respond with a tool call",
                }
            )

        if end:
            break

        if db:
            await db.execute(
                "UPDATE chats SET chat_thread = ? WHERE chat_id = ?",
                (json.dumps(chat_thread), chat_id),
            )
            await db.commit()


async def handle_email(email_payload: EmailPayload):
    logging.info(
        f"\033[93mReceived email\033[0m from {email_payload.from_address.address}"
    )
    if (
        email_payload.previous_emails is not None
        and len(email_payload.previous_emails) > 0
    ):
        chat_id = email_payload.previous_emails[0].id
    else:
        chat_id = email_payload.id

    chat_thread = [{"role": "system", "content": system_prompt}]
    if db:
        # check if chat_id exists
        cursor = await db.execute(
            "SELECT * FROM chats WHERE id = ?", (chat_id,)
        )
        result = await cursor.fetchone()
        if result is not None:
            chat_thread = cast(list[dict], json.loads(result[1]))

    chat_thread.append(
        {"role": "user", "content": email_payload.thread_prompt}
    )
    await handle_model_call(chat_id, chat_thread)


@app.post("/receive-email", status_code=204)
async def receive_email(
    background_tasks: BackgroundTasks,
    email_payload: EmailPayload = Depends(_validate_and_parse_email),
):
    # move to background task to respond to webhook
    background_tasks.add_task(handle_email, email_payload)
    return Response(status_code=204)
