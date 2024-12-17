import express from "express";
import dotenv from "dotenv";
import OpenAI from "openai";
import Exa from "exa-js";
import { BotMailRoom, verifyWebhookSignature } from "botmailroom";
import {
  EmailPayload,
  EmailPayloadSchema,
  ToolSchemaType,
} from "botmailroom/dist/types";
import sqlite3 from "sqlite3";
import { open } from "sqlite";

dotenv.config();

// Settings
const settings = {
  botmailroomWebhookSecret: process.env.BOTMAILROOM_WEBHOOK_SECRET,
  botmailroomApiKey: process.env.BOTMAILROOM_API_KEY!,
  openaiApiKey: process.env.OPENAI_API_KEY!,
  exaApiKey: process.env.EXA_API_KEY,
  maxResponseCycles: 10,
  databaseUrl: process.env.DATABASE_URL || "./sql_app.db",
};

// Initialize clients
const app = express();

const openai = new OpenAI({
  apiKey: settings.openaiApiKey,
});

const exa = settings.exaApiKey ? new Exa(settings.exaApiKey) : undefined;
const bmr = new BotMailRoom(settings.botmailroomApiKey);

// System prompt and tools
const systemPrompt = `
- Respond to the user's instructions carefully
- The user is only able to respond to emails, so if you have a message to send, use the \`botmailroom_send_email\` tool.
- Email content should be formatted as email compliant html.
- When sending emails, prefer responding to an existing email thread over starting a new one.
- Only use one tool at a time
- Always respond with a tool call
`;

const tools: any[] = [];
async function initializeTools() {
  tools.push(
    ...(await bmr.getTools({
      toolSchemaType: ToolSchemaType.OpenAI,
      toolsToInclude: ["botmailroom_send_email"],
    }))
  );
  if (exa) {
    tools.push({
      type: "function",
      function: {
        name: "web_search",
        description:
          "Perform a search query on the web, and retrieve the most relevant URLs/web data.",
        parameters: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "The search query to perform.",
            },
          },
          required: ["query"],
        },
      },
    });
  }
}

// Helper functions
const exaSearch = async (query: string): Promise<string> => {
  if (!exa) {
    throw new Error("Exa client not initialized");
  }
  const response = await exa.searchAndContents(query, {
    type: "auto",
    highlights: true,
    text: true,
  });
  return response.results
    .map((result) => {
      return `URL: ${result.url}\nTitle: ${result.title}\nAuthor: ${result.author}\nContent: ${result.text}`;
    })
    .join("\n\n");
};

// Add database connection
let db: any = null;

// Initialize database
async function initializeDatabase() {
  db = await open({
    filename: settings.databaseUrl,
    driver: sqlite3.Database,
  });

  await db.exec(`
    CREATE TABLE IF NOT EXISTS chats (
      id VARCHAR PRIMARY KEY,
      chat_thread JSON
    )
  `);
}

// Update handleModelCall to match Python version
async function handleModelCall(chatId: string, chatThread: any[]) {
  let cycleCount = 0;
  let end = false;

  while (cycleCount <= settings.maxResponseCycles) {
    cycleCount++;
    const output = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: chatThread,
      tools,
    });

    const message = output.choices[0].message;

    if (message.tool_calls) {
      chatThread.push(message);

      for (const toolCall of message.tool_calls) {
        console.log(
          `\x1b[93mTool call:\x1b[0m ${toolCall.function.name} \x1b[93mwith args:\x1b[0m ${toolCall.function.arguments}`
        );

        const arguments_ = JSON.parse(toolCall.function.arguments);
        let toolOutput: string;

        if (toolCall.function.name.startsWith("botmailroom_")) {
          toolOutput = await bmr.executeTool(
            toolCall.function.name,
            arguments_,
            true,
            true
          );
          end = true;
        } else if (toolCall.function.name === "web_search") {
          toolOutput = await exaSearch(arguments_.query);
        } else {
          throw new Error(`Unknown tool: ${toolCall.function.name}`);
        }

        chatThread.push({
          role: "tool",
          content: toolOutput,
          name: toolCall.function.name,
          tool_call_id: toolCall.id,
        });
        console.log(`\x1b[93mTool output:\x1b[0m ${toolOutput}`);
      }
    } else {
      const content = message.content;
      console.warn(`\x1b[93mInvalid response from model:\x1b[0m ${content}`);
      chatThread.push({
        role: "user",
        content: "Please respond with a tool call",
      });
    }

    if (end) break;

    if (db) {
      await db.run(
        "UPDATE chats SET chat_thread = ? WHERE id = ?",
        JSON.stringify(chatThread),
        chatId
      );
    }
  }
}

// Update handleEmail to match Python version
async function handleEmail(emailPayload: EmailPayload) {
  console.log(
    `\x1b[93mReceived email\x1b[0m from ${emailPayload.from_address.address}`
  );

  const chatId = emailPayload.previous_emails?.length
    ? emailPayload.previous_emails[0].id
    : emailPayload.id;

  let chatThread = [{ role: "system", content: systemPrompt }];

  if (db) {
    const existingChat = await db.get(
      "SELECT * FROM chats WHERE id = ?",
      chatId
    );
    if (existingChat) {
      chatThread = JSON.parse(existingChat.chat_thread);
    }
  }

  chatThread.push({ role: "user", content: emailPayload.thread_prompt });
  await handleModelCall(chatId, chatThread);
}

// Routes
app.get("/healthz", (_, res) => {
  res.json({ status: "ok" });
});

app.post("/receive-email", async (req, res) => {
  const chunks: Buffer[] = [];
  req.on("data", (chunk: Buffer) => {
    chunks.push(chunk);
  });

  await new Promise((resolve) => req.on("end", resolve));
  const rawBody = Buffer.concat(chunks);

  let payload: EmailPayload;
  if (settings.botmailroomWebhookSecret) {
    const signatureHeader = req.headers["x-signature"]?.toString() ?? "";
    if (!signatureHeader) {
      throw new Error("No signature header found");
    }
    payload = verifyWebhookSignature(
      signatureHeader,
      rawBody,
      settings.botmailroomWebhookSecret,
      { minutes: 10 }
    );
  } else {
    console.warn(
      "No botmailroom_webhook_secret found in settings, skipping signature verification"
    );
    payload = EmailPayloadSchema.parse(JSON.parse(rawBody.toString()));
  }

  // Handle email in background
  handleEmail(payload).catch((error) => {
    console.error("Error handling email:", error);
  });

  res.status(204).send();
});

const PORT = process.env.PORT || 8000;
Promise.all([initializeTools(), initializeDatabase()])
  .then(() => {
    app.listen(PORT, () => {
      console.log(`Server is running on port ${PORT}`);
    });
  })
  .catch((error) => {
    console.error("Error during initialization:", error);
    process.exit(1);
  });
