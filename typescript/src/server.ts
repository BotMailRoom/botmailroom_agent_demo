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

dotenv.config();

// Settings
const settings = {
  botmailroomWebhookSecret: process.env.BOTMAILROOM_WEBHOOK_SECRET,
  botmailroomApiKey: process.env.BOTMAILROOM_API_KEY!,
  openaiApiKey: process.env.OPENAI_API_KEY!,
  exaApiKey: process.env.EXA_API_KEY,
  maxResponseCycles: 10,
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
- Follow the user's instructions carefully
- If the task was sent via email, respond to that email
    - Email content should be formatted as email compliant html
- Always start by creating a set of steps to complete the task given by the user.
- Always respond with one of the following:
    - A tool call
    - \`PLAN\` followed by a description of the steps to complete the task
    - \`WAIT\` to wait for a response to an email
    - \`DONE\` to indicate that the task is complete
`;

const chat: OpenAI.Chat.ChatCompletionMessageParam[] = [
  { role: "system", content: systemPrompt },
];

const tools: any[] = [];
async function initializeTools() {
  tools.push(
    ...(await bmr.getTools(ToolSchemaType.OpenAI, ["botmailroom_send_email"]))
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

async function handleModelCall() {
  let cycleCount = 1;
  while (cycleCount <= settings.maxResponseCycles) {
    const output = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: chat,
      tools,
    });

    const message = output.choices[0].message;

    if (message.tool_calls) {
      chat.push(message);
      const toolCall = message.tool_calls[0];
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
      } else if (toolCall.function.name === "web_search") {
        toolOutput = await exaSearch(arguments_.query);
      } else {
        throw new Error(`Unknown tool: ${toolCall.function.name}`);
      }

      chat.push({
        role: "tool",
        content: toolOutput,
        tool_call_id: toolCall.id,
      });
      console.log(`\x1b[93mTool output:\x1b[0m ${toolOutput}`);
    } else {
      const content = message.content;
      chat.push({ role: "assistant", content: content || "" });

      if (!content) {
        console.warn(`\x1b[93mInvalid response from model:\x1b[0m ${content}`);
        chat.push({
          role: "user",
          content:
            "Please respond with either a tool call, PLAN, WAIT, or DONE",
        });
      } else if (content.startsWith("PLAN")) {
        console.log(`\x1b[93mPlan:\x1b[0m ${content.slice(4)}`);
        chat.push({
          role: "user",
          content: "Looks like a good plan, let's do it!",
        });
      } else if (content.startsWith("WAIT")) {
        console.log("\x1b[93mWaiting for user response\x1b[0m");
        return;
      } else if (content.startsWith("DONE")) {
        console.log("\x1b[93mTask complete\x1b[0m");
        // clear chat
        chat.splice(1);
        return;
      } else {
        console.warn(`\x1b[93mInvalid response from model:\x1b[0m ${content}`);
        chat.push({
          role: "user",
          content:
            "Please respond with either a tool call, PLAN, WAIT, or DONE",
        });
      }
    }

    cycleCount++;
  }
}

async function handleEmail(emailPayload: EmailPayload) {
  console.log(
    `\x1b[93mReceived email\x1b[0m from ${emailPayload.from_address.address}`
  );
  chat.push({ role: "user", content: emailPayload.thread_prompt });
  await handleModelCall();
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
initializeTools()
  .then(() => {
    app.listen(PORT, () => {
      console.log(`Server is running on port ${PORT}`);
    });
  })
  .catch((error) => {
    console.error("Error initializing tools:", error);
    process.exit(1);
  });
