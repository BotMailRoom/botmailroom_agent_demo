# Local Development

1. Start ngrok to create a public URL for your local server

```bash
ngrok http localhost:8000
```

> Note: If you have another method to forward webhooks to your local machine, you can use that instead.

2. Setup a BotMailRoom Inbox

   - Visit the [BotMailRoom Quickstart Guide](https://docs.botmailroom.com/documentation/quickstart#web-application)
   - Use the ngrok URL from step 1 (e.g. `https://d0ca..9737.ngrok-free.app/receive-email`) as the webhook URL. Be sure to include `/receive-email` at the end of the URL.
   - Add to `.env`: `BOTMAILROOM_WEBHOOK_SECRET=your_secret`

3. Create a BotMailRoom API Key

   - Visit the [API Key Creation Guide](https://docs.botmailroom.com/documentation/quickstart#1-create-an-api-key-if-you-dont-already-have-one)
   - Add to `.env`: `BOTMAILROOM_API_KEY=your_key`

4. Get an OpenAI API Key

   - Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
   - Add to `.env`: `OPENAI_API_KEY=your_key`

5. Optional: Get an Exa API Key

   If you provide an Exa API key, the agent will be able to perform web searches.

   - Visit [Exa Dashboard](https://dashboard.exa.ai/api-keys)
   - Add to `.env`: `EXA_API_KEY=your_key`

6. Install dependencies

```bash
pip install -r requirements.txt
```

7. Start the server

```bash
bash start_server.sh
```

8. Test the Setup
   Send an email to your BotMailRoom Inbox (created in step 2) with a sample query:

```
To: your-inbox@botmailroom.com
Subject: New Task

Hi Bot,

I'm looking to attend python / data focused conferences in 2025. Could you research some and suggest 5ish I should go to, make sure the list is nicely formatted.

Best,
[Your Name]
```
