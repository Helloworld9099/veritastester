# Azure Translator F0 setup for BridgeCaption

BridgeCaption uses the **standard Azure Translator text API v3.0**. The newer 2026-06-06 API has different request/response schemas; we do not silently switch versions. Azure is a **context-free reference baseline**: it receives only recognized current-utterance text.

Official Microsoft docs:
- https://learn.microsoft.com/en-us/azure/ai-services/translator/translator-how-to-signup
- https://learn.microsoft.com/en-us/azure/ai-services/translator/text-translation/how-to/use-rest-api
- https://azure.microsoft.com/en-us/pricing/details/translator/
- https://learn.microsoft.com/en-us/azure/ai-services/translator/text-translation/reference/v3/translate

## Getting the key

1. Go to https://portal.azure.com/ and sign in with a Microsoft account. Create a free Azure subscription if necessary. Account verification or billing setup may be required by Azure.
2. Search the marketplace for **Translator** and select the **single-service Translator resource**, not document translation.
3. Create/select a resource group (example: bridgecaption) and a resource name; choose **Global** region where offered.
4. Under **Pricing tier**, select **F0 Free**. Verify the selected tier on the review page before clicking Create. Do not select a paid tier accidentally.
5. Select **Review + create**, then **Create**. Once deployment succeeds, click **Go to resource**.
6. Select **Resource Management > Keys and Endpoint**. Copy **KEY 1** privately. Note the resource **Region**.
7. In a local, gitignored .env at the root of this repository, configure:

~~~dotenv
BRIDGE_TRANSLATOR=azure
AZURE_TRANSLATOR_KEY=YOUR_PRIVATE_KEY
AZURE_TRANSLATOR_REGION=
~~~

For a Global single-service resource the region field can usually be blank. If you created a regional or multi-service Azure Translator resource, specify its Azure location, for example eastus. Use the exact identifier shown on the resource. Never paste an API key into README, chat, a GitHub issue, or a commit.

The Azure F0 pricing page lists **2 million standard translation characters per month** at no charge. Check the current resource quota; other Azure services or paid tiers may cost money. A translator key is **not** a key for other Microsoft Foundry language services.

## Test it without sending private conversation

From a feature-branch checkout:

~~~bash
python -m pip install -e '.[test,eval]'
bridgecaption-check --engine azure
~~~

Without --execute, this is only a dry run. To send one **public, hard-coded sample** and measure response time:

~~~bash
bridgecaption-check --engine azure --execute
~~~

The test sends "Hola, buenos días." from Spanish into English. It does not send conversations, recordings, previous context, or credentials in console output. It reads your local .env file. You can also run the app and click **CHECK ENGINE** after selecting Azure.

If you see **HTTP 401 or 403**, check KEY 1, subscription status and region header. **HTTP 429** typically means throttling or quota; check Azure usage before retrying. A connectivity test does not establish contextual slang accuracy.

## Why Azure doesn't receive our memory model's context

The standard Azure text translation endpoint does not accept conversational LLM instructions. The app therefore omits previous utterances, optional slang hypotheses and glossary from Azure requests. This is deliberate for a fair **context-free baseline**.

Microsoft's dynamic dictionary markup is recommended only for names/compound terms, **not** for forcing uncertain slang senses; BridgeCaption does not misuse it. For experiments where our trained context model should influence output, select **Groq, Gemini or Ollama**, which accept instructions and context.

## Evaluate authorized audio

Prepare an examples.csv in the same folder as permissioned speech clips. Required columns: file,language,transcript. Optional: translation,previous_json,topic (where previous_json is a JSON list of at most four earlier utterances).

~~~bash
# Validate manifest without decoding audio or calling a provider
bridgecaption-eval-audio --manifest /path/to/examples.csv

# Decode/recognize local speech only
bridgecaption-eval-audio --manifest /path/to/examples.csv --execute

# Additionally translate recognized text with Azure; requires permission
bridgecaption-eval-audio --manifest /path/to/examples.csv --execute \
  --translate --engine azure --allow-cloud-transcripts
~~~

Reports under results/ include aggregate WER/CER and optional chrF and timing, **not raw audio or full transcript contents**. These scripts are for isolated clips; they do not establish sustained live caption latency or error recovery. Read the training-readiness guide before claiming hackathon evaluation results.
