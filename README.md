# AISub-Translator

## Overview
`AISub-Translator` is a powerful Python script designed to translate subtitle files (`.ass` and `.srt`) line-by-line using multiple AI providers: OpenAI, Claude, OpenRouter, MistralAI, DeepSeek and Ollama. It preserves ASS file headers (including styles, script info, etc.) and processes only the dialogue lines for translation. The script also supports live updating of the output file and provides clean logging without formatting codes in the console.

## Features
- **Multi-AI Support:** Choose from `openai`, `claude`, `openrouter`, `mistral`, `deepseek` or `ollama` via the `--ai` argument.
- **Model Configuration:** Customize models per provider in `.env` file (supports GPT-4, Claude 3, etc.).
- **ASS Header Preservation:** For `.ass` files, all header information is preserved and only the dialogue text is translated.
- **Selective Processing:** Only dialogue lines (those starting with `Dialogue:`) are translated; empty lines are skipped.
- **Contextual Translation:** Supports providing an episode synopsis (via `--context` or `--context-file`) and uses previous dialogue context for continuity.
- **Live Output Updates:** The output file is updated live as each line is processed.
- **Clean Logging:** Console messages omit formatting codes (e.g. `{\an8}`) for readability.
- **Easy Configuration:** API keys and model selection managed via `.env` file.
- **Post-Translation Review File:** For each input file, a CSV review file (e.g., filename_en_review.csv) is generated, containing columns for line number, original text, and translated text.
- **Multiple File Translation:** The input_file argument is renamed to `input_files` with `nargs='+'`, allowing one or more subtitle files to be processed.

## Requirements
- Tested on Python 3.13.1
- Dependencies: `openai`, `requests`, `langdetect`, `pysubs2`, `python-dotenv`, `mistralai`, `ollama`

Install the required packages using:
```pip install openai requests langdetect pysubs2 python-dotenv mistralai ollama```

## Configuration
Create a `.env` file in the project directory with your API keys and optional model configuration:
```
# API Keys
OPENAI_API_KEY=your-openai-key-here
CLAUDE_API_KEY=your-claude-key-here
DEEPSEEK_API_KEY=your-deepseek-key-here
OPENROUTER_API_KEY=your_openrouter_api_key
MISTRAL_API_KEY=your_mistral_api_key_here
OLLAMA_URL=http://localhost:11434

# Model Configuration
OPENAI_MODEL=gpt-4.5-turbo
CLAUDE_MODEL=claude-v1
DEEPSEEK_MODEL=deepseek-chat
OPENROUTER_MODEL=deepseek/deepseek-chat:free
MISTRAL_MODEL=mistral-small-latest
OLLAMA_MODEL=mistral-small
```


## Usage
Run the script from the command line with the following arguments:
```python aisub-translator.py input1.srt input2.srt --target-language en --ai openai --model gpt-4 --temperature 1.0 --context "Episode about a samurai's journey."```

### Command Line Arguments:
- `input_file`: Path to the subtitle file (`.ass` or `.srt`)
- `--target-language`: Target language code (e.g. `en` for English)
- `--ai`: AI provider to use. Valid values: `openai`, `claude`, `deepseek`, `openrouter`, `mistral`, `ollama`.
- `--context`: Episode synopsis for the current film/episode (already translated)
- `--context-file`: Path to a file containing the episode synopsis
- `--output-file`: Path to save the translated subtitle file
- `--verbose`: Enable verbose logging (including debug output)
- `--model`: Choose a specific model that overrites the one in `.env`
- `--temperature`: Choose a specific temperature suited for your needs, by default is set to `1.3`
  

## API Providers
- **OpenAI:** Uses official OpenAI API with configurable model via `OPENAI_MODEL` (default: `gpt-3.5-turbo`)
- **Claude:** Uses Anthropic's Claude API with configurable model via `CLAUDE_MODEL` (default: `claude-v1`)
- **DeepSeek:** Uses OpenAI-compatible API with configurable model via `DEEPSEEK_MODEL` (default: `deepseek-chat`)
- **OpenRouter:** Uses OpenRouter API with configurable model via `OPENROUTER_MODEL` (default: `deepseek/deepseek-chat:free`)
- **MistralAI:** Uses MistralAI API with configurable model via `MISTRAL_MODEL` (default: `mistral-small-latest`)
- **Ollama:** Uses [Ollama](https://ollama.com) local server with configurable model via `OLLAMA_MODEL` (default:`mistral-small` ; Model list available [here](https://ollama.com/library))

## Examples
- **Using DeepSeek with custom model:**
```python aisub-translator.py movie.ass --target-language en --ai deepseek```

- **Using Claude 3 with episode synopsis:**
```python aisub-translator.py episode.srt --target-language fr --ai claude --context "Episode synopsis goes here."```

![Example](https://i.ibb.co/WvVdtKkX/image.png)

## ASS File Handling
For ASS files, the script automatically preserves all header information (script info, styles, etc.). Only the dialogue text (the part after the `Dialogue:` field's initial settings) is translated.

## Support
For support regarding `aisub-translator`, join our Discord:  
[discord.gg/vostfr](https://discord.gg/vostfr)

*Note: This is maintained as a fun side project - contributions welcome!*
