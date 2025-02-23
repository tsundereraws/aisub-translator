# AISub-Translator

## Overview
`AISub-Translator` is a powerful Python script designed to translate subtitle files (`.ass` and `.srt`) line-by-line using multiple AI providers: OpenAI, Claude, and DeepSeek. It preserves ASS file headers (including styles, script info, etc.) and processes only the dialogue lines for translation. The script also supports live updating of the output file and provides clean logging without formatting codes in the console.

## Features
- **Multi-AI Support:** Choose from `openai`, `claude`, or `deepseek` via the `--ai` argument.
- **Model Configuration:** Customize models per provider in `.env` file (supports GPT-4, Claude 3, etc.).
- **ASS Header Preservation:** For `.ass` files, all header information is preserved and only the dialogue text is translated.
- **Selective Processing:** Only dialogue lines (those starting with `Dialogue:`) are translated; empty lines are skipped.
- **Contextual Translation:** Supports providing an episode synopsis (via `--context` or `--context-file`) and uses previous dialogue context for continuity.
- **Live Output Updates:** The output file is updated live as each line is processed.
- **Clean Logging:** Console messages omit formatting codes (e.g. `{\an8}`) for readability.
- **Easy Configuration:** API keys and model selection managed via `.env` file.

## Requirements
- Tested on Python 3.13.1
- Dependencies: `openai`, `requests`, `langdetect`, `pysubs2`, `python-dotenv`

Install the required packages using:
```pip install openai requests langdetect pysubs2 python-dotenv```

## Configuration
Create a `.env` file in the project directory with your API keys and optional model configuration:
```
OPENAI_API_KEY=your_openai_api_key_here
CLAUDE_API_KEY=your_claude_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
# Optional model configurations (defaults shown)
OPENAI_MODEL=gpt-3.5-turbo
CLAUDE_MODEL=claude-v1
DEEPSEEK_MODEL=deepseek-chat
```


## Usage
Run the script from the command line with the following arguments:
```python aisub-translator.py input_file.ass --target-language en --ai deepseek```

### Command Line Arguments:
- `input_file`: Path to the subtitle file (`.ass` or `.srt`)
- `--target-language`: Target language code (e.g. `en` for English)
- `--ai`: AI provider to use. Valid values: `openai`, `claude`, `deepseek`
- `--context`: Episode synopsis for the current film/episode (already translated)
- `--context-file`: Path to a file containing the episode synopsis
- `--output-file`: Path to save the translated subtitle file
- `--verbose`: Enable verbose logging (including debug output)

## API Providers
- **OpenAI:** Uses official OpenAI API with configurable model via `OPENAI_MODEL` (default: `gpt-3.5-turbo`)
- **Claude:** Uses Anthropic's Claude API with configurable model via `CLAUDE_MODEL` (default: `claude-v1`)
- **DeepSeek:** Uses OpenAI-compatible API with configurable model via `DEEPSEEK_MODEL` (default: `deepseek-chat`)

## Examples
- **Using DeepSeek with custom model:**
```python aisub-translator.py movie.ass --target-language en --ai deepseek```

- **Using Claude 3 with episode synopsis:**
```python aisub-translator.py episode.srt --target-language fr --ai claude --context "Episode synopsis goes here."```

![Example](https://i.ibb.co/5XRMDfYJ/image.png)

## ASS File Handling
For ASS files, the script automatically preserves all header information (script info, styles, etc.). Only the dialogue text (the part after the `Dialogue:` field's initial settings) is translated.

## Support
For support regarding `aisub-translator`, join our Discord:  
[discord.gg/vostfr](https://discord.gg/vostfr)

*Note: This is maintained as a fun side project - contributions welcome!*
