import os
import re
import sys
import argparse
import logging
import langdetect
import contextlib
from dotenv import load_dotenv
import pysubs2
import openai
from openai import OpenAI
import http.client
import requests
from tqdm import tqdm
import csv

openai.log = "error"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
VERBOSE = False

def remove_formatting(text):
    return re.sub(r'\{.*?\}', '', text).strip()

@contextlib.contextmanager
def suppress_fd_output():
    try:
        fd_stdout = sys.stdout.fileno()
        fd_stderr = sys.stderr.fileno()
    except Exception:
        yield
        return
    old_stdout_fd = os.dup(fd_stdout)
    old_stderr_fd = os.dup(fd_stderr)
    try:
        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), fd_stdout)
            os.dup2(devnull.fileno(), fd_stderr)
        yield
    finally:
        os.dup2(old_stdout_fd, fd_stdout)
        os.dup2(old_stderr_fd, fd_stderr)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)

def detect_language(text):
    try:
        return langdetect.detect(text)
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return None

def split_text_with_formatting(text):
    pattern = r'(\{.*?\})'
    parts = re.split(pattern, text)
    result = []
    for part in parts:
        if part:
            result.append((False, part)) if part.startswith('{') and part.endswith('}') else result.append((True, part))
    return result

def extract_leading_formatting(text):
    m = re.match(r'^(\{[^}]+\})\s*(.*)', text)
    return (m.group(1), m.group(2)) if m else ("", text)

def postprocess_translation(translation_text):
    return re.sub(r'\s*\{\(.*?\)\}', '', translation_text)

def translate_line_openai(client, text, episode_synopsis, previous_context, original_language, target_language, model, extra_headers=None, temperature=1.3):
    formatting_codes, plain_text = extract_leading_formatting(text)
    parts = split_text_with_formatting(plain_text)
    text_to_translate = " ".join(part for is_text, part in parts if is_text).strip()
    system_message = (
        f"Translate the following subtitle text from {original_language} to {target_language}. "
        "Translate only the plain text and do not include any formatting codes. "
        "Do not add any extra commentary. Your output must contain only the translation of the provided text; "
        "do not include the context below in your output."
    )
    if episode_synopsis:
        system_message += f"\nEpisode Synopsis (for context only): {episode_synopsis}"
    if previous_context:
        system_message += f"\nPrevious Dialogue Context (for context only): {previous_context}"
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": text_to_translate}
    ]
    logger.debug(f"Request messages for line:\nSystem: {system_message}\nUser: {text_to_translate}")
    try:
        if not VERBOSE:
            with suppress_fd_output():
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False,
                    max_tokens=8192,
                    temperature=temperature,
                    extra_headers=extra_headers
                )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                max_tokens=8192,
                temperature=temperature,
                extra_headers=extra_headers
            )
        if response and hasattr(response, 'choices') and response.choices is not None and isinstance(response.choices, list) and len(response.choices) > 0:
            translation_text = response.choices[0].message.content.strip()
            logger.debug(f"Raw translation response: {translation_text}")
        else:
            logger.error(f"Invalid response for text: {text}. Response: {response}")
            return None
    except openai.RateLimitError as e:
        logger.error(f"Rate limit exceeded: {e}")
        return None
    except openai.AuthenticationError as e:
        logger.error(f"Authentication error: {e}")
        return None
    except openai.APIError as e:
        logger.error(f"API error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None
    if not translation_text:
        logger.error("Empty translation received")
        return None
    translation_text = postprocess_translation(translation_text)
    return f"{formatting_codes}{translation_text}"

def translate_line_claude(claude_key, text, episode_synopsis, previous_context, original_language, target_language, model, temperature=1.0):
    formatting_codes, plain_text = extract_leading_formatting(text)
    parts = split_text_with_formatting(plain_text)
    text_to_translate = " ".join(part for is_text, part in parts if is_text).strip()
    prompt = (
        f"Human: Translate the following subtitle text from {original_language} to {target_language}. "
        "Translate only the plain text without any formatting codes. Do not add any extra commentary. "
        "Do not include the context below in your answer."
    )
    if episode_synopsis:
        prompt += f" Episode Synopsis (for context only): {episode_synopsis}."
    if previous_context:
        prompt += f" Previous Dialogue Context (for context only): {previous_context}."
    prompt += f" Now, translate this: {text_to_translate}\n\nAssistant:"
    headers = {"Content-Type": "application/json", "x-api-key": claude_key}
    payload = {
        "prompt": prompt,
        "model": model,
        "max_tokens_to_sample": 300,
        "temperature": temperature
    }
    logger.debug(f"Claude request prompt:\n{prompt}")
    try:
        if not VERBOSE:
            with suppress_fd_output():
                response = requests.post("https://api.anthropic.com/complete", headers=headers, json=payload)
        else:
            response = requests.post("https://api.anthropic.com/complete", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        completion = data.get("completion", "").strip()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logger.error("Rate limit exceeded")
        elif e.response.status_code == 401:
            logger.error("Authentication error")
        else:
            logger.error(f"API error: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None
    if not completion:
        logger.error("Empty completion received from Claude")
        return None
    completion = postprocess_translation(completion)
    return f"{formatting_codes}{completion}"

def main():
    global VERBOSE
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    claude_key = os.getenv("CLAUDE_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    claude_model = os.getenv("CLAUDE_MODEL", "claude-v1")
    deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    openrouter_model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat:free")
    available_keys = {}
    if openai_key:
        available_keys["openai"] = openai_key
    if claude_key:
        available_keys["claude"] = claude_key
    if deepseek_key:
        available_keys["deepseek"] = deepseek_key
    if openrouter_key:
        available_keys["openrouter"] = openrouter_key
    if not available_keys:
        logger.error("No API keys found in .env file. Please set at least one of OPENAI_API_KEY, CLAUDE_API_KEY, DEEPSEEK_API_KEY, or OPENROUTER_API_KEY.")
        return
    parser = argparse.ArgumentParser(
        description="Translate subtitle files (.ass, .srt) line-by-line with context using an AI API. "
                    "Supports OpenAI, Claude, DeepSeek, and OpenRouter. For ASS files, the header is preserved."
    )
    parser.add_argument("input_files", nargs='+', help="Path to input subtitle files (.ass or .srt)")
    parser.add_argument("--target-language", required=True, help="Target language code (e.g., 'en')")
    parser.add_argument(
        "--ai",
        choices=["openai", "claude", "deepseek", "openrouter"],
        help="Which AI provider to use (openai, claude, deepseek, openrouter)"
    )
    parser.add_argument("--model", help="Specify the model to use, overrides default from .env")
    parser.add_argument("--temperature", type=float, default=1.3, help="Set the temperature for translation (0.0 to 2.0)")
    parser.add_argument("--context", help="Episode synopsis for the current film/episode (already translated)")
    parser.add_argument("--context-file", help="File with episode synopsis")
    parser.add_argument("--output-file", help="Path to save the translated file (only for single input file)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    VERBOSE = args.verbose
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        for lib in ("openai", "urllib3", "requests", "http.client", "urllib3.connectionpool", "requests.packages.urllib3.connectionpool"):
            logging.getLogger(lib).setLevel(logging.ERROR)
            logging.getLogger(lib).disabled = True
        try:
            http.client.HTTPConnection.debuglevel = 0
        except Exception:
            pass
    if len(available_keys) == 1:
        ai_choice = next(iter(available_keys))
        logger.info(f"Only {ai_choice} API key found. Using {ai_choice}.")
    else:
        if not args.ai:
            logger.error("Multiple API keys found. Please specify which AI to use with the --ai argument (openai, claude, deepseek, openrouter).")
            return
        ai_choice = args.ai
        if ai_choice not in available_keys:
            logger.error(f"API key for {ai_choice} is not available. Available keys: {', '.join(available_keys.keys())}")
            return
    for input_file in args.input_files:
        ext = os.path.splitext(input_file)[1].lower()
        if ext not in [".ass", ".srt"]:
            logger.error(f"Unsupported file format: {input_file}")
            continue
        try:
            subs = pysubs2.load(input_file)
        except Exception as e:
            logger.error(f"Failed to load subtitle file {input_file}: {e}")
            continue
        if len(args.input_files) == 1 and args.output_file:
            output_file = args.output_file
        else:
            output_file = os.path.splitext(input_file)[0] + f"_{args.target_language}{ext}"
        review_file = os.path.splitext(output_file)[0] + "_review.csv"
        all_text = " ".join(sub.text for sub in subs if sub.type == "Dialogue")
        original_language = detect_language(all_text)
        if not original_language:
            logger.error(f"Failed to detect original language for {input_file}")
            continue
        logger.info(f"Detected original language for {input_file}: {original_language}")
        if ai_choice == "openai":
            model = args.model if args.model else openai_model
        elif ai_choice == "claude":
            model = args.model if args.model else claude_model
        elif ai_choice == "deepseek":
            model = args.model if args.model else deepseek_model
        elif ai_choice == "openrouter":
            model = args.model if args.model else openrouter_model
        if ai_choice in ["openai", "deepseek", "openrouter"]:
            if ai_choice == "openai":
                client = OpenAI(api_key=available_keys["openai"], base_url="https://api.openai.com")
                extra_headers = None
            elif ai_choice == "deepseek":
                client = OpenAI(api_key=available_keys["deepseek"], base_url="https://api.deepseek.com")
                extra_headers = None
            elif ai_choice == "openrouter":
                client = OpenAI(api_key=available_keys["openrouter"], base_url="https://openrouter.ai/api/v1")
                extra_headers = {
                    "HTTP-Referer": "https://example.com",
                    "X-Title": "Subtitle Translator",
                }
            translate_func = lambda text, ep_syn, prev_ctx, orig_lang, tgt_lang: translate_line_openai(
                client, text, ep_syn, prev_ctx, orig_lang, tgt_lang, model, extra_headers, temperature=args.temperature
            )
        elif ai_choice == "claude":
            translate_func = lambda text, ep_syn, prev_ctx, orig_lang, tgt_lang: translate_line_claude(
                available_keys["claude"], text, ep_syn, prev_ctx, orig_lang, tgt_lang, model, temperature=args.temperature
            )
        episode_synopsis = args.context
        if args.context_file:
            try:
                with open(args.context_file, "r", encoding="utf-8") as f:
                    episode_synopsis = f.read()
            except Exception as e:
                logger.error(f"Failed to read context file: {e}")
                continue
        total_lines = sum(1 for sub in subs if sub.type == "Dialogue")
        pbar = tqdm(total=total_lines, desc=f"Translating {os.path.basename(input_file)}")
        previous_lines = []
        line_number = 0

        # Write review CSV header before starting
        try:
            with open(review_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Line Number", "Original Text", "Translated Text"])
        except Exception as e:
            logger.error(f"Failed to write review CSV header for {input_file}: {e}")
            continue

        for sub in subs:
            if sub.type == "Dialogue":
                line_number += 1
                original_text = sub.text
                if not remove_formatting(original_text):
                    logger.info(f"Line {line_number} is empty, skipping translation.")
                    pbar.update(1)
                    continue
                previous_context = "\n".join(line.text for line in previous_lines)
                logger.info(f"Translating line {line_number}: {remove_formatting(original_text)}")
                translation = translate_func(original_text, episode_synopsis, previous_context, original_language, args.target_language)
                if translation is None:
                    logger.error(f"Translation failed for line: {remove_formatting(original_text)}")
                    pbar.update(1)
                    continue
                logger.info(f"Translated line {line_number} to: {remove_formatting(translation)}")
                sub.text = translation
                # Save subtitle file immediately after successful translation
                try:
                    subs.save(output_file)
                except Exception as e:
                    logger.error(f"Failed to save subtitle file after line {line_number}: {e}")
                # Append to review CSV
                try:
                    with open(review_file, 'a', encoding='utf-8', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([line_number, remove_formatting(original_text), remove_formatting(translation)])
                except Exception as e:
                    logger.error(f"Failed to append to review CSV after line {line_number}: {e}")
                previous_lines.append(sub)
                if len(previous_lines) > 5:
                    previous_lines.pop(0)
                pbar.update(1)

if __name__ == "__main__":
    main()
