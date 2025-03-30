import click
import json
import os
from rich.console import Console
from pathlib import Path
from transformers import AutoTokenizer
import anthropic
import openai
from typing import List, Dict
from datetime import datetime

console = Console()

def ensure_output_dir():
    Path('ablationmage_outputs').mkdir(exist_ok=True)
    console.print("[blue]Created or verified 'ablationmage' directory[/blue]")

def generate_output_filename():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'ablationmage_outputs/output_{timestamp}.txt'

def read_template(template_file: str) -> dict:
    with open(template_file, 'r') as f:
        return json.load(f)

def collect_files(paths: List[str]) -> List[Path]:
    """Collect all files from given paths and directories with specific extensions."""
    ALLOWED_EXTENSIONS = {'.py', '.md', '.txt', '.yaml', '.yml', '.rst'}
    all_files = []
    
    for path in paths:
        path_obj = Path(path)
        if path_obj.is_file():
            if path_obj.suffix.lower() in ALLOWED_EXTENSIONS:
                all_files.append(path_obj)
        elif path_obj.is_dir():
            # Recursively collect all files from directory with allowed extensions
            all_files.extend(
                file for file in path_obj.rglob('*') 
                if file.is_file() and file.suffix.lower() in ALLOWED_EXTENSIONS
            )
    
    # Log the number of files found with each extension
    extension_counts = {}
    for file in all_files:
        ext = file.suffix.lower()
        extension_counts[ext] = extension_counts.get(ext, 0) + 1
    
    for ext, count in extension_counts.items():
        console.print(f"[blue]Found {count} {ext} file(s)[/blue]")
    
    return all_files

def call_api(api: str, formatted_prompt: str, model: str):
    console.print(f"[yellow]Calling {api.upper()} API with model: {model}[/yellow]")
    
    if api == "anthropic":
        console.print("[yellow]Sending request to Claude...[/yellow]")
        client = anthropic.Client(api_key=os.getenv('ANTHROPIC_API_KEY'))
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": formatted_prompt}
            ]
        )
        console.print("[green]Received response from Claude[/green]")
        return response.content[0].text
    
    elif api == "openai":
        console.print("[yellow]Sending request to OpenAI...[/yellow]")
        client = openai.Client(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": formatted_prompt}]
        )
        console.print("[green]Received response from OpenAI[/green]")
        return response.choices[0].message.content
    
    raise ValueError(f"Unsupported API: {api}")

def process_first_call(chat_template, template_file, api, model, docs):
    try:
        ensure_output_dir()
        
        console.print(f"[blue]Loading template from {template_file}[/blue]")
        conversation = read_template(template_file)
        
        if docs:
            console.print("[blue]Processing files and directories...[/blue]")
            all_files = collect_files(docs)
            doc_messages = [
                {
                    "role": "user", 
                    "content": f"Document '{file.relative_to(file.parent)}':\n```\n{file.read_text()}\n```"
                } 
                for file in all_files
            ]
            sys_msg_idx = next((i for i, msg in enumerate(conversation['messages']) if msg['role'] == 'system'), 0)
            conversation['messages'][sys_msg_idx+1:sys_msg_idx+1] = doc_messages
            console.print(f"[green]Added {len(all_files)} file(s) to conversation[/green]")
        
        console.print(f"[blue]Loading chat template from {chat_template}[/blue]")
        tokenizer = AutoTokenizer.from_pretrained(chat_template)
        console.print("[blue]Applying chat template...[/blue]")
        formatted_prompt = tokenizer.apply_chat_template(conversation['messages'], tokenize=False, add_generation_prompt=True)
        console.print("[green]Chat template applied successfully[/green]")
        
        result = call_api(api, formatted_prompt, model)
        output_file = generate_output_filename()
        
        console.print(f"[blue]Saving response to {output_file}[/blue]")
        with open(output_file, 'w') as f:
            f.write(result)
        console.print(f"[green]Output saved successfully to {output_file}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

def process_followup_call(chat_template, template_file, api, model, docs, output_result):
    try:
        ensure_output_dir()
        
        console.print(f"[blue]Loading template from {template_file}[/blue]")
        conversation = read_template(template_file)
        
        # Add documents as first user message
        if docs:
            console.print("[blue]Processing files and directories...[/blue]")
            all_files = collect_files(docs)
            docs_content = "\n\n".join([
                f"Document '{file.relative_to(file.parent)}':\n```\n{file.read_text()}\n```" 
                for file in all_files
            ])
            conversation['messages'].append({
                "role": "user",
                "content": docs_content
            })
            console.print(f"[green]Added {len(all_files)} file(s) to conversation[/green]")
        
        # Add output result as second user message
        conversation['messages'].append({
            "role": "user",
            "content": f"Output of running the code:\n```\n{output_result}\n```"
        })
        console.print("[green]Added output result to conversation[/green]")
        
        console.print(f"[blue]Loading chat template from {chat_template}[/blue]")
        tokenizer = AutoTokenizer.from_pretrained(chat_template)
        console.print("[blue]Applying chat template...[/blue]")
        formatted_prompt = tokenizer.apply_chat_template(conversation['messages'], tokenize=False, add_generation_prompt=True)
        console.print("[green]Chat template applied successfully[/green]")
        
        result = call_api(api, formatted_prompt, model)
        output_file = generate_output_filename()
        
        console.print(f"[blue]Saving response to {output_file}[/blue]")
        with open(output_file, 'w') as f:
            f.write(result)
        console.print(f"[green]Output saved successfully to {output_file}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

@click.group()
def cli():
    """CLI tool for chat templates with first and followup calls."""
    pass

@cli.command()
@click.argument('chat_template')
@click.option('--api', '-a', type=click.Choice(['anthropic', 'openai']), required=True)
@click.option('--model', '-m', required=True)
@click.option('--docs', '-d', multiple=True, type=click.Path(exists=True, dir_okay=True, file_okay=True))
def first_call(chat_template, api, model, docs):
    """Make initial call using first_call_template.json"""
    console.print("\n[bold blue]Starting first call process...[/bold blue]")
    process_first_call(chat_template, "first_call_template.json", api, model, docs)
    console.print("\n[bold green]Process completed![/bold green]")

@cli.command()
@click.argument('chat_template')
@click.argument('output_result', type=click.Path(exists=True))
@click.option('--api', '-a', type=click.Choice(['anthropic', 'openai']), required=True)
@click.option('--model', '-m', required=True)
@click.option('--docs', '-d', multiple=True, type=click.Path(exists=True, dir_okay=True, file_okay=True))
def followup_call(chat_template, output_result, api, model, docs):
    """Make followup call using followup_call_template.json"""
    console.print("\n[bold blue]Starting followup call process...[/bold blue]")
    with open(output_result, 'r') as f:
        result_content = f.read()
    process_followup_call(chat_template, "followup_call_template.json", api, model, docs, result_content)
    console.print("\n[bold green]Process completed![/bold green]")

if __name__ == '__main__':
    cli()