from rich.console import Console
from rich.panel import Panel

console = Console()

def show_summary(success: bool, title: str, details: dict):
    """Show a summary of the operation"""
    panel_style = "green" if success else "red"
    status = "[green]SUCCESS[/green]" if success else "[red]FAILED[/red]"
    
    content = [
        title,
        f"Status: {status}",
        "",
        "Details:",
    ]
    
    for key, value in details.items():
        content.append(f"  {key}: {value}")
    
    panel_content = "\n".join(content)
    
    console.print("\n")
    console.print(Panel(
        panel_content,
        style=panel_style,
        title="AlphaMask Operation Summary"
    ))