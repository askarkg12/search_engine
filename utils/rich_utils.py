from rich.console import Console

from contextlib import contextmanager

# Initialize the console for rich output
console = Console()


@contextmanager
def task(description):
    # Start a spinner with a description
    with console.status(f"[bold cyan]{description}...", spinner="dots") as status:
        try:
            yield  # Run the code inside the "with task(...)"
        finally:
            # Update the line when done
            console.log(f"[bold green]✔ {description} completed!")
