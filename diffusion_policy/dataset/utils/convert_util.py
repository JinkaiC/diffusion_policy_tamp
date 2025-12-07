import os
from termcolor import cprint

def handle_existing_data(output_dir: str, force_overwrite: bool = True) -> None:
    """Handle existing data at the output directory."""
    if os.path.exists(output_dir):
        print("")
        cprint(f"Data already exists at {output_dir}", "yellow")

        if force_overwrite:
            cprint(f"Overwriting existing data at {output_dir}", "red")
            os.system(f"rm -rf {output_dir}")
            print("")
        else:
            cprint(
                "If you want to overwrite, delete the existing directory first.",
                "yellow",
            )
            user_input = input("Overwrite? (y/n): ")
            if user_input.lower() == "y":
                cprint(f"Overwriting existing data at {output_dir}", "red")
                os.system(f"rm -rf {output_dir}")
                print("")
            else:
                cprint("Exiting", "yellow")
                exit()

    os.makedirs(output_dir, exist_ok=True)
    cprint(f"Creating new data at {output_dir}", "green")