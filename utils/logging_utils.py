import rich

_log_styles = {
    "Viewer": "bold green",
    "ODGS-SLAM": "bold green",
    "Frontend": "bold light_green",
    "Backend": "bold dark_green",
    "GUI": "bold magenta",
    "Eval": "bold blue",
    "Info": "bold white",
    "Warn": "bold orange",
    "Error": "bold red",
}


def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"


def Log(*args, tag="ODGS-SLAM"):
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args)
