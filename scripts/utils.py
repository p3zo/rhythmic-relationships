import subprocess

import matplotlib.pyplot as plt


def save_fig(filepath, title=None):
    """Save a figure to a file and close it"""
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    print(f"Saved {filepath}")
    plt.close()


def play_midi_file(filepath):
    """Play a MIDI file using FluidSynth"""
    print("Playing MIDI...")
    subprocess.check_output(
        [
            "fluidsynth",
            "-i",  # deactivates the shell & causes FluidSynth to quit as soon as MIDI playback is completed
            "/usr/local/share/fluidsynth/generaluser.v.1.471.sf2",
            filepath,
        ]
    )
