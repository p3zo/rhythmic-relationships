import matplotlib.pyplot as plt


def save_fig(filepath, title=None):
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    print(f"Saved {filepath}")
    plt.close()
