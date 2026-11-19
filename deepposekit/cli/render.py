"""Figure publication shared by the review commands."""

from deepposekit.cli.operations import atomic_output
from deepposekit.utils.keypoints import graph_to_edges


def require_display():
    import matplotlib

    if str(matplotlib.get_backend()).lower() in {
        "agg",
        "cairo",
        "pdf",
        "pgf",
        "ps",
        "svg",
        "template",
    }:
        raise ValueError(
            "No interactive display is available. Use --output to save the result."
        )


def display_or_save(figure, output, overwrite):
    from matplotlib import pyplot as plt

    try:
        if output is None:
            plt.show()
        else:
            with atomic_output(output, overwrite) as temporary:
                figure.savefig(temporary)
            print(output)
    finally:
        plt.close(figure)


def skeleton_edges(graph):
    graph_to_edges(graph)
    return [(index, int(parent)) for index, parent in enumerate(graph) if parent >= 0]
