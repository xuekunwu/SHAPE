"""CLI for preparation, public table replay, figures and integrity checking."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Curate a private native export into public aggregate evidence")
    p.add_argument("--export-root", required=True)
    p.add_argument("--codex-trajectory", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("tables", help="Replay primary metrics, split/donor tables and F3 from public evidence")
    p.add_argument("--result", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("figures", help="Plot aggregate quality, prediction, donor boxplots and descriptive costs")
    p.add_argument("--result", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("umap", help="Plot all own-arm cells by split and cTnT, using hash-verified private coordinates")
    p.add_argument("--result", required=True)
    p.add_argument("--coordinates", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("manifest", help="Write or verify all published result file hashes")
    p.add_argument("--result", required=True)
    p.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        from prepare import prepare
        prepare(args.export_root, args.codex_trajectory, args.out)
    elif args.command == "tables":
        from report import build
        build(args.result, args.out)
    elif args.command == "figures":
        from figures import summary
        summary(args.result, args.out)
    elif args.command == "umap":
        from figures import umap
        umap(args.result, args.coordinates, args.out)
    else:
        from report import manifest
        manifest(args.result, args.verify)


if __name__ == "__main__":
    main()
