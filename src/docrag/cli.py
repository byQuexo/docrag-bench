"""Command line entry point: `uv run docrag <command>`.

docrag ingest    --config configs/baseline.yaml [--docs N]
docrag eval      configs/a.yaml configs/b.yaml ...
docrag summary
docrag calibrate --config configs/baseline.yaml
docrag ask       --config configs/baseline.yaml --doc <doc_id> "question"
docrag label-sample    results/<run> [--n 50]   (blind CSV for human labels)
docrag label           <labels.csv>             (y/n per answer, resumable)
docrag judge-agreement results/<run> <labels.csv>
docrag serve     --config configs/baseline.yaml [--port 8000]
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from docrag.shared.cache import Cache
from docrag.shared.config import ExperimentConfig
from docrag.shared.settings import get_settings
from docrag.shared.tracing import setup_tracing

log = logging.getLogger("docrag")


def _cmd_ingest(args: argparse.Namespace) -> None:
    from docrag.ingest.indexer import index_documents
    from docrag.wiring import build_ocr, build_retriever

    settings = get_settings()
    config = ExperimentConfig.from_yaml(args.config)
    pdfs = sorted(settings.pdf_dir.glob("*.pdf"))[: args.docs]
    retriever = build_retriever(settings, config, Cache(settings.cache_path))
    added = index_documents(
        {p.stem: p for p in pdfs},
        retriever.store,
        retriever.embeddings,
        prefix_headings=config.prefix_headings,
        cache=Cache(settings.cache_path),
        workers=args.workers,
        ocr=build_ocr(settings, config, Cache(settings.cache_path)),
    )
    print(
        f"added {added} chunks; "
        f"{retriever.store.collection} now has {retriever.store.count()} chunks"
    )


def _cmd_eval(args: argparse.Namespace) -> None:
    from docrag.eval.runner import run_experiment, summarize

    settings = get_settings()
    setup_tracing(settings.otel_endpoint, service_name="docrag-eval")
    cache = Cache(settings.cache_path)
    for path in args.configs:
        config = ExperimentConfig.from_yaml(path)
        run_dir = asyncio.run(run_experiment(config, settings, cache))
        print(f"{config.name}: {run_dir}")
    print(summarize(settings.results_dir))


def _cmd_summary(_: argparse.Namespace) -> None:
    from docrag.eval.runner import summarize

    print(summarize(get_settings().results_dir))


def _cmd_calibrate(args: argparse.Namespace) -> None:
    from docrag.eval.calibrate import calibrate

    settings = get_settings()
    threshold, f1 = calibrate(
        ExperimentConfig.from_yaml(args.config), settings, Cache(settings.cache_path)
    )
    print(f"rerank_threshold: {threshold:.4f}   # dev F1 = {f1:.3f}")


def _cmd_label_sample(args: argparse.Namespace) -> None:
    from docrag.eval.agreement import label_sample
    from docrag.eval.dataset import load_questions
    from docrag.store.qdrant import make_client

    settings = get_settings()
    out = args.out or settings.results_dir / "human_labels" / f"{args.run.name}.csv"
    if out.exists():  # never overwrite labels someone spent an hour on
        sys.exit(f"{out} exists; delete it or pass --out")
    questions = {q.qid: q for q in load_questions(settings.qa_glob)}
    client = make_client(settings.qdrant_location)
    sample = label_sample(args.run, questions, client, n=args.n, seed=args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.write_csv(out, include_bom=True)  # BOM: Excel then reads UTF-8 correctly
    print(
        f"wrote {sample.height} answers to {out}\n"
        f"Label it (one answer at a time, y/n): docrag label {out}\n"
        f"Then: docrag judge-agreement {args.run} {out}"
    )


def _cmd_label(args: argparse.Namespace) -> None:
    from docrag.eval.agreement import label_interactive

    n = label_interactive(args.labels)
    print(f"\nset {n} labels in {args.labels} (saved after each one)")


def _cmd_judge_agreement(args: argparse.Namespace) -> None:
    from docrag.eval.agreement import judge_agreement

    for field, r in judge_agreement(args.run, args.labels).items():
        kappa = "undefined (one label only)" if r["kappa"] != r["kappa"] else r["kappa"]
        print(f"{field}: {r['agree']}/{r['n']} agree, Cohen's kappa {kappa}")


def _cmd_ask(args: argparse.Namespace) -> None:
    from docrag.wiring import build_answer_service

    settings = get_settings()
    setup_tracing(settings.otel_endpoint, service_name="docrag-ask")
    service = build_answer_service(
        settings, ExperimentConfig.from_yaml(args.config), Cache(settings.cache_path)
    )

    async def run() -> None:
        try:
            result, hits = await service.answer(args.question, doc_ids=args.doc or None)
        finally:
            await service.proxy.aclose()
        print(f"\n{result.answer}\n\nstatus: {result.status}")
        for c in result.citations:
            mark = "OK " if c.verified else "REJ"
            print(
                f'  [{mark}] {c.id}: "{c.quote}"'
                + (f"  ({c.reason})" if c.reason else "")
            )
        print(
            "\nretrieved:",
            ", ".join(f"{h.chunk.id} ({h.score:.3f})" for h in hits[:10]),
        )

    asyncio.run(run())


def _cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from docrag.api.app import create_app
    from docrag.wiring import build_answer_service

    settings = get_settings()
    # _cmd_ingest deliberately never calls setup_tracing: no traces for the
    # offline batch job (its timings go to the logs / run.json instead).
    setup_tracing(settings.otel_endpoint, service_name="docrag-api")
    service = build_answer_service(
        settings, ExperimentConfig.from_yaml(args.config), Cache(settings.cache_path)
    )
    uvicorn.run(
        create_app(
            service,
            pdf_dir=settings.pdf_dir,
            results_dir=settings.results_dir,
            ui_dir=settings.ui_dir,
        ),
        host=args.host,
        port=args.port,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Third-party libraries are chatty at INFO.
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        prog="docrag",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="parse, chunk, embed and index PDFs")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--docs", type=int, default=None, help="only the first N PDFs")
    p.add_argument(
        "--workers", type=int, default=3, help="parallel PDF parser processes (CPU)"
    )
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser("eval", help="run one or more experiment configs")
    p.add_argument("configs", type=Path, nargs="+")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("summary", help="print the results table")
    p.set_defaults(func=_cmd_summary)

    p = sub.add_parser("calibrate", help="choose the rerank threshold on the dev split")
    p.add_argument("--config", type=Path, required=True)
    p.set_defaults(func=_cmd_calibrate)

    p = sub.add_parser("label-sample", help="blind CSV of judged answers to label")
    p.add_argument("run", type=Path, help="a results/<run> folder")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=_cmd_label_sample)

    p = sub.add_parser("label", help="label a sample in the terminal, one by one")
    p.add_argument("labels", type=Path, help="CSV written by label-sample")
    p.set_defaults(func=_cmd_label)

    p = sub.add_parser("judge-agreement", help="judge vs. human labels (kappa)")
    p.add_argument("run", type=Path)
    p.add_argument("labels", type=Path)
    p.set_defaults(func=_cmd_judge_agreement)

    p = sub.add_parser("ask", help="answer one question from the command line")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--doc", action="append", help="restrict to this doc_id (repeatable)"
    )
    p.add_argument("question")
    p.set_defaults(func=_cmd_ask)

    p = sub.add_parser("serve", help="run the FastAPI server")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)


# Required on Windows for multiprocessing: worker processes re-import this
# module, and without the guard each one would start the program again.
if __name__ == "__main__":
    main(sys.argv[1:])
