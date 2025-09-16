# test_chatbot.py
import argparse
import datetime as dt
import sys
from pathlib import Path
import chatbot


DEFAULT_QUESTIONS = [
    "hola",
    "dime info del guerrero tanque",
    "explícame el picaro machete",
    "qué poderes tiene el chamn?",
    "qué son los créditos",
    "cómo hago una subasta",
    "qué es la compra inmediata",
    "cuál es la capital de Francia?",
]


def run_batch(questions, save_txt=None, threshold=0.5):
    rt = chatbot.ChatRuntime()
    rows = []
    print("=" * 80)
    print(f"TEST CHATBOT  |  {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    for i, q in enumerate(questions, 1):
        try:
            a = rt.get_response(q, nn_threshold=threshold)
        except Exception as e:
            a = f"[ERROR] {e}"

        # Consola
        print(f"\n[{i:02d}] Q: {q}")
        print(f"     A: {a}")

        rows.append((q, a))

    # Guardar en TXT si se pidió
    if save_txt:
        p = Path(save_txt).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            f.write(f"TEST CHATBOT  |  {dt.datetime.now()}\n")
            f.write("=" * 60 + "\n")
            for i, (q, a) in enumerate(rows, 1):
                f.write(f"[{i:02d}] Q: {q}\n")
                f.write(f"     A: {a}\n\n")
        print(f"\n💾 Resultados guardados en: {p}")

    print("\n✅ Lote finalizado.")


def main():
    parser = argparse.ArgumentParser(
        description="Ejecuta un lote de preguntas contra ChatRuntime (chatbot.py)."
    )
    parser.add_argument(
        "--txt",
        help="Ruta de salida para guardar resultados en TXT (opcional).",
        default=None,
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Archivo de texto con una pregunta por línea (opcional).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Umbral de la NN (si no hay modelo, se usa semántica/fallback).",
    )
    parser.add_argument(
        "questions",
        nargs="*",
        help="Preguntas sueltas a evaluar (si no se pasan, se usan las por defecto).",
    )
    args = parser.parse_args()

    # Fuente de preguntas
    if args.questions:
        questions = args.questions
    elif args.file:
        p = Path(args.file)
        if not p.exists():
            print(f"[x] No existe el archivo: {p}", file=sys.stderr)
            sys.exit(1)
        questions = [
            line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    else:
        questions = DEFAULT_QUESTIONS

    run_batch(questions, save_txt=args.txt, threshold=args.threshold)


if __name__ == "__main__":
    main()
