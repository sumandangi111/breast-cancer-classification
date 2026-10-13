from pathlib import Path
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_report(file_path: str) -> dict:
    """Load a text or PDF report and return its extracted text."""

    path = Path(file_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")

    if not path.is_file():
        raise ValueError("The provided path is not a file.")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported file type. Please provide a PDF or TXT report."
        )

    try:
        if path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8-sig")

        else:
            reader = PdfReader(str(path))

            if reader.is_encrypted:
                raise ValueError("The PDF is encrypted or password-protected.")

            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")

            text = "\n".join(pages)

    except UnicodeDecodeError as exc:
        raise ValueError(
            "The TXT file is not valid UTF-8 text."
        ) from exc
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Could not read the report: {exc}") from exc

    if not text.strip():
        raise ValueError(
            "No readable text found. The PDF may be scanned and require OCR."
        )

    return {
        "file_name": path.name,
        "file_type": path.suffix.lower(),
        "text": text.strip(),
        "character_count": len(text.strip()),
    }
