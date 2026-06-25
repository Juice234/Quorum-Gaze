from __future__ import annotations

import queue
import threading
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, Button, Entry, Frame, Label, Listbox, StringVar, Text, Tk, filedialog
from tkinter import messagebox

from PIL import Image, ImageTk

from quorum.detector import detect_people, draw_people, save_detection_json
from quorum.scraper import scrape_botb_result


APP_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = APP_DIR / "data"
OUTPUT_DIR = APP_DIR / "outputs"


def log(message_box: Text, message: str) -> None:
    # The text box is used like a simple console inside the GUI.
    message_box.insert(END, message + "\n")
    message_box.see(END)
    message_box.update_idletasks()


def run_in_thread(work, done_queue: queue.Queue) -> None:
    # Scraping and YOLO can take a while, so this keeps the window responsive.
    def wrapper() -> None:
        try:
            done_queue.put(("ok", work()))
        except Exception as exc:
            done_queue.put(("error", str(exc)))

    threading.Thread(target=wrapper, daemon=True).start()


def poll_queue(root: Tk, done_queue: queue.Queue, on_ok, on_error) -> None:
    # Tkinter should be updated from the main thread, so the worker gives results back through a queue.
    try:
        status, payload = done_queue.get_nowait()
    except queue.Empty:
        root.after(100, lambda: poll_queue(root, done_queue, on_ok, on_error))
        return

    if status == "ok":
        on_ok(payload)
    else:
        on_error(payload)


def show_image(label: Label, image_path: Path) -> None:
    # Resize the image for preview only. The original file is not changed.
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((760, 520))
    preview = ImageTk.PhotoImage(image)
    label.configure(image=preview)
    label.image = preview


def add_images_to_list(image_list: Listbox, scrape_result: dict) -> None:
    # Only scraped files with a real saved path are shown to the user.
    image_list.delete(0, END)
    for item in scrape_result.get("images", []):
        if "path" in item:
            image_list.insert(END, item["path"])


def selected_image_from_list(image_list: Listbox) -> Path | None:
    selection = image_list.curselection()
    if not selection:
        return None
    return Path(image_list.get(selection[0]))


def build_gui() -> Tk:
    # The GUI is built in one function on purpose so the project stays easy to follow.
    root = Tk()
    root.title("Quorum - BOTB Scraper + YOLO People Detection")
    root.geometry("1180x760")

    state = {"selected_image": None}

    left_panel = Frame(root, padx=10, pady=10)
    left_panel.pack(side=LEFT, fill="y")

    right_panel = Frame(root, padx=10, pady=10)
    right_panel.pack(side=RIGHT, fill=BOTH, expand=True)

    Label(left_panel, text="BOTB scrape").pack(anchor="w")

    competition_var = StringVar(value="dreamcar")
    year_var = StringVar(value="26")
    week_var = StringVar(value="1")
    model_var = StringVar(value="yolov8n.pt")

    Label(left_panel, text="Competition").pack(anchor="w")
    Entry(left_panel, textvariable=competition_var, width=24).pack(anchor="w")
    Label(left_panel, text="Year, e.g. 26").pack(anchor="w")
    Entry(left_panel, textvariable=year_var, width=24).pack(anchor="w")
    Label(left_panel, text="Week, e.g. 1").pack(anchor="w")
    Entry(left_panel, textvariable=week_var, width=24).pack(anchor="w")

    image_list = Listbox(left_panel, width=55, height=12)
    image_list.pack(pady=8)

    Label(left_panel, text="YOLO model").pack(anchor="w")
    Entry(left_panel, textvariable=model_var, width=24).pack(anchor="w")

    message_box = Text(left_panel, width=55, height=16)
    message_box.pack(pady=8)

    preview_label = Label(right_panel, text="No image selected")
    preview_label.pack(fill=BOTH, expand=True)

    def set_selected_image(image_path: Path) -> None:
        state["selected_image"] = image_path
        show_image(preview_label, image_path)
        log(message_box, f"Selected image: {image_path}")

    def scrape_clicked() -> None:
        # Read the small form, scrape one BOTB page, then add the saved images to the list.
        try:
            year = int(year_var.get())
            week = int(week_var.get())
        except ValueError:
            messagebox.showerror("Input error", "Year and week must be numbers.")
            return

        competition = competition_var.get().strip() or "dreamcar"
        done_queue: queue.Queue = queue.Queue()
        log(message_box, f"Scraping {competition} year {year}, week {week}...")

        def work() -> dict:
            return scrape_botb_result(year, week, competition, DATA_DIR)

        def on_ok(result: dict) -> None:
            add_images_to_list(image_list, result)
            log(message_box, f"Scrape finished. Saved {result['image_count']} images.")
            if result["judge_x"] is not None and result["judge_y"] is not None:
                log(message_box, f"Judge coordinate found: X={result['judge_x']} Y={result['judge_y']}")
            if image_list.size() > 0:
                image_list.selection_set(0)
                set_selected_image(Path(image_list.get(0)))

        def on_error(error: str) -> None:
            log(message_box, f"Scrape failed: {error}")
            messagebox.showerror("Scrape failed", error)

        run_in_thread(work, done_queue)
        poll_queue(root, done_queue, on_ok, on_error)

    def browse_clicked() -> None:
        # This is useful when testing YOLO without scraping first.
        path = filedialog.askopenfilename(
            title="Choose image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp"), ("All files", "*.*")],
        )
        if path:
            set_selected_image(Path(path))

    def list_selected(_event=None) -> None:
        image_path = selected_image_from_list(image_list)
        if image_path:
            set_selected_image(image_path)

    def detect_clicked() -> None:
        # Run YOLO on the selected image and show the annotated output.
        image_path = state.get("selected_image")
        if not image_path:
            messagebox.showerror("No image", "Choose or scrape an image first.")
            return

        model_name = model_var.get().strip() or "yolov8n.pt"
        done_queue: queue.Queue = queue.Queue()
        log(message_box, f"Running YOLO people detection: {model_name}")

        def work() -> dict:
            image = Path(image_path)
            people = detect_people(image, model_name=model_name)
            stem = image.stem
            out_image = OUTPUT_DIR / f"{stem}_people.jpg"
            out_json = OUTPUT_DIR / f"{stem}_people.json"
            draw_people(image, people, out_image)
            save_detection_json(image, people, out_json, model_name)
            return {"people": people, "out_image": out_image, "out_json": out_json}

        def on_ok(result: dict) -> None:
            log(message_box, f"Detected {len(result['people'])} people.")
            log(message_box, f"Annotated image: {result['out_image']}")
            log(message_box, f"Detection JSON: {result['out_json']}")
            set_selected_image(result["out_image"])

        def on_error(error: str) -> None:
            log(message_box, f"Detection failed: {error}")
            messagebox.showerror("Detection failed", error)

        run_in_thread(work, done_queue)
        poll_queue(root, done_queue, on_ok, on_error)

    Button(left_panel, text="Scrape BOTB page", command=scrape_clicked, width=24).pack(anchor="w", pady=4)
    Button(left_panel, text="Choose local image", command=browse_clicked, width=24).pack(anchor="w", pady=4)
    Button(left_panel, text="Run YOLO people detection", command=detect_clicked, width=24).pack(anchor="w", pady=4)

    image_list.bind("<<ListboxSelect>>", list_selected)
    log(message_box, "Ready. Scrape a BOTB page or choose an image.")
    return root


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    root = build_gui()
    root.mainloop()
