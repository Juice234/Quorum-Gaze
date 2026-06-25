from __future__ import annotations

import asyncio
import queue
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import (
    BOTH,
    END,
    LEFT,
    RIGHT,
    X,
    Y,
    BooleanVar,
    Button,
    Checkbutton,
    Entry,
    Frame,
    Label,
    LabelFrame,
    Radiobutton,
    Scrollbar,
    StringVar,
    Text,
    Tk,
    ttk,
)
from tkinter import messagebox

from PIL import Image, ImageDraw, ImageTk

from quorum_scraper.database import ResultsDatabase
from quorum_scraper.scraper import BotbScraper


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"

BG = "#070d1a"
PANEL = "#101a2d"
PANEL_2 = "#0b1324"
HEADER = "#16233d"
TEXT = "#c8d6f0"
MUTED = "#8796b6"
BUTTON = "#1d2d4a"
BLACK_PREVIEW = "#141719"


def setup_style(root: Tk) -> None:
    # colours copied from the BOTB Studio screen so the first version looks familiar
    root.configure(bg=BG)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Treeview", background=PANEL_2, foreground=TEXT, fieldbackground=PANEL_2, rowheight=28)
    style.configure("Treeview.Heading", background=HEADER, foreground=TEXT, relief="flat")
    style.map("Treeview", background=[("selected", "#24385f")])
    style.configure("TCombobox", fieldbackground=PANEL_2, background=PANEL_2, foreground=TEXT, arrowcolor=TEXT)


def make_panel(parent, title: str | None = None) -> LabelFrame:
    # boxed sections make the page easier to scan
    panel = LabelFrame(parent, text=title or "", bg=PANEL, fg=TEXT, bd=1, relief="solid", padx=12, pady=10)
    panel.configure(labelanchor="nw")
    return panel


def make_label(parent, text: str) -> Label:
    return Label(parent, text=text, bg=PANEL, fg=TEXT, anchor="w")


def make_entry(parent, value: str = "", width: int = 12) -> Entry:
    entry = Entry(parent, width=width, bg=PANEL_2, fg=TEXT, insertbackground=TEXT, relief="solid", bd=1)
    entry.insert(0, value)
    return entry


def make_button(parent, text: str, command, width: int | None = None) -> Button:
    button = Button(
        parent,
        text=text,
        command=command,
        width=width or 18,
        bg=BUTTON,
        fg=TEXT,
        activebackground="#294066",
        activeforeground="white",
        relief="solid",
        bd=1,
        pady=5,
    )
    return button


def make_combo(parent, values: list[str], width: int = 18) -> ttk.Combobox:
    combo = ttk.Combobox(parent, values=values, width=width, state="readonly")
    if values:
        combo.set(values[0])
    return combo


def log(log_box: Text, message: str) -> None:
    # small log area so I can see what happened without opening a terminal
    log_box.insert(END, message + "\n")
    log_box.see(END)
    log_box.update_idletasks()


def run_in_thread(job, result_queue: queue.Queue) -> None:
    # scraper runs in the background because the browser part can take a while
    def wrapper() -> None:
        try:
            result_queue.put(("ok", job()))
        except Exception as exc:
            result_queue.put(("error", str(exc)))

    threading.Thread(target=wrapper, daemon=True).start()


def poll_thread(root: Tk, result_queue: queue.Queue, on_ok, on_error, on_progress=None) -> None:
    try:
        status, payload = result_queue.get_nowait()
    except queue.Empty:
        root.after(100, lambda: poll_thread(root, result_queue, on_ok, on_error, on_progress))
        return

    if status == "progress":
        if on_progress:
            on_progress(payload)
        root.after(100, lambda: poll_thread(root, result_queue, on_ok, on_error, on_progress))
    elif status == "ok":
        on_ok(payload)
    else:
        on_error(payload)


def load_rows() -> list[dict]:
    db_path = DATA_DIR / "botb.sqlite"
    if not db_path.exists():
        return []
    with ResultsDatabase(db_path) as db:
        return db.rows()


def cell(value) -> str:
    return "" if value is None else str(value)


def row_image_path(row: dict | None, preview_mode: str) -> str | None:
    if not row:
        return None
    if preview_mode == "judged":
        return row.get("judged_image_path")
    return row.get("clean_image_path")


def image_with_cross(path: str, x: int | None, y: int | None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if x is None or y is None:
        return image
    draw = ImageDraw.Draw(image)
    radius = max(14, int(min(image.size) * 0.018))
    draw.line((x - radius, y, x + radius, y), fill="red", width=5)
    draw.line((x, y - radius, x, y + radius), fill="red", width=5)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="red", width=4)
    return image


def show_preview(preview_label: Label, row: dict | None, preview_mode: str) -> None:
    path = row_image_path(row, preview_mode)
    if not path or not Path(path).exists():
        preview_label.configure(image="", text="Select a result row")
        preview_label.image = None
        return

    if preview_mode == "cross":
        image = image_with_cross(path, row.get("judge_x"), row.get("judge_y"))
    else:
        image = Image.open(path).convert("RGB")

    image.thumbnail((1420, 320))
    tk_image = ImageTk.PhotoImage(image)
    preview_label.configure(image=tk_image, text="")
    preview_label.image = tk_image


def build_gui() -> Tk:
    root = Tk()
    root.title("BOTB Studio")
    root.geometry("1500x820")
    setup_style(root)

    rows: list[dict] = []
    rows_by_id: dict[str, dict] = {}
    selected_row_id = StringVar(value="")
    preview_mode = StringVar(value="clean")

    # first version is only the image scraper so there is no navbar yet
    header = Frame(root, bg=BG, height=44)
    header.pack(fill=X)
    Label(header, text="Image Scraper", bg=BG, fg="white", font=("Segoe UI", 14, "bold"), padx=16, pady=10).pack(
        side=LEFT
    )

    main = Frame(root, bg=BG)
    main.pack(fill=BOTH, expand=True, padx=8, pady=8)

    left = Frame(main, bg=BG, width=420)
    left.pack(side=LEFT, fill=Y)
    left.pack_propagate(False)

    right = Frame(main, bg=BG)
    right.pack(side=RIGHT, fill=BOTH, expand=True, padx=(8, 0))

    filters = make_panel(left, "Filters")
    filters.pack(fill=X)

    search_var = StringVar(value="")
    make_label(filters, "Search").grid(row=0, column=0, sticky="ew", pady=4)
    search_entry = make_entry(filters, width=34)
    search_entry.grid(row=0, column=1, columnspan=3, sticky="ew", pady=4)

    make_label(filters, "Competition type").grid(row=1, column=0, sticky="ew", pady=4)
    competition_filter = make_combo(filters, ["All competitions"], width=18)
    competition_filter.grid(row=1, column=1, sticky="ew", pady=4)

    make_label(filters, "Status").grid(row=1, column=2, sticky="ew", pady=4, padx=(8, 0))
    status_filter = make_combo(filters, ["All statuses"], width=18)
    status_filter.grid(row=1, column=3, sticky="ew", pady=4)

    make_label(filters, "Coordinate").grid(row=2, column=0, sticky="ew", pady=4)
    coordinate_filter = make_combo(filters, ["All coordinates", "Valid only", "Invalid only", "Missing only"], width=18)
    coordinate_filter.grid(row=2, column=1, sticky="ew", pady=4)

    make_label(filters, "Year").grid(row=2, column=2, sticky="ew", pady=4, padx=(8, 0))
    year_filter = make_combo(filters, ["All years"], width=18)
    year_filter.grid(row=2, column=3, sticky="ew", pady=4)

    make_label(filters, "Week").grid(row=3, column=0, sticky="ew", pady=4)
    week_filter = make_combo(filters, ["All weeks"], width=18)
    week_filter.grid(row=3, column=1, sticky="ew", pady=4)
    filters.grid_columnconfigure(1, weight=1)
    filters.grid_columnconfigure(3, weight=1)

    table_box = Frame(left, bg=PANEL_2, bd=1, relief="solid")
    table_box.pack(fill=BOTH, expand=True, pady=(8, 0))

    columns = ("competition_type", "year", "week", "status", "judge_x", "judge_y", "coordinate_invalid")
    table_scroll_y = Scrollbar(table_box)
    table_scroll_y.pack(side=RIGHT, fill=Y)
    table_scroll_x = Scrollbar(table_box, orient="horizontal")
    table_scroll_x.pack(side="bottom", fill=X)
    table = ttk.Treeview(
        table_box,
        columns=columns,
        show="headings",
        yscrollcommand=table_scroll_y.set,
        xscrollcommand=table_scroll_x.set,
    )
    table_scroll_y.config(command=table.yview)
    table_scroll_x.config(command=table.xview)
    for column in columns:
        table.heading(column, text=column)
        table.column(column, width=108, stretch=False)
    table.column("year", width=58, anchor="center")
    table.column("week", width=58, anchor="center")
    table.column("status", width=80)
    table.column("judge_x", width=72, anchor="center")
    table.column("judge_y", width=72, anchor="center")
    table.column("coordinate_invalid", width=126, anchor="center")
    table.pack(side=LEFT, fill=BOTH, expand=True)

    preview_panel = Frame(right, bg=BLACK_PREVIEW, bd=1, relief="solid", height=320)
    preview_panel.pack(fill=X)
    preview_panel.pack_propagate(False)
    preview_label = Label(preview_panel, text="Select a result row", bg=BLACK_PREVIEW, fg=MUTED)
    preview_label.pack(fill=BOTH, expand=True)

    selected_panel = make_panel(right, "Selected Result")
    selected_panel.pack(fill=X, pady=(8, 0))
    detail_url = Entry(selected_panel, bg=PANEL_2, fg=TEXT, relief="solid", bd=0)
    detail_url.grid(row=0, column=1, sticky="ew", padx=4, pady=3)
    detail_image = Entry(selected_panel, bg=PANEL_2, fg=TEXT, relief="solid", bd=0)
    detail_image.grid(row=1, column=1, sticky="ew", padx=4, pady=3)
    detail_xy = Entry(selected_panel, bg=PANEL_2, fg=TEXT, relief="solid", bd=0)
    detail_xy.grid(row=1, column=3, sticky="ew", padx=4, pady=3)
    detail_debug = Entry(selected_panel, bg=PANEL_2, fg=TEXT, relief="solid", bd=0)
    detail_debug.grid(row=2, column=1, columnspan=3, sticky="ew", padx=4, pady=3)
    for label_text, row_num, col_num in (("URL", 0, 0), ("Image:", 1, 0), ("X: -   Y: -", 1, 2), ("Debug:", 2, 0)):
        make_label(selected_panel, label_text).grid(row=row_num, column=col_num, sticky="w", padx=4)
    selected_panel.grid_columnconfigure(1, weight=1)
    selected_panel.grid_columnconfigure(3, weight=1)

    preview_actions = make_panel(right, "Preview")
    preview_actions.pack(fill=X, pady=(8, 0))
    for idx, (label, value) in enumerate(
        [
            ("Show clean image", "clean"),
            ("Show BOTB judges circle", "judged"),
            ("Show clean image with generated cross", "cross"),
        ]
    ):
        radio = Radiobutton(
            preview_actions,
            text=label,
            variable=preview_mode,
            value=value,
            bg=PANEL,
            fg=TEXT,
            selectcolor=PANEL_2,
            activebackground=PANEL,
            activeforeground=TEXT,
            command=lambda: show_preview(preview_label, rows_by_id.get(selected_row_id.get()), preview_mode.get()),
        )
        radio.grid(row=0, column=idx, sticky="w", padx=8)

    scrape_panel = make_panel(right, "Scrape")
    scrape_panel.pack(fill=BOTH, expand=True, pady=(8, 0))

    banner = Label(
        scrape_panel,
        text="QUORUM-GAZE SCRAPER\nBOTB result pages -> clean image, judged image, coordinates, metadata",
        bg=PANEL_2,
        fg=MUTED,
        justify="left",
        anchor="w",
        padx=10,
        pady=8,
    )
    banner.pack(fill=X)

    scrape_form = Frame(scrape_panel, bg=PANEL)
    scrape_form.pack(fill=X, pady=(8, 0))

    competition_var = StringVar(value="dreamcar")
    start_year_var = StringVar(value="26")
    end_year_var = StringVar(value="26")
    start_week_var = StringVar(value="1")
    end_week_var = StringVar(value="1")
    delay_var = StringVar(value="1.0")
    overwrite_var = BooleanVar(value=False)
    headed_var = BooleanVar(value=False)

    form_fields = [
        ("Competition", competition_var, 0),
        ("Start year", start_year_var, 1),
        ("End year", end_year_var, 2),
        ("Start week", start_week_var, 3),
        ("End week", end_week_var, 4),
        ("Delay", delay_var, 5),
    ]
    for label, var, col in form_fields:
        make_label(scrape_form, label).grid(row=0, column=col, sticky="w", padx=4)
        Entry(scrape_form, textvariable=var, bg=PANEL_2, fg=TEXT, insertbackground=TEXT, width=13).grid(
            row=1, column=col, sticky="ew", padx=4
        )

    Checkbutton(scrape_form, text="Overwrite", variable=overwrite_var, bg=PANEL, fg=TEXT, selectcolor=PANEL_2).grid(
        row=1, column=6, sticky="w", padx=8
    )
    Checkbutton(scrape_form, text="Show browser", variable=headed_var, bg=PANEL, fg=TEXT, selectcolor=PANEL_2).grid(
        row=1, column=7, sticky="w", padx=8
    )

    button_grid = Frame(scrape_panel, bg=PANEL)
    button_grid.pack(fill=X, pady=(8, 0))

    log_box = Text(scrape_panel, height=6, bg=PANEL_2, fg=TEXT, insertbackground=TEXT, relief="solid", bd=1)
    log_box.pack(fill=X, pady=(8, 0))

    def current_row() -> dict | None:
        return rows_by_id.get(selected_row_id.get())

    def update_detail_entries(row: dict | None) -> None:
        for entry in (detail_url, detail_image, detail_xy, detail_debug):
            entry.delete(0, END)

        if not row:
            detail_url.insert(0, "No row selected")
            detail_image.insert(0, "-")
            detail_xy.insert(0, "X: -   Y: -")
            detail_debug.insert(0, "-")
            return

        detail_url.insert(0, cell(row.get("url")))
        detail_image.insert(0, cell(row.get("clean_image_path")))
        detail_xy.insert(0, f"X: {cell(row.get('judge_x'))}   Y: {cell(row.get('judge_y'))}")
        detail_debug.insert(0, cell(row.get("debug_dir")))

    def filter_rows() -> list[dict]:
        text = search_entry.get().strip().lower()
        competition_value = competition_filter.get()
        status_value = status_filter.get()
        coordinate_value = coordinate_filter.get()
        year_value = year_filter.get()
        week_value = week_filter.get()

        filtered = []
        for row in rows:
            search_text = " ".join(
                cell(row.get(key)).lower()
                for key in ("url", "competition", "competition_type", "year", "week", "status")
            )
            if text and text not in search_text:
                continue
            if competition_value != "All competitions" and cell(row.get("competition_type")) != competition_value:
                continue
            if status_value != "All statuses" and cell(row.get("status")) != status_value:
                continue
            if year_value != "All years" and cell(row.get("year")) != year_value:
                continue
            if week_value != "All weeks" and cell(row.get("week")).zfill(2) != week_value:
                continue
            if coordinate_value == "Valid only" and row.get("coordinate_invalid") not in (0, False):
                continue
            if coordinate_value == "Invalid only" and row.get("coordinate_invalid") not in (1, True):
                continue
            if coordinate_value == "Missing only" and (row.get("judge_x") is not None and row.get("judge_y") is not None):
                continue
            filtered.append(row)
        return filtered

    def update_filter_choices() -> None:
        competitions = sorted({cell(row.get("competition_type")) for row in rows if row.get("competition_type")})
        statuses = sorted({cell(row.get("status")) for row in rows if row.get("status")})
        years = sorted({cell(row.get("year")) for row in rows if row.get("year") is not None})
        weeks = sorted({cell(row.get("week")).zfill(2) for row in rows if row.get("week") is not None})
        competition_filter.configure(values=["All competitions", *competitions])
        status_filter.configure(values=["All statuses", *statuses])
        year_filter.configure(values=["All years", *years])
        week_filter.configure(values=["All weeks", *weeks])
        for combo, default in (
            (competition_filter, "All competitions"),
            (status_filter, "All statuses"),
            (year_filter, "All years"),
            (week_filter, "All weeks"),
        ):
            if combo.get() not in combo["values"]:
                combo.set(default)

    def refresh_table() -> None:
        rows_by_id.clear()
        table.delete(*table.get_children())
        for row in filter_rows():
            row_id = str(row["id"])
            rows_by_id[row_id] = row
            table.insert(
                "",
                END,
                iid=row_id,
                values=(
                    cell(row.get("competition_type")),
                    cell(row.get("year")),
                    cell(row.get("week")).zfill(2),
                    cell(row.get("status")),
                    cell(row.get("judge_x")),
                    cell(row.get("judge_y")),
                    cell(row.get("coordinate_invalid")).lower(),
                ),
            )
        log(log_box, f"Loaded {len(rows_by_id)} rows into the table.")

    def reload_rows() -> None:
        nonlocal rows
        rows = load_rows()
        update_filter_choices()
        refresh_table()

    def row_clicked(_event=None) -> None:
        selected = table.selection()
        selected_row_id.set(selected[0] if selected else "")
        row = current_row()
        update_detail_entries(row)
        show_preview(preview_label, row, preview_mode.get())

    def read_form() -> dict:
        try:
            return {
                "competition": competition_var.get().strip() or "dreamcar",
                "start_year": int(start_year_var.get()),
                "end_year": int(end_year_var.get()),
                "start_week": int(start_week_var.get()),
                "end_week": int(end_week_var.get()),
                "delay": float(delay_var.get()),
                "overwrite": overwrite_var.get(),
                "headed": headed_var.get(),
            }
        except ValueError as exc:
            raise ValueError("Years, weeks, and delay need to be numbers.") from exc

    def start_scrape() -> None:
        try:
            form = read_form()
        except ValueError as exc:
            messagebox.showerror("Input error", str(exc))
            return

        result_queue: queue.Queue = queue.Queue()
        log(log_box, "Starting scrape...")

        def progress(event: dict) -> None:
            if event.get("event") in {"entry-ok", "entry-missing", "entry-error", "entry-skip"}:
                result_queue.put(("progress", f"{event.get('code')}: {event.get('event')}"))

        def job() -> str:
            scraper = BotbScraper(
                DATA_DIR,
                delay=form["delay"],
                headless=not form["headed"],
                overwrite=form["overwrite"],
                competition=form["competition"],
                progress_callback=progress,
            )
            asyncio.run(
                scraper.scrape_range(
                    start_year=form["start_year"],
                    end_year=form["end_year"],
                    start_week=form["start_week"],
                    end_week=form["end_week"],
                )
            )
            return "Scrape finished."

        run_in_thread(job, result_queue)
        poll_thread(root, result_queue, lambda msg: (log(log_box, msg), reload_rows()), lambda err: log(log_box, err), lambda msg: log(log_box, msg))

    def rescrape_selected() -> None:
        row = current_row()
        if not row:
            messagebox.showinfo("No row", "Select a row first.")
            return
        start_year_var.set(cell(row.get("year")))
        end_year_var.set(cell(row.get("year")))
        start_week_var.set(cell(row.get("week")))
        end_week_var.set(cell(row.get("week")))
        competition_var.set(cell(row.get("competition")) or "dreamcar")
        overwrite_var.set(True)
        start_scrape()

    def install_requirements() -> None:
        # handy when I open the project on another machine
        command = [sys.executable, "-m", "pip", "install", "-e", "."]
        log(log_box, "Installing project requirements...")
        result = subprocess.run(command, cwd=PROJECT_DIR, capture_output=True, text=True, check=False)
        if result.stdout:
            log(log_box, result.stdout.strip())
        if result.stderr:
            log(log_box, result.stderr.strip())
        log(log_box, f"Install command finished with code {result.returncode}.")

    def preset_2017() -> None:
        competition_var.set("dreamcar")
        start_year_var.set("17")
        end_year_var.set("17")
        start_week_var.set("1")
        end_week_var.set("52")

    def preset_current() -> None:
        competition_var.set("dreamcar")
        start_year_var.set("26")
        end_year_var.set("26")
        start_week_var.set("1")
        end_week_var.set("52")

    # only keeping the scraper action here for this stage
    make_button(preview_actions, "Rescrape selected entry", rescrape_selected, 30).grid(
        row=1, column=0, columnspan=3, sticky="ew", padx=6, pady=4
    )
    for col in range(3):
        preview_actions.grid_columnconfigure(col, weight=1)

    make_button(button_grid, "Install requirements", install_requirements, 32).grid(row=0, column=0, columnspan=4, sticky="ew", padx=4, pady=4)
    make_button(button_grid, "Start scrape", start_scrape, 32).grid(row=1, column=0, columnspan=4, sticky="ew", padx=4, pady=4)
    Label(button_grid, text="Next step: add YOLO player detection after the scraper is stable", bg=PANEL, fg=MUTED).grid(
        row=2, column=0, columnspan=4, sticky="w", padx=4, pady=(10, 4)
    )
    Label(button_grid, text="Older-year presets", bg=PANEL, fg=TEXT).grid(row=3, column=0, sticky="w", padx=4, pady=(8, 4))
    make_button(button_grid, "Dream Car 2017", preset_2017, 18).grid(row=4, column=0, sticky="ew", padx=4, pady=4)
    make_button(button_grid, "Dream Car 2026", preset_current, 18).grid(row=4, column=1, sticky="ew", padx=4, pady=4)
    for col in range(4):
        button_grid.grid_columnconfigure(col, weight=1)

    for widget in (search_entry, competition_filter, status_filter, coordinate_filter, year_filter, week_filter):
        if isinstance(widget, Entry):
            widget.bind("<KeyRelease>", lambda _event: refresh_table())
        else:
            widget.bind("<<ComboboxSelected>>", lambda _event: refresh_table())

    table.bind("<<TreeviewSelect>>", row_clicked)
    update_detail_entries(None)
    reload_rows()
    return root


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    root = build_gui()
    root.mainloop()

